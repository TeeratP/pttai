"""Generated pipeline for task 'dialogue_next_turn' (sample 0).

TASK: From a customer conversation, work out the requested action and the details still needed, then write the next agent turn.
"""

from pttai import AgentNode, ConditionNode, DecisionNode, AgenticGraph, fanout


def build_graph(llm):
    # 1) Extract what the customer is asking for + what must be clarified
    #    (done with an LLM; we only need to wire outputs via state keys)
    extract = AgentNode(
        name="extract",
        llm=llm,
        node_prompt=(
            "You are a service coordinator.\n"
            "Given the customer's conversation, produce:\n"
            "1) action: the requested next action (short phrase, e.g., 'reset password', "
            "'upgrade plan', 'cancel subscription', 'book a consultation', etc.).\n"
            "2) details_needed: a concise list of concrete missing details required to proceed.\n"
            "3) intent_summary: one sentence summarizing the customer's goal.\n"
            "\n"
            "Return the result as a JSON-like object in plain text with labeled sections.\n"
            "Use this exact format:\n"
            "action: <...>\n"
            "details_needed: <item1; item2; item3 or 'none'>\n"
            "intent_summary: <...>"
        ),
        # We treat incoming input as conversation history in `messages`.
        # Default behavior appends the LLM reply to `messages`.
        reads=["messages"],
        writes=["messages"],
    )

    # 2) Decide whether we already have enough details to take action, or must ask clarifying questions.
    #    The decision is explicitly constrained to valid branches.
    decide_enough = DecisionNode(
        name="decide_enough",
        llm=llm,
        node_prompt=(
            "Decide whether the extracted plan has enough details to proceed with the requested action.\n"
            "If details_needed is 'none' (or equivalent), choose 'proceed'.\n"
            "Otherwise choose 'ask_clarifying_questions'.\n\n"
            "You will see the conversation plus the extraction output in messages."
        ),
        choices=["proceed", "ask_clarifying_questions"],
        reads=["messages"],
    )

    # 3a) If we have enough information, write the next agent turn to proceed.
    proceed_turn = AgentNode(
        name="proceed_turn",
        llm=llm,
        node_prompt=(
            "You are the next-turn agent reply.\n"
            "Using the customer's conversation and the extraction output, do BOTH:\n"
            "1) Confirm the action you will take.\n"
            "2) Provide the next steps (or instructions) the customer should follow.\n"
            "Include any assumptions explicitly and keep it concise.\n\n"
            "Write the agent's next message only."
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # 3b) If we need more info, ask targeted clarifying questions.
    clarifying_turn = AgentNode(
        name="clarifying_turn",
        llm=llm,
        node_prompt=(
            "You are the next-turn agent reply.\n"
            "Using the customer's conversation and the extraction output, ask only the essential clarifying questions.\n"
            "Rules:\n"
            "- Ask 1-5 questions maximum.\n"
            "- Each question must be directly tied to items in details_needed.\n"
            "- Keep questions short and customer-friendly.\n"
            "Write the agent's next message only."
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # 4) Optional final polish / ensure correct handoff tone.
    #    (Deterministic check for empty messages is not robust; we instead use an LLM polish with a single turn.)
    polish = AgentNode(
        name="polish",
        llm=llm,
        node_prompt=(
            "Polish the last agent message for clarity and helpfulness.\n"
            "Keep meaning unchanged; do not add new requirements beyond what was asked.\n"
            "Return only the final polished message text."
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # Wire: extract -> decide -> (proceed or ask) -> polish
    # The decision branches are indexed by choice labels.
    decide_enough["proceed"] > proceed_turn
    decide_enough["ask_clarifying_questions"] > clarifying_turn

    # Keep wiring linear after join at polish.
    extract > decide_enough
    proceed_turn > polish
    clarifying_turn > polish

    return AgenticGraph(start_node=extract, end_nodes={polish})
