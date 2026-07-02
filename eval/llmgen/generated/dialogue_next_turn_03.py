"""Generated pipeline for task 'dialogue_next_turn' (sample 3).

TASK: From a customer conversation, work out the requested action and the details still needed, then write the next agent turn.
"""

from pttai import AgentNode, DecisionNode, ConditionNode, AgenticGraph, fanout, AgenticState


def build_graph(llm):
    # 1) Turn the conversation into a single requested action + extracted details.
    frame = AgentNode(
        name="frame",
        llm=llm,
        reads=["messages"],
        node_prompt=(
            "You are an operations analyst for a customer support system.\n"
            "Read the conversation so far and produce:\n"
            "1) The requested action as a short imperative (e.g., 'reset password', 'upgrade plan', 'open ticket').\n"
            "2) The details already provided that are relevant to executing that action.\n"
            "3) The specific details still needed (make them concrete and actionable).\n"
            "4) If something is ambiguous, state the ambiguity as a question.\n"
            "Write your output as clear bullet points under headings: "
            "Requested action, Provided details, Missing details, Questions.\n"
            "Do not invent missing customer-specific facts."
        ),
        writes={"analysis": str},
    )

    # 2) Classify what the next step should be.
    decide_next = DecisionNode(
        name="decide_next",
        llm=llm,
        node_prompt=(
            "Based on the analysis produced earlier, decide what the next agent turn should do.\n"
            "Choose exactly one option.\n\n"
            "If the customer has provided enough info to proceed, choose 'proceed'.\n"
            "If important required details are missing, choose 'ask_missing'.\n"
            "If the request is not actionable / requires human escalation, choose 'escalate'."
        ),
        choices=["proceed", "ask_missing", "escalate"],
        input_field="messages",
        reads=["analysis"],
    )

    # 3a) If proceeding: produce the next turn that summarizes action and confirms.
    proceed = AgentNode(
        name="proceed",
        llm=llm,
        reads=["messages", "analysis"],
        node_prompt=(
            "You are the support agent who will write the next customer-facing message.\n\n"
            "Using the earlier analysis, draft the next agent turn that:\n"
            "- Restates the requested action succinctly.\n"
            "- Summarizes the key details the customer already provided.\n"
            "- States the next steps you will take (high level) without claiming you already completed them.\n"
            "- Asks for any final confirmation ONLY if needed; otherwise, does not ask extra questions.\n"
            "Tone: professional, concise, helpful.\n"
            "Output only the customer-facing message."
        ),
        writes=["messages"],
    )

    # 3b) If missing details: ask targeted questions, not a wall of text.
    ask_missing = AgentNode(
        name="ask_missing",
        llm=llm,
        reads=["messages", "analysis"],
        node_prompt=(
            "You are the support agent who will write the next customer-facing message.\n\n"
            "Using the earlier analysis, draft a message that:\n"
            "- Clearly lists the missing details you need, as 3-7 targeted questions.\n"
            "- For each question, specify what format/example the customer can use.\n"
            "- Reassure the customer that once they provide these details, you can proceed.\n"
            "Tone: empathetic, efficient.\n"
            "Output only the customer-facing message."
        ),
        writes=["messages"],
    )

    # 3c) If escalation: draft a message indicating handoff to a human queue.
    escalate = AgentNode(
        name="escalate",
        llm=llm,
        reads=["messages", "analysis"],
        node_prompt=(
            "You are the support agent who will write the next customer-facing message.\n\n"
            "Based on the earlier analysis, draft a message that:\n"
            "- Explains that this request needs human review/escalation.\n"
            "- Summarizes what you have understood so far (no invented facts).\n"
            "- If appropriate, asks for any critical info that will help the human reviewer (1-3 max).\n"
            "- Sets expectations about next steps (e.g., 'a specialist will follow up').\n"
            "Tone: professional, calm.\n"
            "Output only the customer-facing message."
        ),
        writes=["messages"],
    )

    # 4) Optional guard: if the analysis indicates 'no action' / spam / unrelated, route to ask_missing anyway
    # via an LLM-free deterministic check on the analysis text.
    def is_no_action(state):
        text = state.get("analysis", "").lower()
        # Heuristic: if frame indicates no clear requested action.
        # If uncertain, let decide_next handle it.
        return "clarify" if ("no clear requested action" in text or "unrelated" in text) else "ok"

    clarify_guard = ConditionNode(
        name="clarify_guard",
        condition=is_no_action,
        choices=["ok", "clarify"],
        reads=["analysis"],
    )

    # Reuse ask_missing for clarification; if clarify, ask_missing will ask questions anyway.
    # Wire clarify_guard to a handler (clarify -> ask_missing, ok -> decide_next).
    # Note: DecisionNode wiring must be via choice indexing.
    clarify_to = ask_missing
    ok_to = decide_next

    # 5) Wiring
    # frame -> clarify_guard -> (either proceed/ask/escalate handled by decide_next, or go directly to ask_missing)
    frame > clarify_guard
    clarify_guard["clarify"] > clarify_to
    clarify_guard["ok"] > ok_to

    decide_next["proceed"] > proceed
    decide_next["ask_missing"] > ask_missing
    decide_next["escalate"] > escalate

    # End nodes: whichever terminal handler we reach.
    return AgenticGraph(start_node=frame, end_nodes={proceed, ask_missing, escalate})
