"""Generated pipeline for task 'dialogue_next_turn' (sample 4).

TASK: From a customer conversation, work out the requested action and the details still needed, then write the next agent turn.
"""

from pttai import AgentNode, DecisionNode, ConditionNode, AgenticGraph


def build_graph(llm):
    # 1) Frame: extract the requested action + what details are missing.
    frame = AgentNode(
        name="frame",
        llm=llm,
        node_prompt=(
            "You are an assistant that plans the next customer-facing action.\n\n"
            "Given the conversation history, do TWO things:\n"
            "1) Determine the requested action (a short verb+noun phrase).\n"
            "2) List the details still needed to complete that action.\n\n"
            "Rules:\n"
            "- Be concrete and operational.\n"
            "- If the customer already provided everything needed, state 'No missing details'.\n"
            "- Output your result as normal text.\n"
            "- Do NOT ask questions yet; just identify missing details.\n"
            "Conversation:\n{messages}"
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # 2) Decide whether we should ask for missing details now.
    need_details = DecisionNode(
        name="need_details",
        llm=llm,
        node_prompt=(
            "Decide whether the assistant must ask the customer for missing details.\n\n"
            "Choose ONE:\n"
            "- 'ask' if there are still missing details required to complete the requested action\n"
            "- 'proceed' if there are no missing details\n\n"
            "Base your decision on the framed analysis in the conversation messages."
        ),
        choices=["ask", "proceed"],
        reads=["messages"],
        writes=None,  # DecisionNode writes its choice to the built-in `decision` field
    )

    # 3a) If missing details exist: ask a concise next turn (questions).
    ask_missing = AgentNode(
        name="ask_missing",
        llm=llm,
        node_prompt=(
            "You are the assistant.\n"
            "Write the NEXT customer-facing message.\n\n"
            "The frame indicates the requested action and the details still needed.\n"
            "If there are missing details, ask the minimum set of specific questions to unblock the request.\n"
            "Guidelines:\n"
            "- Keep it concise.\n"
            "- Ask numbered questions if multiple.\n"
            "- Do not include any extra analysis.\n"
            "- Do not re-state the entire conversation.\n\n"
            "Conversation / frame context:\n{messages}"
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # 3b) If no missing details: proceed with the requested action / next step.
    proceed = AgentNode(
        name="proceed",
        llm=llm,
        node_prompt=(
            "You are the assistant.\n"
            "The customer already provided enough information to complete the requested action.\n\n"
            "Write the NEXT customer-facing message that:\n"
            "- Confirms what you will do (requested action),\n"
            "- Provides the completed next step / deliverable / instructions,\n"
            "- Includes any relevant assumptions explicitly (only if necessary).\n\n"
            "Do NOT ask questions unless absolutely required.\n\n"
            "Conversation / frame context:\n{messages}"
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # 4) Safety gate (optional deterministic): ensure the next turn isn't empty.
    # This condition checks the last assistant message exists in `messages`.
    def non_empty_last_turn(state):
        msgs = state.get("messages", [])
        # Accept if there is at least one message and the last content is non-empty.
        if not msgs:
            return "bad"
        last = getattr(msgs[-1], "content", None)
        if isinstance(last, str) and last.strip():
            return "ok"
        return "bad"

    gate = ConditionNode(
        name="gate",
        condition=non_empty_last_turn,
        choices=["ok", "bad"],
        reads=["messages"],
    )

    # If bad, repair with a short clarification.
    repair = AgentNode(
        name="repair",
        llm=llm,
        node_prompt=(
            "Write a short corrected next message to the customer. "
            "Ensure it is non-empty and helpful, based on the existing conversation context:\n{messages}"
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # End nodes: ask_missing or proceed or repair.
    # (gate routes to repair only if needed)
    # Wiring:
    # frame -> need_details
    # need_details['ask'] -> ask_missing -> gate -> (ok => ask_missing ends, bad => repair)
    # need_details['proceed'] -> proceed -> gate -> (ok => proceed ends, bad => repair)
    #
    # Note: a choice must be routed to concrete nodes; gate is shared downstream.

    need_details["ask"] > ask_missing
    need_details["proceed"] > proceed

    ask_missing > gate
    proceed > gate

    gate["ok"] > ask_missing  # terminal behavior: for ok, we can just end at the node reached
    gate["bad"] > repair

    return AgenticGraph(
        start_node=frame,
        end_nodes={ask_missing, proceed, repair},
    )
