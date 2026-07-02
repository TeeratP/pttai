"""Generated pipeline for task 'dialogue_next_turn' (sample 2).

TASK: From a customer conversation, work out the requested action and the details still needed, then write the next agent turn.
"""

from typing import Dict, Any, List

from pttai import AgentNode, DecisionNode, AgenticGraph, ConditionNode, fanout
from pttai import HumanNode, AgenticState


def build_graph(llm):
    """
    One pipeline:
      1) Extract requested action + required missing details from the conversation.
      2) Decide what the next step should be (ask clarifying questions vs proceed).
      3) If clarifying: draft a concise next agent turn asking only what’s missing.
         If proceed: draft the next agent turn executing/confirming the action (without inventing missing details).
    """

    # --- Step 1: Frame the task as a structured extraction problem ---
    extract = AgentNode(
        name="extract",
        llm=llm,
        node_prompt=(
            "You are an assistant that extracts requests from a customer conversation.\n"
            "Given the conversation messages, determine:\n"
            "1) The requested action (a short verb phrase), e.g. 'reset password', 'upgrade plan', "
            "'schedule a demo', 'change shipping address'.\n"
            "2) Concrete details mentioned for that action.\n"
            "3) Exactly what details are still needed to complete the action.\n"
            "4) If the request is ambiguous or contradictory, list the specific points that must be clarified.\n\n"
            "Output in a single message using the following labeled format (must keep labels):\n"
            "Action: <...>\n"
            "DetailsProvided: <...>\n"
            "DetailsNeeded: <comma-separated questions or specific missing items>\n"
            "AssumptionsToAvoid: <anything you must NOT assume>\n"
            "NextAgentGoal: <one sentence: what you will do in the next agent turn>\n"
        ),
        # Use the conversation as history; AgenticState already uses `messages`.
        reads=["messages"],
        writes=["messages"],
    )

    # --- Step 2: Decide whether we can proceed or must ask clarifying questions ---
    # DecisionNode writes its choice into the built-in `decision` field.
    decide_clarify_or_proceed = DecisionNode(
        name="decide_clarify_or_proceed",
        llm=llm,
        node_prompt=(
            "You will choose what the next agent turn should do.\n"
            "You have access to the full conversation messages and the extracted analysis.\n\n"
            "Choose exactly one:\n"
            "1) 'clarify' if key details are missing or the request is ambiguous.\n"
            "2) 'proceed' if enough details exist to safely execute/confirm the action without inventing missing info.\n\n"
            "Be strict: if anything essential is missing, choose 'clarify'."
        ),
        choices=["clarify", "proceed"],
        reads=["messages"],
        # (DecisionNode always routes immediately after it writes its choice to `decision`.)
        # No writes/output_field here per API.
    )

    # --- Step 3a: Clarifying questions turn ---
    ask_clarification = AgentNode(
        name="ask_clarification",
        llm=llm,
        node_prompt=(
            "Write the next agent turn to the customer.\n\n"
            "Use the conversation context. You must:\n"
            "- Restate the requested action in 1 short sentence.\n"
            "- Ask only for the missing details (from the extraction), as a short numbered list.\n"
            "- Keep it customer-friendly and concise.\n"
            "- Do NOT add extra questions beyond what is missing.\n"
            "- Do NOT assume missing information.\n\n"
            "Constraints:\n"
            "- If there are multiple possible interpretations, ask targeted questions to disambiguate.\n"
            "- End with an invitation to reply with the requested items.\n"
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # --- Step 3b: Proceed/confirm turn (without inventing missing details) ---
    proceed_action = AgentNode(
        name="proceed_action",
        llm=llm,
        node_prompt=(
            "Write the next agent turn to the customer to proceed.\n\n"
            "Use the conversation context and extracted analysis.\n"
            "You must:\n"
            "- Confirm the requested action.\n"
            "- Summarize the key details you will use.\n"
            "- If any details are still missing, do NOT execute; instead, briefly note what is missing and ask a minimal follow-up.\n"
            "- Otherwise, provide the next practical step (e.g., confirm changes, explain what happens next, or ask for any final non-blocking confirmation).\n"
            "- Keep it concise and action-oriented.\n"
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # --- Step 4: Optional polish step (ensures the final message is the last assistant response) ---
    polish = AgentNode(
        name="polish",
        llm=llm,
        node_prompt=(
            "Polish the latest draft agent message for clarity, tone, and concision.\n"
            "Return ONLY the final customer-facing message content (no extra formatting outside the message).\n"
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # Routing: decide_clarify_or_proceed["clarify"] -> ask_clarification, ["proceed"] -> proceed_action
    # Then both go to polish.
    graph = extract > decide_clarify_or_proceed
    graph = graph["clarify"] > ask_clarification
    graph = graph > polish

    # Note: DecisionNode wiring must be explicit per choice; we continue wiring for the other branch.
    decide_clarify_or_proceed["proceed"] > proceed_action > polish

    return AgenticGraph(
        start_node=extract,
        end_nodes={polish},
        # No explicit schema needed; default AgenticState supports messages/log/decision/token.
        # This pipeline reads `messages` from input shorthand at invoke-time.
    )
