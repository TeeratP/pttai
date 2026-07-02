"""Generated pipeline for task 'triage' (sample 3).

TASK: Triage an incoming support message: classify it as billing, technical, or other, then route it to a handler that drafts the appropriate reply.
"""

from pttai import AgentNode, DecisionNode, AgenticGraph, ConditionNode


def build_graph(llm):
    classify = DecisionNode(
        name="classify",
        llm=llm,
        node_prompt=(
            "You triage incoming support messages. "
            "Classify the user's message into exactly one category:\n"
            "- billing: invoices, charges, refunds, payment failures, subscription status\n"
            "- technical: bugs, errors, integration issues, how-to, troubleshooting\n"
            "- other: anything else (general questions, feedback, account questions not covered above)\n"
            "Return only the category label."
        ),
        choices=["billing", "technical", "other"],
        input_field="messages",
    )

    # Optional guardrail: if the input is empty/whitespace, route to "other".
    # (Runs deterministically; still keeps the LLM classifier as the main triage.)
    has_content = ConditionNode(
        name="has_content",
        condition=lambda state: "yes" if str(state["messages"][-1].content).strip() else "no",
        choices=["yes", "no"],
        reads=["messages"],
    )
    classify_or_other = AgentNode(
        name="classify_or_other",
        llm=llm,
        node_prompt=(
            "Classify the following support message into one category label: "
            "'billing', 'technical', or 'other'.\n"
            "If the message is empty or meaningless, return 'other'.\n"
            "Message:\n{message_text}"
        ),
        reads=["message_text"],
        writes={"category": str},
    )

    billing_handler = AgentNode(
        name="billing_handler",
        llm=llm,
        node_prompt=(
            "You are a support agent. Draft a helpful, concise reply for a billing issue.\n"
            "Requirements:\n"
            "1) Acknowledge the problem.\n"
            "2) Ask up to 3 targeted questions needed to resolve it (e.g., invoice date, last 4 digits, plan).\n"
            "3) Provide safe next steps and troubleshooting that doesn't require sensitive data.\n"
            "4) Include a brief disclaimer that billing-specific refunds depend on verification.\n"
            "Do not mention internal policies.\n"
            "Original message:\n{message_text}"
        ),
        reads=["message_text"],
    )

    technical_handler = AgentNode(
        name="technical_handler",
        llm=llm,
        node_prompt=(
            "You are a support agent. Draft a helpful, concise reply for a technical issue.\n"
            "Requirements:\n"
            "1) Restate the issue.\n"
            "2) Provide step-by-step troubleshooting (3-6 steps).\n"
            "3) Ask for key diagnostics (e.g., error message, environment, versions) if missing.\n"
            "4) Suggest how to reproduce and how to confirm the fix.\n"
            "Original message:\n{message_text}"
        ),
        reads=["message_text"],
    )

    other_handler = AgentNode(
        name="other_handler",
        llm=llm,
        node_prompt=(
            "You are a support agent. Draft a helpful, concise reply for a non-billing, non-technical issue.\n"
            "Requirements:\n"
            "1) Address the user's request or concern.\n"
            "2) If information is missing, ask 1-3 clarifying questions.\n"
            "3) Provide next steps and set expectations.\n"
            "Original message:\n{message_text}"
        ),
        reads=["message_text"],
    )

    # Create a scalar 'message_text' from the latest user message content for easy prompt interpolation.
    extract_text = AgentNode(
        name="extract_text",
        llm=llm,
        node_prompt=(
            "Extract the latest user message content as plain text.\n"
            "If messages contain multiple roles, use the last message content.\n"
            "Return only the extracted text."
        ),
        reads=["messages"],
        writes={"message_text": str},
    )

    # Main route based on classifier output.
    # Note: DecisionNode writes to the built-in `decision` field; we route by indexing choices.
    router = DecisionNode(
        name="router",
        llm=llm,
        node_prompt=(
            "Classify the support message into exactly one label: "
            "'billing', 'technical', or 'other'. Return only one of the labels."
        ),
        choices=["billing", "technical", "other"],
        input_field="messages",
    )

    # Wire: extract_text -> has_content? -> router, then handlers.
    # If empty, route to other_handler; else use router's structured decision.
    start = extract_text

    # Use DecisionNode router for the real classification; has_content is a deterministic fallback.
    start > has_content
    has_content["no"] > other_handler
    has_content["yes"] > router

    router["billing"] > billing_handler
    router["technical"] > technical_handler
    router["other"] > other_handler

    return AgenticGraph(start_node=start, end_nodes={billing_handler, technical_handler, other_handler})
