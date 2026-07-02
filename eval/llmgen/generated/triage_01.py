"""Generated pipeline for task 'triage' (sample 1).

TASK: Triage an incoming support message: classify it as billing, technical, or other, then route it to a handler that drafts the appropriate reply.
"""

from pttai import AgentNode, DecisionNode, AgenticGraph


def build_graph(llm):
    triage = DecisionNode(
        name="triage",
        llm=llm,
        node_prompt=(
            "You are a support triage classifier. "
            "Classify the incoming support message into exactly ONE category: "
            "billing, technical, or other. "
            "Return only the category via the structured routing choices."
        ),
        choices=["billing", "technical", "other"],
        input_field="messages",
    )

    draft_billing = AgentNode(
        name="draft_billing",
        llm=llm,
        node_prompt=(
            "You are a billing support specialist. "
            "Draft a helpful, concise reply to the customer support message. "
            "If key details are missing (e.g., plan, invoice id, last 4 digits, account email), "
            "ask for them politely. "
            "Include next steps and a friendly closing."
        ),
        tools=None,
        reads=["messages"],
        writes=["messages"],
    )

    draft_technical = AgentNode(
        name="draft_technical",
        llm=llm,
        node_prompt=(
            "You are a technical support specialist. "
            "Draft a helpful, concise reply to the customer support message. "
            "If troubleshooting steps are needed, propose a short ordered list. "
            "Ask for relevant diagnostics (error message, device/browser, steps to reproduce, logs). "
            "Keep it actionable and non-judgmental."
        ),
        tools=None,
        reads=["messages"],
        writes=["messages"],
    )

    draft_other = AgentNode(
        name="draft_other",
        llm=llm,
        node_prompt=(
            "You are a general support specialist. "
            "Draft a helpful, concise reply to the customer support message. "
            "Acknowledge the issue, ask clarifying questions, and suggest next steps. "
            "If it seems like it might be billing or technical, gently indicate what info would be needed."
        ),
        tools=None,
        reads=["messages"],
        writes=["messages"],
    )

    triage["billing"] > draft_billing
    triage["technical"] > draft_technical
    triage["other"] > draft_other

    return AgenticGraph(start_node=triage, end_nodes={draft_billing, draft_technical, draft_other})
