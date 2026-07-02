"""Generated pipeline for task 'triage' (sample 0).

TASK: Triage an incoming support message: classify it as billing, technical, or other, then route it to a handler that drafts the appropriate reply.
"""

from pttai import AgentNode, DecisionNode, AgenticGraph
from pttai import AgenticState


def build_graph(llm):
    classify_prompt = (
        "You are a support triage classifier.\n"
        "Classify the user's incoming support message into exactly one category:\n"
        "- billing: questions about invoices, charges, refunds, subscriptions, payment failures\n"
        "- technical: bugs, errors, troubleshooting, integration issues, how-to for product features\n"
        "- other: everything else (account issues that aren't billing, general questions, feedback, etc.)\n\n"
        "Message:\n{messages}"
    )

    classify = DecisionNode(
        name="classify",
        llm=llm,
        node_prompt=classify_prompt,
        choices=["billing", "technical", "other"],
        input_field="messages",
        reads=["messages"],
    )

    billing_draft = AgentNode(
        name="billing_draft",
        llm=llm,
        reads=["messages"],
        node_prompt=(
            "Draft a helpful, concise customer support reply for a BILLING issue.\n"
            "Requirements:\n"
            "- Be empathetic and professional.\n"
            "- Ask for the minimum necessary details (e.g., last 4 digits / invoice id / email) if missing.\n"
            "- Provide 2-4 actionable steps.\n"
            "- Avoid making up policy specifics; if uncertain, suggest checking billing settings or contacting support with details.\n\n"
            "Incoming message:\n{messages}"
        ),
        writes=["messages"],
    )

    technical_draft = AgentNode(
        name="technical_draft",
        llm=llm,
        reads=["messages"],
        node_prompt=(
            "Draft a helpful, concise customer support reply for a TECHNICAL issue.\n"
            "Requirements:\n"
            "- Be empathetic and professional.\n"
            "- Diagnose likely causes based on the message.\n"
            "- Provide 3-6 troubleshooting steps.\n"
            "- Ask for relevant details if needed (error text, environment, steps to reproduce).\n\n"
            "Incoming message:\n{messages}"
        ),
        writes=["messages"],
    )

    other_draft = AgentNode(
        name="other_draft",
        llm=llm,
        reads=["messages"],
        node_prompt=(
            "Draft a helpful, concise customer support reply for an OTHER issue.\n"
            "Requirements:\n"
            "- Be empathetic and professional.\n"
            "- Clarify the request and provide appropriate next steps.\n"
            "- If it's a question, answer it; if it's feedback, acknowledge and guide on how it will be handled.\n\n"
            "Incoming message:\n{messages}"
        ),
        writes=["messages"],
    )

    classify["billing"] > billing_draft
    classify["technical"] > technical_draft
    classify["other"] > other_draft

    return AgenticGraph(start_node=classify, end_nodes={billing_draft, technical_draft, other_draft})
