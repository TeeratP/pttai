"""Generated pipeline for task 'triage' (sample 2).

TASK: Triage an incoming support message: classify it as billing, technical, or other, then route it to a handler that drafts the appropriate reply.
"""

from pttai import AgentNode, DecisionNode, AgenticGraph


def build_graph(llm):
    # 1) Classify the incoming support message into a constrained label.
    triage = DecisionNode(
        name="triage",
        llm=llm,
        node_prompt=(
            "You triage customer support messages.\n"
            "Classify the message as exactly ONE of: billing, technical, other.\n"
            "Rules:\n"
            "- billing: charges, invoices, refunds, payment failures, subscriptions/plan changes.\n"
            "- technical: bugs, errors, broken features, integrations, outages, how to fix.\n"
            "- other: anything else (general questions, account/profile, feedback, escalation).\n"
            "Message:\n{messages}\n\n"
            "Return only the correct label."
        ),
        choices=["billing", "technical", "other"],
        input_field="messages",
    )

    # 2) Draft an appropriate reply per category.
    billing_reply = AgentNode(
        name="billing_reply",
        llm=llm,
        node_prompt=(
            "You are a support agent specializing in billing.\n"
            "Draft a helpful, concise reply to the customer.\n\n"
            "Requirements:\n"
            "- Acknowledge the issue politely.\n"
            "- Ask for the minimum necessary details (e.g., email/account ID, invoice date, last 4 digits) if missing.\n"
            "- Provide safe next steps and relevant policy guidance at a high level.\n"
            "- Avoid promising refunds you cannot verify.\n\n"
            "Customer message:\n{messages}\n"
        ),
        reads=["messages"],
        writes=["messages"],
    )

    technical_reply = AgentNode(
        name="technical_reply",
        llm=llm,
        node_prompt=(
            "You are a support agent specializing in technical troubleshooting.\n"
            "Draft a helpful, step-by-step reply to the customer.\n\n"
            "Requirements:\n"
            "- Identify the likely cause category (without claiming certainty).\n"
            "- Request key diagnostics (error message, steps to reproduce, device/browser/app version, logs if applicable).\n"
            "- Provide 2-4 concrete troubleshooting steps.\n"
            "- If the issue suggests an outage, suggest checking status and gathering evidence.\n\n"
            "Customer message:\n{messages}\n"
        ),
        reads=["messages"],
        writes=["messages"],
    )

    other_reply = AgentNode(
        name="other_reply",
        llm=llm,
        node_prompt=(
            "You are a general support agent.\n"
            "Draft a helpful reply to the customer.\n\n"
            "Requirements:\n"
            "- Clarify what they need (if ambiguous) with 1-2 targeted questions.\n"
            "- Provide general guidance or the next best action.\n"
            "- If it seems like it should be an escalation, propose that as the next step.\n\n"
            "Customer message:\n{messages}\n"
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # 3) Wire the LLM classification to the correct drafting handler.
    triage["billing"] > billing_reply
    triage["technical"] > technical_reply
    triage["other"] > other_reply

    return AgenticGraph(start_node=triage, end_nodes={billing_reply, technical_reply, other_reply})
