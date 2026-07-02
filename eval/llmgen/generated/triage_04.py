"""Generated pipeline for task 'triage' (sample 4).

TASK: Triage an incoming support message: classify it as billing, technical, or other, then route it to a handler that drafts the appropriate reply.
"""

from pttai import AgentNode, DecisionNode, AgenticGraph, ConditionNode
from pttai import AgenticState


def build_graph(llm):
    # 1) Ensure we have a stable input: always append the incoming support message.
    prep = AgentNode(
        name="prep",
        llm=llm,
        node_prompt=(
            "You are a support triage intake assistant.\n"
            "Convert the incoming support message into a clear, single user request.\n"
            "Do NOT draft the final reply yet—only rewrite/clean up the request for downstream steps.\n"
            "If the message is already clear, keep it essentially the same."
        ),
        writes=["messages"],
    )

    # 2) LLM classification (constrained to exact choices).
    classify = DecisionNode(
        name="classify",
        llm=llm,
        node_prompt=(
            "Classify the user's support request into exactly one category.\n"
            "Categories:\n"
            "- billing: invoices, payments, refunds, charges, subscription changes, pricing.\n"
            "- technical: bugs, errors, integrations, outages, performance, how-to troubleshooting.\n"
            "- other: everything else.\n\n"
            "Return only the category."
        ),
        choices=["billing", "technical", "other"],
        input_field="messages",
    )

    # 3) Optional deterministic guard: if the message seems empty/meaningless, route to "other".
    is_meaningful = ConditionNode(
        name="is_meaningful",
        condition=lambda state: (
            "meaningful"
            if any(
                (m.content or "").strip()
                for m in (state.get("messages") or [])
                if hasattr(m, "content")
            )
            else "empty"
        ),
        choices=["meaningful", "empty"],
        reads=["messages"],
    )

    # 4) Draft category-specific responses.
    billing_draft = AgentNode(
        name="billing_draft",
        llm=llm,
        node_prompt=(
            "You are a billing support agent.\n"
            "Draft a helpful, concise reply addressing the customer's concern.\n"
            "Include:\n"
            "- A brief acknowledgement\n"
            "- What info you need from the user (e.g., invoice date/last4/plan)\n"
            "- Clear next steps\n"
            "If the user asks for a refund or cancellation, ask for the minimum details needed to investigate.\n"
            "Do not mention internal policies; be polite and actionable."
        ),
        writes=["messages"],
    )

    technical_draft = AgentNode(
        name="technical_draft",
        llm=llm,
        node_prompt=(
            "You are a technical support agent.\n"
            "Draft a helpful, concise troubleshooting reply.\n"
            "Include:\n"
            "- A brief acknowledgement\n"
            "- 2-4 targeted questions or checks to narrow the issue\n"
            "- Step-by-step instructions where appropriate\n"
            "- Any safe diagnostics (logs, error codes, environment, reproduction)\n"
            "If the issue seems like an outage, suggest basic mitigations and ask for affected details.\n"
            "Be clear and non-technical where possible."
        ),
        writes=["messages"],
    )

    other_draft = AgentNode(
        name="other_draft",
        llm=llm,
        node_prompt=(
            "You are a general support agent.\n"
            "Draft a helpful, concise reply.\n"
            "If the request is ambiguous, ask clarifying questions.\n"
            "If it looks like it might fit billing or technical, lightly suggest what details to provide.\n"
            "Be polite and actionable."
        ),
        writes=["messages"],
    )

    # 5) Routing wiring
    # Prep -> meaningful? -> (classify) -> category drafts -> end
    meaningful_router = is_meaningful
    prep > meaningful_router

    meaningful_router["empty"] > other_draft
    meaningful_router["meaningful"] > classify

    classify["billing"] > billing_draft
    classify["technical"] > technical_draft
    classify["other"] > other_draft

    return AgenticGraph(
        start_node=prep,
        end_nodes={billing_draft, technical_draft, other_draft},
    )
