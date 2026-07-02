"""Generated pipeline for task 'escalation' (sample 2).

TASK: Read a support thread, judge how severe it is, then either auto-resolve it, ask the customer a clarifying question, or escalate it to a human agent.
"""

from pttai import AgentNode, DecisionNode, ConditionNode, HumanNode, AgenticGraph


def build_graph(llm):
    # --- 1) Frame the thread into a severity + desired outcome ---
    frame = AgentNode(
        name="frame",
        llm=llm,
        node_prompt=(
            "You are a support triage analyst.\n"
            "Given the support thread, produce:\n"
            "1) severity: one of [low, medium, high]\n"
            "2) resolution_path: one of [auto_resolve, clarify, escalate]\n"
            "3) auto_resolution_candidate: concise steps if auto_resolve is appropriate, otherwise ''\n"
            "4) clarifying_question: the single best question to ask if clarify is appropriate, otherwise ''\n"
            "5) escalation_reason: why it must escalate if escalate is appropriate, otherwise ''\n\n"
            "Heuristics:\n"
            "- high: security/privacy issues, billing charge disputes, account lockouts, legal/compliance, repeated failures, or user reports of major outages.\n"
            "- medium: confusing errors, missing features, non-critical bugs, or recurring workflow issues that require basic info.\n"
            "- low: typos, how-to questions, minor usability questions, or well-specified issues.\n"
            "Output must be a single message with clearly labeled fields exactly:\n"
            "severity: <low|medium|high>\n"
            "resolution_path: <auto_resolve|clarify|escalate>\n"
            "auto_resolution_candidate: <text or empty string>\n"
            "clarifying_question: <text or empty string>\n"
            "escalation_reason: <text or empty string>\n"
        ),
    )

    # --- 2) Route with an LLM DecisionNode constrained to valid outcomes ---
    route = DecisionNode(
        name="route",
        llm=llm,
        node_prompt=(
            "Decide the next action based on the framed triage message.\n"
            "Choose exactly one:\n"
            "- auto_resolve: you can confidently answer/fix without additional customer info\n"
            "- clarify: need one specific piece of information to proceed\n"
            "- escalate: must hand off to a human agent\n"
            "Return only the choice label."
        ),
        choices=["auto_resolve", "clarify", "escalate"],
    )

    # --- 3) Auto-resolve path ---
    auto_resolve = AgentNode(
        name="auto_resolve",
        llm=llm,
        reads=["messages"],
        node_prompt=(
            "You are an expert customer support agent.\n"
            "Write the final resolution to the customer.\n\n"
            "Rules:\n"
            "- Be concise but complete.\n"
            "- If there are steps, number them.\n"
            "- Do not request additional info.\n"
            "- If the thread is ambiguous but severity isn't high, make reasonable assumptions and confirm the key outcome.\n\n"
            "Use the conversation history in {messages} and respond as the assistant to the customer."
        ),
    )

    # --- 4) Clarify path (ask exactly one question) ---
    clarify = AgentNode(
        name="clarify",
        llm=llm,
        node_prompt=(
            "You are a support agent. Ask exactly one clarifying question that will unblock resolution.\n"
            "Be polite and specific. Do not provide a full resolution yet.\n"
            "Consider what the triage decided.\n"
            "Output should be the customer-facing message only."
        ),
    )

    # --- 5) Escalate path ---
    escalate = HumanNode(
        name="escalate",
        node_prompt=(
            "A support thread was escalated.\n"
            "Review the conversation and triage rationale and take ownership of the customer reply.\n"
            "When you respond to the customer, ensure it addresses the issue and includes any needed next steps."
        ),
        n=3,
        show=None,
        into="messages",
    )

    # Optional guard: if severity is low/medium, avoid escalation.
    # This condition reads the framed triage text to make a deterministic gating decision.
    def severity_allows_auto(state):
        text = ""
        try:
            text = state.get("messages", [])[-1].content
        except Exception:
            text = ""
        t = text.lower()
        # We allow auto_resolve/clarify; disallow escalation only if explicitly low.
        return "allow" if "severity: low" not in t else "allow"  # keep simple; no extra gating

    _severity_gate = ConditionNode(
        name="severity_gate",
        condition=severity_allows_auto,
        choices=["allow"],
    )

    # Routing wiring:
    # route["auto_resolve"] and route["clarify"] and route["escalate"] must each lead to exactly one handler.
    frame > route
    route["auto_resolve"] > auto_resolve
    route["clarify"] > clarify
    route["escalate"] > escalate

    return AgenticGraph(start_node=frame, end_nodes={auto_resolve, clarify, escalate})
