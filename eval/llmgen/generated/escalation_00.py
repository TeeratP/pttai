"""Generated pipeline for task 'escalation' (sample 0).

TASK: Read a support thread, judge how severe it is, then either auto-resolve it, ask the customer a clarifying question, or escalate it to a human agent.
"""

from pttai import AgentNode, DecisionNode, ConditionNode, AgenticGraph


def build_graph(llm):
    # 1) Frame: convert the raw support thread into a single decision
    frame = AgentNode(
        name="frame",
        llm=llm,
        node_prompt=(
            "You are a support triage assistant.\n\n"
            "Given the SUPPORT_THREAD below, decide the correct outcome.\n"
            "Your output must be concise and decision-like.\n\n"
            "SUPPORT_THREAD:\n{messages}\n\n"
            "Return a single sharp decision description that includes:\n"
            "- severity (low/medium/high)\n"
            "- the category (billing/bug/account/setup/other)\n"
            "- whether it is safe to auto-resolve\n"
            "- what additional info is needed if not safe\n"
            "One sentence only."
        ),
        reads=["messages"],
        writes=["messages"],  # replace conversation with the framed decision
    )

    # 2) Route (LLM): choose one of the three actions
    route = DecisionNode(
        name="route",
        llm=llm,
        node_prompt=(
            "Route the support thread to exactly ONE action.\n\n"
            "The input 'messages' is a single decision description produced earlier.\n"
            "Choose the best action:\n"
            "- auto_resolve: we can fix it safely without customer clarification\n"
            "- ask_clarifying_question: we need one specific question to proceed\n"
            "- escalate_to_human: needs human review due to high severity, risk, policy, or complex debugging\n"
            "Follow severity guidance in the decision description."
        ),
        choices=["auto_resolve", "ask_clarifying_question", "escalate_to_human"],
        input_field="messages",
        reads=["messages"],
        # DecisionNode writes into the built-in `decision` field.
    )

    # 3) Auto-resolve branch: generate resolution steps
    auto_resolve = AgentNode(
        name="auto_resolve",
        llm=llm,
        node_prompt=(
            "You are an automated support agent.\n\n"
            "Write a complete customer-facing resolution.\n"
            "Use this decision context:\n{messages}\n\n"
            "Requirements:\n"
            "- Be clear and actionable\n"
            "- Include any specific steps, settings, or commands\n"
            "- If relevant, include a short 'If this doesn't work' fallback\n"
            "- Do not ask more than 0 questions (auto-resolve only)\n"
            "- Keep it professional and concise"
        ),
        reads=["messages"],
        writes={"answer": str},
    )

    # 4) Clarify branch: ask exactly one question
    ask_clarifying_question = AgentNode(
        name="ask_clarifying_question",
        llm=llm,
        node_prompt=(
            "You are a support agent that must ask exactly ONE clarifying question.\n\n"
            "Decision context:\n{messages}\n\n"
            "Requirements:\n"
            "- Ask a single question that unblocks resolution\n"
            "- Keep it short (one sentence if possible)\n"
            "- Avoid multiple-part questions\n"
            "- State why you need it in a brief clause"
        ),
        reads=["messages"],
        writes={"answer": str},
    )

    # 5) Escalate branch: produce an internal summary for a human agent
    escalate_to_human = AgentNode(
        name="escalate_to_human",
        llm=llm,
        node_prompt=(
            "You are preparing an escalation packet for a human support agent.\n\n"
            "Decision context:\n{messages}\n\n"
            "Requirements: produce a structured packet with:\n"
            "1) Summary of issue\n"
            "2) Severity and category\n"
            "3) What we tried / what is known\n"
            "4) Key missing info (if any)\n"
            "5) Recommended next action\n\n"
            "Do NOT include customer-facing language; this is for internal use."
        ),
        reads=["messages"],
        writes={"answer": str},
    )

    # 6) Optional safety gate (deterministic): ensure routing is consistent with severity word
    #    This catches obvious mismatch; if it mismatches, we force escalation.
    #    It reads from the framed decision in `messages` and the LLM decision in `decision`.
    #    Choices are terminal tags that determine the final routing.
    def severity_mismatch(state):
        decision_choice = state.get("decision")
        decision_text = state.get("messages", "")
        sev = "high" if "high" in str(decision_text).lower() else ("medium" if "medium" in str(decision_text).lower() else "low")
        if sev == "high" and decision_choice != "escalate_to_human":
            return "force_escalate"
        return "ok"

    gate = ConditionNode(
        name="gate",
        condition=severity_mismatch,
        choices=["ok", "force_escalate"],
        reads=["messages", "decision"],
    )

    # When forced, we ignore the original branch and go to human.
    # We need a small router node, but we can just route the gate choices:
    # ok -> original route children; force_escalate -> escalate_to_human
    # However, DecisionNode routing is already wired; to keep it single pipeline,
    # we add the gate after route and wire its outcomes.
    # We'll wire route -> gate, and then gate -> terminal branches.
    route_choice_to_gate = route  # readability alias

    # Wiring:
    # start (frame) -> route (DecisionNode) -> gate (ConditionNode) -> terminals
    frame > route_choice_to_gate > gate
    gate["force_escalate"] > escalate_to_human
    gate["ok"] > auto_resolve  # default for ok is overwritten by decision routing below? can't.
    # To correctly honor original LLM choice on "ok", we additionally branch from `route`
    # into the terminals, but we can't "route from gate to multiple based on route choice".
    # Instead, we wire gate["ok"] to a second decision that uses the existing `decision` field.
    #
    # Implement second DecisionNode to select among the three using structured output constrained
    # (it will simply echo based on the existing `decision` context).
    select_from_decision = DecisionNode(
        name="select_from_decision",
        llm=llm,
        node_prompt=(
            "You will receive:\n"
            "- state.decision: an action from earlier routing\n"
            "- state.messages: the decision context\n\n"
            "Return exactly ONE choice among:\n"
            "auto_resolve, ask_clarifying_question, escalate_to_human\n"
            "Use the state.decision as the source of truth; if it disagrees with severity implied by messages,\n"
            "prefer severity implied by messages only when it is high-risk (otherwise follow state.decision)."
        ),
        choices=["auto_resolve", "ask_clarifying_question", "escalate_to_human"],
        input_field="messages",
        reads=["messages", "decision"],
    )

    gate["ok"] > select_from_decision
    select_from_decision["auto_resolve"] > auto_resolve
    select_from_decision["ask_clarifying_question"] > ask_clarifying_question
    select_from_decision["escalate_to_human"] > escalate_to_human

    # Return schema-free graph; upstream caller seeds `messages` with the support thread.
    return AgenticGraph(start_node=frame, end_nodes={auto_resolve, ask_clarifying_question, escalate_to_human})
