"""Generated pipeline for task 'escalation' (sample 4).

TASK: Read a support thread, judge how severe it is, then either auto-resolve it, ask the customer a clarifying question, or escalate it to a human agent.
"""

from pttai import AgentNode, DecisionNode, ConditionNode, HumanNode, AgenticGraph


def build_graph(llm):
    # 1) Frame: extract the ticket + determine what "resolution intent" should be
    frame = AgentNode(
        name="frame",
        llm=llm,
        node_prompt=(
            "You are a support triage analyst.\n"
            "Given the support thread in the conversation, produce a clean, actionable brief.\n\n"
            "Output ONLY by appending to the messages history (pttai messages).\n"
            "Your brief must include:\n"
            "1) One-sentence issue summary.\n"
            "2) Customer requested outcome (what they want).\n"
            "3) Product/feature involved (if any).\n"
            "4) Any explicit error messages / symptoms.\n"
            "5) What info is missing or ambiguous (if any).\n"
            "6) If you can infer severity, suggest a severity level: low/medium/high/urgent."
        ),
        input_field="messages",
        writes=["messages"],
        retry=False,
    )

    # 2) Decide severity with constrained routing
    severity_router = DecisionNode(
        name="severity_router",
        llm=llm,
        node_prompt=(
            "Classify severity for support handling.\n"
            "Choose EXACTLY one option based on the thread.\n"
            "Guidance:\n"
            "- low: minor inconvenience, non-blocking, no security/financial impact\n"
            "- medium: impacting a feature but workaround exists, or moderate frustration\n"
            "- high: major functionality broken for the user, no immediate workaround, or repeated incidents\n"
            "- urgent: security/privacy incident, data loss, legal/compliance risk, widespread outage impact, or payment fraud\n\n"
            "Base your decision solely on the provided thread. If uncertain, prefer the safer higher severity."
        ),
        choices=["low", "medium", "high", "urgent"],
        input_field="messages",
    )

    # 3) Decide next action (auto-resolve vs clarifying question vs escalate)
    #    We use a second decision node so we can separately reason about severity and action.
    action_router = DecisionNode(
        name="action_router",
        llm=llm,
        node_prompt=(
            "Decide the best next action for the support team.\n"
            "Choose EXACTLY one option.\n\n"
            "Options:\n"
            "A) auto_resolve: You have enough information to provide a complete fix/workaround and next steps.\n"
            "B) ask_customer: You must ask a clarifying question to proceed (missing details needed to resolve or verify).\n"
            "C) escalate_human: The issue likely requires human judgment/escalation (high-risk, billing/security, complex edge cases, or user indicates urgent harm).\n\n"
            "Base your decision on the provided thread and inferred severity."
        ),
        choices=["auto_resolve", "ask_customer", "escalate_human"],
        input_field="messages",
    )

    # 4) Clarifying question path
    clarifying_question = HumanNode(
        name="clarifying_question",
        node_prompt=(
            "You are the human support agent.\n"
            "Formulate ONE concise clarifying question to the customer that enables resolution.\n"
            "If multiple questions are required, pick the single most critical question.\n\n"
            "Return just the question text."
        ),
        n=1,
        into="messages",
        retry=False,
    )

    # 5) Auto-resolve path
    auto_resolve = AgentNode(
        name="auto_resolve",
        llm=llm,
        node_prompt=(
            "You are an expert customer support agent.\n"
            "Auto-resolve the customer's thread.\n\n"
            "Requirements:\n"
            "- Provide a clear, friendly response.\n"
            "- Include a step-by-step fix or workaround when possible.\n"
            "- If relevant, include how to avoid recurrence.\n"
            "- If you had to assume anything, state it briefly and offer the next step to confirm.\n"
            "- Do NOT ask more than one question; prefer completing resolution.\n\n"
            "Thread context is the conversation so far."
        ),
        input_field="messages",
        writes=["messages"],
        retry=False,
    )

    # 6) Escalate path (human-in-the-loop handoff)
    escalate_to_human = HumanNode(
        name="escalate_to_human",
        node_prompt=(
            "You are the human support agent.\n"
            "Create an internal handoff note for escalation.\n"
            "Include:\n"
            "1) Severity and rationale\n"
            "2) What is known from the thread\n"
            "3) What needs a human to decide/do\n"
            "4) Any suggested next actions\n\n"
            "Return ONLY the handoff note text."
        ),
        n=1,
        into="messages",
        retry=False,
    )

    # 7) Secondary guard: if severity is urgent/high, prefer escalation or ask.
    #    We'll gate the action_router decision using a deterministic condition.
    def gate_by_severity(state):
        # decision and severity are set by prior nodes. `severity_router` writes into `decision`.
        # The most recent DecisionNode overwrites `state["decision"]`, so we must preserve severity.
        # We'll instead interpret that `severity_router` decision is already in the conversation history
        # via log; but for correctness, we will re-decision by asking the model to embed severity
        # in the brief already. To keep the pipeline deterministic, we use the message content.
        #
        # Since we cannot read structured fields beyond declared state keys, do a robust heuristic:
        # If the conversation contains 'urgent' or 'high' in the latest assistant brief, escalate.
        txt = ""
        if isinstance(state.get("messages"), list) and state["messages"]:
            last = state["messages"][-1]
            # last can be a Message-like object with .content
            txt = getattr(last, "content", "") or str(last)
        txt_l = txt.lower()
        if "urgent" in txt_l or "high" in txt_l:
            return "escalate_if_needed"
        return "normal"

    severity_gate = ConditionNode(
        name="severity_gate",
        condition=gate_by_severity,
        choices=["escalate_if_needed", "normal"],
        reads=["messages"],
        retry=False,
    )

    # 8) Final action combiner:
    #    If we are in high/urgent territory, we override:
    #    - auto_resolve -> ask_customer (if uncertain) OR escalate_human (if risky). We'll route to escalate_human
    #      for safety; otherwise ask_customer via a model decision.
    #    - ask_customer stays ask_customer
    #    - escalate_human stays escalate_human
    #
    # Since ConditionNode cannot inspect prior action_router choice reliably via state["decision"] (it gets overwritten),
    # we will route using the latest brief and let the human-decision nodes decide in the right direction.
    #
    # We'll route deterministically:
    # - in escalate_if_needed: go to escalate_to_human directly
    # - in normal: use action_router output to choose.
    #
    # To avoid dangling routes, we create a wiring where:
    # frame -> severity_router -> action_router -> severity_gate -> (either escalate_to_human or use action_router handlers)
    # But we can't branch on action_router after severity_gate without reliable state. Instead, we wire:
    # frame -> action_router, and independently wire severity_gate to escalation path before action routing.
    #
    # We'll implement as:
    # frame -> severity_gate -> if urgent/high => escalate_to_human
    # else => action_router => {auto_resolve, ask_customer, escalate_human}
    #
    # Note: severity_gate reads messages and uses heuristic; action_router still provides best choice for normal cases.

    # Rewire: gate first, then route normal to action_router
    # (We still keep severity_router in the graph for severity classification logging/analysis.)
    # For wiring, create "normal_to_router" by placing action_router after gate in the normal path.

    # The only compile-time requirement: all routes must be wired.
    # We'll wire as described below.

    # Normal path starts from action_router (auto_resolve/ask_customer/escalate_human).
    action_router["auto_resolve"] > auto_resolve
    action_router["ask_customer"] > clarifying_question
    action_router["escalate_human"] > escalate_to_human

    # After severity_gate:
    severity_gate["escalate_if_needed"] > escalate_to_human
    severity_gate["normal"] > action_router

    # Tie everything together:
    # Start with frame, then a severity_router (for richer context/log), then severity_gate (heuristic),
    # then proceed as above. severity_router choice is not used structurally, but it does refine the thread.
    frame > severity_router > severity_gate

    # Terminal nodes:
    # - auto_resolve, clarifying_question, escalate_to_human are end nodes.
    return AgenticGraph(start_node=frame, end_nodes={auto_resolve, clarifying_question, escalate_to_human})
