"""Generated pipeline for task 'escalation' (sample 1).

TASK: Read a support thread, judge how severe it is, then either auto-resolve it, ask the customer a clarifying question, or escalate it to a human agent.
"""

from pttai import AgentNode, DecisionNode, AgenticGraph


def build_graph(llm):
    # --- Node prompts ---
    read_thread = AgentNode(
        name="read_thread",
        llm=llm,
        node_prompt=(
            "You will be given a customer support thread in `messages`.\n"
            "Extract and restate:\n"
            "1) The customer's issue/request (1-3 sentences)\n"
            "2) Any explicit facts (plans, versions, errors, dates)\n"
            "3) The customer's current severity/urgency signals (if any)\n"
            "Return a concise support dossier for downstream steps.\n"
            "If the thread is unclear or missing key details, state what is missing."
        ),
    )

    # Decide severity + path. DecisionNode writes its chosen label to state["decision"].
    severity_route = DecisionNode(
        name="severity_route",
        llm=llm,
        node_prompt=(
            "Classify this support thread into exactly ONE bucket:\n"
            "A) auto_resolve: Straightforward issue with enough info to provide a correct fix/workaround.\n"
            "B) ask_clarifying: Needs one or more missing details; customer should be asked 1-3 focused questions.\n"
            "C) escalate_human: Potentially high risk, requires policy/account action, security/privacy concern,\n"
            "billing/legal dispute, repeated failures, or insufficient safety/eligibility to auto-handle.\n\n"
            "Choose the best fit. Be conservative: if unsure between B and C, prefer C."
        ),
        choices=["auto_resolve", "ask_clarifying", "escalate_human"],
    )

    auto_resolve = AgentNode(
        name="auto_resolve",
        llm=llm,
        node_prompt=(
            "You are an expert support agent. Using the prior extracted dossier in the conversation,\n"
            "write the final customer-facing response that resolves the issue.\n\n"
            "Requirements:\n"
            "- Be concise but complete.\n"
            "- Provide step-by-step instructions if needed.\n"
            "- Include a brief 'What to do next' checklist.\n"
            "- If relevant, include troubleshooting steps and how to verify success.\n"
            "- Do not ask clarifying questions; assume what you need is already present.\n"
        ),
    )

    ask_clarifying = AgentNode(
        name="ask_clarifying",
        llm=llm,
        node_prompt=(
            "You are an expert support agent. The thread lacks enough details for safe resolution.\n"
            "Write a customer-facing reply that asks 1-3 targeted clarifying questions.\n\n"
            "Requirements:\n"
            "- Start with a short acknowledgement of the issue.\n"
            "- Ask only the minimum questions needed to proceed.\n"
            "- Phrase questions to be easy to answer (include examples or where to find info).\n"
            "- Do not escalate to a human unless truly necessary.\n"
        ),
    )

    escalate_human = AgentNode(
        name="escalate_human",
        llm=llm,
        node_prompt=(
            "You are a support triage agent preparing an escalation to a human agent.\n"
            "Write TWO parts:\n"
            "1) Customer-facing message: polite, brief, and transparent that a human agent will assist.\n"
            "2) Internal escalation notes for the human: summarize issue, evidence/facts from the thread,\n"
            "   why it must be escalated, and any recommended next steps or information needed.\n\n"
            "Be careful about sensitive data; avoid requesting secrets.\n"
        ),
    )

    # --- Wiring (schema-free, message-driven) ---
    # Input is expected in `messages` (pttai supports invoke(message=...) as a shorthand).
    read_thread > severity_route
    severity_route["auto_resolve"] > auto_resolve
    severity_route["ask_clarifying"] > ask_clarifying
    severity_route["escalate_human"] > escalate_human

    return AgenticGraph(
        start_node=read_thread,
        end_nodes={auto_resolve, ask_clarifying, escalate_human},
    )
