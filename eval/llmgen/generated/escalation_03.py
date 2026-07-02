"""Generated pipeline for task 'escalation' (sample 3).

TASK: Read a support thread, judge how severe it is, then either auto-resolve it, ask the customer a clarifying question, or escalate it to a human agent.
"""

from pttai import AgentNode, DecisionNode, AgenticGraph, HumanNode, ConditionNode


def build_graph(llm):
    # --- Node 1: classify severity + intent (LLM) ---
    route = DecisionNode(
        name="route",
        llm=llm,
        node_prompt=(
            "You are a support triage agent.\n"
            "Given the support thread, decide what should happen next.\n\n"
            "Decisions:\n"
            "- auto_resolve: The issue is straightforward; you can provide the resolution immediately.\n"
            "- clarify: Missing critical details; ask exactly ONE concise clarifying question.\n"
            "- escalate: High severity, safety, legal/compliance risk, account risk, billing disputes needing review, "
            "or any situation requiring a human.\n\n"
            "Base your choice on severity and whether enough information exists to safely resolve."
        ),
        choices=["auto_resolve", "clarify", "escalate"],
        input_field="messages",
    )

    # --- Node 2: auto-resolve (LLM) ---
    auto_resolve = AgentNode(
        name="auto_resolve",
        llm=llm,
        node_prompt=(
            "You are a customer support assistant.\n"
            "Use the support thread to draft a complete, helpful, and policy-safe resolution.\n\n"
            "Requirements:\n"
            "- Be specific and reference what the customer said.\n"
            "- Provide actionable steps.\n"
            "- If any details are uncertain but not critical, state reasonable assumptions.\n"
            "- Do NOT ask more than necessary; prefer completing the resolution.\n"
        ),
        writes={"answer": str},
    )

    # --- Node 3: clarifying question (LLM) ---
    clarify = AgentNode(
        name="clarify",
        llm=llm,
        node_prompt=(
            "You are a customer support assistant.\n"
            "Ask exactly ONE clarifying question needed to resolve the customer's issue.\n"
            "Requirements:\n"
            "- The question must be concise.\n"
            "- It must be the minimum detail required.\n"
            "- Do not provide additional commentary beyond the question.\n"
        ),
        writes={"answer": str},
    )

    # --- Node 4: escalate to human (no LLM in the human review node) ---
    human_escalation = HumanNode(
        name="human_escalation",
        node_prompt=(
            "Please review this support thread and the drafted context.\n"
            "Decide next steps and craft the final response to the customer."
        ),
        n=0,  # show nothing automatically; the thread is already in messages
        into="messages",
    )

    # --- Node 5: create escalation summary for human (LLM) ---
    escalate_brief = AgentNode(
        name="escalate_brief",
        llm=llm,
        node_prompt=(
            "Create a brief escalation packet for a human agent.\n"
            "Include:\n"
            "- Severity level (low/med/high)\n"
            "- The core problem in one sentence\n"
            "- Evidence from the thread (key quotes)\n"
            "- What is missing / what should be checked\n"
            "- Any safety/compliance notes if relevant\n"
            "- Suggested next steps\n\n"
            "Output ONLY the escalation packet text."
        ),
        writes={"answer": str},
    )

    # --- Node 6: safety gate for clarification (deterministic) ---
    # If the clarifying question is somehow empty/too vague, reroute to human.
    # (This is a small safeguard; routing already uses an LLM decision.)
    def clarification_too_vague(state) -> str:
        q = (state.get("answer") or "").strip()
        if not q:
            return "to_human"
        # very rough heuristic: question should end with a question mark
        if "?" not in q[-3:]:
            return "to_human"
        # keep as clarify path
        return "keep"

    clarify_gate = ConditionNode(
        name="clarify_gate",
        condition=clarification_too_vague,
        choices=["keep", "to_human"],
        reads=["answer"],
    )

    # Wire: route -> branch
    route["auto_resolve"] > auto_resolve
    route["clarify"] > clarify > clarify_gate
    clarify_gate["keep"] > clarify
    clarify_gate["to_human"] > escalate_brief > human_escalation

    route["escalate"] > escalate_brief > human_escalation

    return AgenticGraph(start_node=route, end_nodes={auto_resolve, clarify, human_escalation})
