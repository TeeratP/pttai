"""Generated pipeline for task 'fact_check' (sample 3).

TASK: Given a factual claim, gather supporting and contradicting evidence, then issue a verdict with a short rationale.
"""

from pttai import AgentNode, ConditionNode, DecisionNode, AgenticGraph, fanout


def build_graph(llm):
    # 1) Frame the claim into a sharp, checkable decision
    frame = AgentNode(
        name="frame",
        llm=llm,
        node_prompt=(
            "Given the user's claim, restate it as a checkable decision statement.\n"
            "Output MUST be a single concise sentence suitable for fact-checking.\n"
            "User claim will be provided as: {messages}\n"
            "Decision statement:"
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # 2) Determine what evidence we should look for / how to weigh it
    #    (LLM decides; choices constrained)
    triage = DecisionNode(
        name="triage",
        llm=llm,
        node_prompt=(
            "You are helping fact-check. Choose the best evidence mode for the claim.\n"
            "If the claim is about events, pick EVENTS; if it is about a quantitative or comparative statement, pick QUANT;\n"
            "if it is about definitions/claims of fact from sources, pick SOURCES.\n"
            "Decision statement is in messages."
        ),
        choices=["EVENTS", "QUANT", "SOURCES"],
        input_field="messages",
        reads=["messages"],
    )

    # 3) Evidence gathering sub-agents (supporting and contradicting)
    #    Each agent produces an evidence package in messages for its role.
    support_search = AgentNode(
        name="support_search",
        llm=llm,
        node_prompt=(
            "You are a diligent evidence gatherer for SUPPORTING evidence.\n"
            "Task: Gather the strongest supporting evidence for the claim.\n"
            "You must output:\n"
            "- 2-5 specific factual points (with dates, entities, and quantities when applicable)\n"
            "- any uncertainty/limitations you see\n"
            "- brief citations-style references (just descriptive, e.g. 'According to X, ...')\n\n"
            "Use the decision statement and triage mode.\n"
            "Format strictly:\n"
            "SUPPORT_EVIDENCE:\n"
            "- ...\n"
            "LIMITATIONS:\n"
            "- ...\n"
        ),
        reads=["messages"],
        writes=["messages"],
    )

    contradict_search = AgentNode(
        name="contradict_search",
        llm=llm,
        node_prompt=(
            "You are a rigorous evidence gatherer for CONTRADICTING evidence.\n"
            "Task: Gather the strongest contradicting evidence for the claim.\n"
            "You must output:\n"
            "- 2-5 specific factual points that challenge the claim\n"
            "- any uncertainty/limitations you see\n"
            "- brief citations-style references (just descriptive, e.g. 'According to Y, ...')\n\n"
            "Use the decision statement and triage mode.\n"
            "Format strictly:\n"
            "CONTRADICT_EVIDENCE:\n"
            "- ...\n"
            "LIMITATIONS:\n"
            "- ...\n"
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # 4) Optional quality gate (deterministic) to ensure we have enough evidence.
    #    The predicate checks for presence of markers.
    def has_enough_evidence(state):
        text = state["messages"][-1].content if state.get("messages") else ""
        # Minimal heuristic: require either SUPPORT_EVIDENCE or CONTRADICT_EVIDENCE markers.
        support_ok = "SUPPORT_EVIDENCE" in text
        contradict_ok = "CONTRADICT_EVIDENCE" in text
        # If we have neither marker, consider it insufficient and ask to re-run both searches.
        if not (support_ok and contradict_ok):
            return "redo"
        return "ok"

    evidence_gate = ConditionNode(
        name="evidence_gate",
        condition=has_enough_evidence,
        choices=["redo", "ok"],
        reads=["messages"],
    )

    # 5) Verdict: weigh support vs contradiction into a short rationale.
    verdict = AgentNode(
        name="verdict",
        llm=llm,
        node_prompt=(
            "You are the fact-checking chair.\n"
            "Create a verdict on whether the original claim is True, False, or Unverifiable\n"
            "based on the gathered supporting and contradicting evidence.\n\n"
            "Requirements:\n"
            "- Verdict must be one of: TRUE / FALSE / UNVERIFIABLE\n"
            "- Provide a short rationale (2-5 sentences) that explicitly references which evidence points\n"
            "support/contradict the claim and why.\n"
            "- If evidence quality is insufficient, choose UNVERIFIABLE.\n\n"
            "Output format strictly:\n"
            "VERDICT: <TRUE|FALSE|UNVERIFIABLE>\n"
            "RATIONALE: <short rationale>\n"
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # Wiring:
    # frame -> triage
    # triage chosen evidence mode is included in messages context through message history.
    # Then gather support and contradict concurrently, join into gate, then verdict.
    # If gate says redo, we re-run both evidence agents.
    support_and_contra_join = AgentNode(
        name="merge_evidence",
        llm=llm,
        node_prompt=(
            "Merge the gathered supporting and contradicting evidence into one coherent evidence dossier.\n"
            "Keep both sections, and remove duplicates.\n\n"
            "Output strictly:\n"
            "SUPPORT_EVIDENCE:\n"
            "- ...\n"
            "CONTRADICT_EVIDENCE:\n"
            "- ...\n"
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # Parallel fan-out then join: support & contradict gather concurrently, then merge.
    gather = fanout(support_search, contradict_search) > support_and_contra_join

    # Decision routing: all modes still go to the same gather+verdict pipeline,
    # but we keep the triage node as a structured decomposition step.
    triage["EVENTS"] > gather
    triage["QUANT"] > gather
    triage["SOURCES"] > gather

    # Gate then verdict; optionally loop evidence gathering if not enough evidence.
    support_and_contra_join > evidence_gate
    evidence_gate["ok"] > verdict
    evidence_gate["redo"] > gather > support_and_contra_join  # loop back to merge

    return AgenticGraph(start_node=frame, end_nodes={verdict})
