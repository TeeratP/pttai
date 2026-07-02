"""Generated pipeline for task 'fact_check' (sample 4).

TASK: Given a factual claim, gather supporting and contradicting evidence, then issue a verdict with a short rationale.
"""

from pttai import AgentNode, AgenticGraph, fanout, DecisionNode


def build_graph(llm):
    # Frame the task into one concrete decision target (for consistent evidence retrieval/analysis)
    frame = AgentNode(
        name="frame",
        llm=llm,
        node_prompt=(
            "You are preparing a fact-check. "
            "Given the user's factual claim, restate it as ONE clear verification target "
            "(a single sentence) that can be supported or contradicted by evidence. "
            "Return only the verification target."
        ),
    )

    # Optional triage to decide whether evidence is likely available or if we must fallback to reasoning.
    # Choice is still guided by the model, but keeps evidence gathering crisp.
    triage = DecisionNode(
        name="triage",
        llm=llm,
        node_prompt=(
            "Decide how to proceed for fact-checking.\n"
            "Choose 'needs_research' if specific external evidence would likely be required, "
            "or if web-style sources would help. Choose 'reason_only' if general reasoning is sufficient "
            "or the claim is too vague for meaningful evidence gathering.\n\n"
            "The output must be exactly one of the provided choices."
        ),
        choices=["needs_research", "reason_only"],
    )

    # Evidence gatherers: supporting vs contradicting.
    # They only use the conversation messages as context; no external tools are wired here.
    # (They can be swapped later with retriever tools if desired.)
    support_evidence = AgentNode(
        name="support_evidence",
        llm=llm,
        node_prompt=(
            "You are the Support Evidence Collector for a fact-check.\n"
            "Task: Collect the strongest supporting evidence for the verification target.\n"
            "- List 2-5 distinct evidence points.\n"
            "- For each point: state what would count as evidence, and what kind of source/type "
            "(e.g., peer-reviewed study, official statistics, documented case, primary document) would support it.\n"
            "- If you cannot cite real sources, propose concrete evidence to look for.\n\n"
            "Verification target:\n{messages}"
        ),
    )

    contradict_evidence = AgentNode(
        name="contradict_evidence",
        llm=llm,
        node_prompt=(
            "You are the Contradiction Evidence Collector for a fact-check.\n"
            "Task: Collect the strongest contradicting evidence for the verification target.\n"
            "- List 2-5 distinct evidence points.\n"
            "- For each point: state what would count as evidence, and what kind of source/type "
            "(e.g., peer-reviewed study, official statistics, documented case, primary document) would dispute it.\n"
            "- If you cannot cite real sources, propose concrete evidence to look for.\n\n"
            "Verification target:\n{messages}"
        ),
    )

    # Weigh arguments into a verdict.
    verdict = AgentNode(
        name="verdict",
        llm=llm,
        node_prompt=(
            "You are the fact-check chair.\n"
            "Using the collected supporting and contradicting evidence, "
            "issue a verdict about the user's original factual claim.\n\n"
            "Output format (exactly 3 lines):\n"
            "Verdict: Supported / Contradicted / Inconclusive\n"
            "Rationale: 2-4 sentences, explicitly referencing the strongest evidence points.\n"
            "Next: If inconclusive, specify what evidence would most resolve the uncertainty.\n\n"
            "Include the supporting evidence and contradicting evidence you were given."
        ),
    )

    # Minimal branch-specific prompts: if reason_only, still produce evidence-like reasoning points.
    reason_support = AgentNode(
        name="reason_support",
        llm=llm,
        node_prompt=(
            "Reasoning-only mode (no external research). "
            "Create 2-4 evidence-like reasoning points that would support the verification target. "
            "For each, briefly explain the logic/assumptions and what would falsify it."
        ),
    )

    reason_contradict = AgentNode(
        name="reason_contradict",
        llm=llm,
        node_prompt=(
            "Reasoning-only mode (no external research). "
            "Create 2-4 evidence-like reasoning points that would contradict the verification target. "
            "For each, briefly explain the logic/assumptions and what would falsify it."
        ),
    )

    # Wire: frame -> triage -> (parallel evidence) -> verdict
    #
    # Note: DecisionNode routes by choice; we fan out from both routed branches into parallel collectors.
    # To keep the graph single and clean, we route into two different parallel fanouts and both join at verdict.
    triage["needs_research"] > fanout(support_evidence, contradict_evidence) > verdict
    triage["reason_only"] > fanout(reason_support, reason_contradict) > verdict

    return AgenticGraph(start_node=frame, end_nodes={verdict})
