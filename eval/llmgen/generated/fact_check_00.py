"""Generated pipeline for task 'fact_check' (sample 0).

TASK: Given a factual claim, gather supporting and contradicting evidence, then issue a verdict with a short rationale.
"""

from pttai import AgentNode, AgenticGraph, fanout, DecisionNode


def build_graph(llm):
    frame = AgentNode(
        name="frame",
        llm=llm,
        node_prompt=(
            "You are a research triage assistant.\n\n"
            "Given a factual claim, extract:\n"
            "1) The claim rewritten as a single, testable factual statement.\n"
            "2) The minimal set of key sub-claims / facets that evidence should cover.\n"
            "Output as concise bullet points.\n"
        ),
    )

    gather_support = AgentNode(
        name="gather_support",
        llm=llm,
        node_prompt=(
            "You are the SUPPORT evidence researcher.\n\n"
            "Using ONLY the provided framed claim and facets, gather strong supporting evidence.\n"
            "Requirements:\n"
            "- Provide 3-6 evidence items.\n"
            "- For each item: what it says (1-2 sentences), why it's relevant, and what to check in it.\n"
            "- Prefer verifiable sources (e.g., studies, official reports, reputable data) and be explicit about "
            "any uncertainty.\n"
            "- Avoid speculation; if you cannot cite strong evidence, say so.\n"
        ),
    )

    gather_contra = AgentNode(
        name="gather_contra",
        llm=llm,
        node_prompt=(
            "You are the CONTRADICT evidence researcher.\n\n"
            "Using ONLY the provided framed claim and facets, gather strong contradicting evidence.\n"
            "Requirements:\n"
            "- Provide 3-6 evidence items.\n"
            "- For each item: what it says (1-2 sentences), why it's relevant, and what to check in it.\n"
            "- Prefer verifiable sources (e.g., studies, official reports, reputable data) and be explicit about "
            "any uncertainty.\n"
            "- Avoid speculation; if you cannot cite strong evidence, say so.\n"
        ),
    )

    evidence_decision = DecisionNode(
        name="evidence_decision",
        llm=llm,
        node_prompt=(
            "Decide the verdict direction based on the gathered evidence.\n"
            "You will be given:\n"
            "- framed claim\n"
            "- support evidence notes\n"
            "- contradict evidence notes\n\n"
            "Choose exactly one option:\n"
            "positive: evidence supports the claim as stated\n"
            "negative: evidence contradicts the claim as stated\n"
            "mixed: evidence is conflicting / incomplete; cannot clearly confirm or refute\n"
            "insufficient: little or no reliable evidence is available\n"
        ),
        choices=["positive", "negative", "mixed", "insufficient"],
    )

    verdict = AgentNode(
        name="verdict",
        llm=llm,
        node_prompt=(
            "You are a careful fact-checking judge.\n\n"
            "Write a short verdict with a short rationale (2-4 sentences).\n"
            "Use this format:\n"
            "Verdict: <positive|negative|mixed|insufficient>\n"
            "Rationale: <short rationale grounded in the evidence>\n\n"
            "Constraints:\n"
            "- Be specific about what evidence tipped the decision.\n"
            "- If mixed/insufficient, say exactly what additional evidence would resolve it.\n"
        ),
    )

    # One-liner wiring: parallel evidence gathering, then decide, then issue verdict
    frame > fanout(gather_support, gather_contra) > evidence_decision > verdict

    return AgenticGraph(start_node=frame, end_nodes={verdict})
