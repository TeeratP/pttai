"""Generated pipeline for task 'fact_check' (sample 2).

TASK: Given a factual claim, gather supporting and contradicting evidence, then issue a verdict with a short rationale.
"""

from pttai import AgentNode, DecisionNode, ConditionNode, HumanNode, AgenticGraph, fanout
from pttai.tools import make_retriever_tool


def build_graph(llm):
    # 1) Frame the claim into a decision-ready artifact
    frame = AgentNode(
        name="frame",
        llm=llm,
        node_prompt=(
            "You are a careful evidence analyst.\n"
            "Turn the user's factual claim into a decision-ready form.\n"
            "Output format (one line each):\n"
            "1) claim: <restated factual claim>\n"
            "2) entities: <key entities/people/places/organizations>\n"
            "3) time_scope: <explicit date range if implied, else 'unspecified'>\n"
            "4) key_terms: <5-10 search terms/phrases>\n"
            "5) verification_questions: <2-4 questions that would confirm or refute the claim>\n"
        ),
        # Use the existing user content
        reads=["messages"],
        writes=["messages"],
    )

    # 2) Decide what kind of evidence gathering is needed (broad vs. targeted)
    decide_depth = DecisionNode(
        name="decide_depth",
        llm=llm,
        node_prompt=(
            "Decide the evidence-gathering depth for the claim.\n"
            "Choose 'light' when the claim is narrow and likely verifiable with few searches.\n"
            "Choose 'deep' when the claim is broad, complex, or time-dependent and needs more sourcing.\n"
            "Return only the chosen label."
        ),
        choices=["light", "deep"],
        input_field="messages",
    )

    # 3) Supporting evidence persona (optimist / pro)
    support = AgentNode(
        name="support",
        llm=llm,
        node_prompt=(
            "You are the SUPPORTING EVIDENCE analyst.\n"
            "Goal: gather strong evidence that the claim is TRUE.\n"
            "Use the provided claim framing and propose specific evidence types:\n"
            "- primary sources (if applicable), reputable secondary sources\n"
            "- quantitative data / official statistics when possible\n"
            "- direct quotations or documented results\n"
            "Then write a structured list:\n"
            "support_points:\n"
            "- <point>\n"
            "evidence_hypotheses:\n"
            "- <what source would confirm this>\n"
            "search_queries:\n"
            "- <one query per line>\n"
            "confidence_assumptions:\n"
            "- <what you assume about data availability>\n"
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # 4) Contradicting evidence persona (skeptic / con)
    contradict = AgentNode(
        name="contradict",
        llm=llm,
        node_prompt=(
            "You are the CONTRADICTING EVIDENCE analyst.\n"
            "Goal: gather strong evidence that the claim is FALSE or unsupported.\n"
            "Use the provided claim framing and propose specific evidence types:\n"
            "- official reports/statistics that disagree\n"
            "- methodological critiques / counterexamples\n"
            "- documents that limit or contradict the claim\n"
            "Then write a structured list:\n"
            "contradiction_points:\n"
            "- <point>\n"
            "evidence_hypotheses:\n"
            "- <what source would refute this>\n"
            "search_queries:\n"
            "- <one query per line>\n"
            "confidence_assumptions:\n"
            "- <what you assume about data availability>\n"
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # 5) Verdict node: weighs supporting vs contradicting evidence and gives a short rationale
    verdict = AgentNode(
        name="verdict",
        llm=llm,
        node_prompt=(
            "You are the chair issuing a verdict on a factual claim.\n"
            "You will be given:\n"
            "- a decision-ready claim framing\n"
            "- SUPPORTING evidence analysis\n"
            "- CONTRADICTING evidence analysis\n\n"
            "Task:\n"
            "1) Decide the verdict label: True / False / Unclear\n"
            "2) Provide a short rationale (max 5 sentences) grounded in the evidence points.\n"
            "3) Include 'what would change your mind' with 1-2 bullets.\n\n"
            "Return exactly this format:\n"
            "verdict: <True|False|Unclear>\n"
            "rationale: <short rationale>\n"
            "change_mind:\n"
            "- <bullet 1>\n"
            "- <bullet 2>\n"
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # Wire: frame -> (depth decision) -> parallel evidence -> verdict
    # Both depth choices still run the same evidence analysts; the analysts handle depth internally.
    decide_depth["light"] > fanout(support, contradict) > verdict
    decide_depth["deep"] > fanout(support, contradict) > verdict

    return AgenticGraph(start_node=frame, end_nodes={verdict})
