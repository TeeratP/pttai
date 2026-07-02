"""Generated pipeline for task 'interview_grader' (sample 4).

TASK: Grade a candidate's interview answer on several rubric dimensions and give an overall hire / no-hire recommendation.
"""

from typing import Dict, List

from pttai import AgentNode, AgenticGraph, ConditionNode, DecisionNode, fanout
from pttai import AgenticState


def build_graph(llm):
    # 1) Frame: convert candidate answer + job context into a single "grade request"
    frame = AgentNode(
        name="frame",
        llm=llm,
        reads=["role", "rubrics", "question", "candidate_answer"],
        writes={"grade_request": str},
        node_prompt=(
            "You are an interview grader assistant.\n\n"
            "Job role: {role}\n"
            "Question asked: {question}\n"
            "Rubric dimensions (JSON-like text): {rubrics}\n"
            "Candidate answer:\n{candidate_answer}\n\n"
            "Task: Produce a compact 'grade_request' that includes:\n"
            "- A list of rubric dimensions to grade (one line each)\n"
            "- What evidence in the candidate answer should be looked for for EACH dimension\n"
            "- Any scoring scale implied by the rubric text (e.g., 1-5) or explicitly state 'use rubric text as authority'\n"
            "- A note about whether the candidate addressed the question directly\n\n"
            "Return ONLY the grade_request text."
        ),
        retry=False,
    )

    # 2) Judge per rubric dimension (parallel)
    # Expect rubric dimensions as a list-like string in reads["rubrics"].
    # Each worker independently produces a numeric-ish score and justification text.
    # Note: We deliberately let the LLM parse the rubrics rather than enforce strict schema types.
    def _worker_prompt(dimension_placeholder: str) -> str:
        return (
            "You are a meticulous rubric scorer.\n\n"
            "GRADE REQUEST:\n{grade_request}\n\n"
            f"RUBRIC DIMENSION TO SCORE: {dimension_placeholder}\n\n"
            "From the candidate answer evidence described in the grade_request, score THIS dimension.\n"
            "Output format (STRICT):\n"
            "DIMENSION: <exact dimension name>\n"
            "SCORE: <number or label exactly matching rubric scale>\n"
            "JUSTIFICATION: <2-6 sentences referencing specific evidence>\n"
            "RED FLAGS: <bullets or 'None'>\n"
            "STRENGTHS: <bullets or 'None'>\n"
        )

    # We will ask an LLM to extract dimensions first (so the workers can be fixed set via decision).
    extract_dims = AgentNode(
        name="extract_dims",
        llm=llm,
        reads=["grade_request"],
        writes={"dimensions": str},
        node_prompt=(
            "Extract the rubric dimension names to score.\n\n"
            "GRADE REQUEST:\n{grade_request}\n\n"
            "Return ONLY JSON with shape:\n"
            '{"dimensions":[<string dimension names>]}'
        ),
        retry=False,
    )

    # Since pttai wiring is static, we will assume at most 5 dimensions and have the LLM return up to 5.
    # We route with DecisionNode based on presence/number is not directly supported.
    # Instead, we score all five slots; if fewer exist, the workers will self-handle with 'N/A'.
    dim1_score = AgentNode(
        name="dim1_score",
        llm=llm,
        reads=["grade_request", "dimensions"],
        writes={"dim1": str},
        node_prompt=_worker_prompt("Slot 1 (first dimension from dimensions JSON)"),
        retry=False,
    )
    dim2_score = AgentNode(
        name="dim2_score",
        llm=llm,
        reads=["grade_request", "dimensions"],
        writes={"dim2": str},
        node_prompt=_worker_prompt("Slot 2 (second dimension from dimensions JSON)"),
        retry=False,
    )
    dim3_score = AgentNode(
        name="dim3_score",
        llm=llm,
        reads=["grade_request", "dimensions"],
        writes={"dim3": str},
        node_prompt=_worker_prompt("Slot 3 (third dimension from dimensions JSON)"),
        retry=False,
    )
    dim4_score = AgentNode(
        name="dim4_score",
        llm=llm,
        reads=["grade_request", "dimensions"],
        writes={"dim4": str},
        node_prompt=_worker_prompt("Slot 4 (fourth dimension from dimensions JSON)"),
        retry=False,
    )
    dim5_score = AgentNode(
        name="dim5_score",
        llm=llm,
        reads=["grade_request", "dimensions"],
        writes={"dim5": str},
        node_prompt=_worker_prompt("Slot 5 (fifth dimension from dimensions JSON)"),
        retry=False,
    )

    # 3) Aggregate: combine all dimension scores into an overall recommendation
    aggregate = AgentNode(
        name="aggregate",
        llm=llm,
        reads=["role", "question", "candidate_answer", "grade_request", "dimensions", "dim1", "dim2", "dim3", "dim4", "dim5"],
        writes={"grading_report": str},
        node_prompt=(
            "You are the chair of an interview panel.\n\n"
            "ROLE: {role}\n"
            "QUESTION: {question}\n\n"
            "GRADE REQUEST:\n{grade_request}\n\n"
            "CANDIDATE ANSWER:\n{candidate_answer}\n\n"
            "DIMENSIONS JSON:\n{dimensions}\n\n"
            "Per-dimension scoring outputs (may include N/A if fewer dimensions exist):\n"
            "DIM1:\n{dim1}\n\n"
            "DIM2:\n{dim2}\n\n"
            "DIM3:\n{dim3}\n\n"
            "DIM4:\n{dim4}\n\n"
            "DIM5:\n{dim5}\n\n"
            "Now produce a STRICT grading report:\n"
            "1) Summary (max 5 sentences): what the candidate did well / poorly.\n"
            "2) Dimension-by-dimension: for each dimension present, include SCORE and JUSTIFICATION.\n"
            "3) Overall recommendation:\n"
            "- 'HIRE' or 'NO-HIRE'\n"
            "- A one-paragraph rationale referencing the rubric dimensions\n"
            "- If applicable: what the candidate would need to demonstrate to reverse a NO-HIRE\n\n"
            "Output ONLY the report text (no extra commentary)."
        ),
        retry=False,
    )

    # 4) Quality gate (deterministic): if report lacks a required marker, request regeneration.
    # We do this with ConditionNode on the already-created report text.
    def quality_gate(state: Dict) -> str:
        report = (state.get("grading_report") or "").strip()
        if "OVERALL" in report or "HIRE" in report or "NO-HIRE" in report:
            return "ok"
        return "retry"

    quality = ConditionNode(
        name="quality",
        condition=quality_gate,
        choices=["ok", "retry"],
        reads=["grading_report"],
        retry=False,
    )

    # Rework node for retry path
    rework = AgentNode(
        name="rework",
        llm=llm,
        reads=["role", "question", "candidate_answer", "grade_request", "dimensions", "dim1", "dim2", "dim3", "dim4", "dim5", "grading_report"],
        writes={"grading_report": str},
        node_prompt=(
            "Your previous report may have missed required sections or markers.\n"
            "Please regenerate the grading report with the following requirements:\n"
            "- Must include an explicit 'HIRE' or 'NO-HIRE'\n"
            "- Must include a clear rationale paragraph\n"
            "- Must include dimension-by-dimension scores for the dimensions present\n\n"
            "ROLE: {role}\n"
            "QUESTION: {question}\n\n"
            "CANDIDATE ANSWER:\n{candidate_answer}\n\n"
            "GRADE REQUEST:\n{grade_request}\n\n"
            "DIMENSIONS JSON:\n{dimensions}\n\n"
            "DIM1:\n{dim1}\n\n"
            "DIM2:\n{dim2}\n\n"
            "DIM3:\n{dim3}\n\n"
            "DIM4:\n{dim4}\n\n"
            "DIM5:\n{dim5}\n\n"
            "PREVIOUS REPORT:\n{grading_report}\n\n"
            "Now output ONLY the corrected report text."
        ),
        retry=False,
    )

    # Use an AgentNode to decide hire/no-hire label robustly (LLM),
    # but still keep overall structure deterministic by the quality gate.
    # This helps produce a clean final marker for downstream use.
    finalize = DecisionNode(
        name="finalize_decision",
        llm=llm,
        node_prompt=(
            "Read the grading report and extract the final decision.\n"
            "If the report indicates the candidate should be hired, choose 'hire'.\n"
            "Otherwise choose 'no_hire'.\n\n"
            "GRADING REPORT:\n{grading_report}"
        ),
        choices=["hire", "no_hire"],
        reads=["grading_report"],
        retry=False,
    )

    # Terminal nodes (end) for both outcomes, each writes a short overall field
    hire_end = AgentNode(
        name="hire_end",
        llm=llm,
        reads=["grading_report"],
        writes={"recommendation": str},
        node_prompt=(
            "Return ONLY: recommendation='HIRE' and include a short 1-2 sentence justification.\n\n"
            "GRADING REPORT:\n{grading_report}"
        ),
        retry=False,
    )

    nohire_end = AgentNode(
        name="nohire_end",
        llm=llm,
        reads=["grading_report"],
        writes={"recommendation": str},
        node_prompt=(
            "Return ONLY: recommendation='NO-HIRE' and include a short 1-2 sentence justification.\n\n"
            "GRADING REPORT:\n{grading_report}"
        ),
        retry=False,
    )

    # Wire graph:
    # frame -> extract_dims -> parallel dimension scorers -> aggregate -> quality gate
    # quality ok -> finalize_decision -> hire/nohire ends
    # quality retry -> rework -> finalize_decision -> hire/nohire ends
    start = frame > extract_dims
    start > fanout(dim1_score, dim2_score, dim3_score, dim4_score, dim5_score) > aggregate
    aggregate > quality
    quality["ok"] > finalize
    quality["retry"] > rework > finalize
    finalize["hire"] > hire_end
    finalize["no_hire"] > nohire_end

    return AgenticGraph(start_node=frame, end_nodes={hire_end, nohire_end})
