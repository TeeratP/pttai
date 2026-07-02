"""Generated pipeline for task 'interview_grader' (sample 2).

TASK: Grade a candidate's interview answer on several rubric dimensions and give an overall hire / no-hire recommendation.
"""

from pttai import AgentNode, DecisionNode, AgenticGraph, fanout, ConditionNode


def build_graph(llm):
    # 1) Frame the candidate's answer into a single evaluation target
    frame = AgentNode(
        name="frame",
        llm=llm,
        node_prompt=(
            "You are an interview rubric grader.\n"
            "Given:\n"
            "1) rubric dimensions (a list of dimensions, each with criteria),\n"
            "2) the job description (optional but helpful),\n"
            "3) the candidate's interview answer,\n\n"
            "Produce a structured evaluation task description for the graders.\n"
            "Requirements:\n"
            "- Extract the question/prompt being answered.\n"
            "- Restate the rubric dimensions and criteria succinctly.\n"
            "- Identify any missing info the graders should flag.\n"
            "- Ensure every rubric dimension is addressed by subsequent steps.\n\n"
            "Use the following state keys as context.\n"
            "Candidate answer: {candidate_answer}\n"
            "Rubric dimensions: {rubric_dimensions}\n"
            "Job description: {job_description}\n"
            "Question/prompt: {interview_question}\n"
        ),
        reads=["candidate_answer", "rubric_dimensions", "job_description", "interview_question"],
        writes={"evaluation_task": str},
    )

    # 2) Create rubric-specific graders in parallel
    make_dimension_prompt = (
        "You are a strict interview grader for one rubric dimension.\n\n"
        "Rubric dimension:\n"
        "{dimension}\n\n"
        "Evaluation task:\n"
        "{evaluation_task}\n\n"
        "Candidate answer:\n"
        "{candidate_answer}\n\n"
        "Grade this dimension using:\n"
        "- score: 0-5 (5 best) as an integer\n"
        "- evidence: 2-5 specific quotes or precise references to the candidate's answer\n"
        "- gaps: what is missing or incorrect relative to the dimension\n"
        "- actionable_feedback: one concise improvement suggestion\n\n"
        "Return ONLY a single JSON object with keys: score, evidence, gaps, actionable_feedback."
    )

    # Critics (one per dimension). We wire them with tools-free agents and rely on the
    # input placeholders/reads provided by the caller.
    # Note: we assume caller seeds dimension-specific prompts as state keys.
    # If not provided, graders will still work but may produce generic feedback.
    grade_communication = AgentNode(
        name="grade_communication",
        llm=llm,
        node_prompt=make_dimension_prompt,
        reads=["evaluation_task", "candidate_answer", "dimension_communication"],
        writes={"communication": str},
    )

    grade_technical = AgentNode(
        name="grade_technical",
        llm=llm,
        node_prompt=make_dimension_prompt,
        reads=["evaluation_task", "candidate_answer", "dimension_technical"],
        writes={"technical": str},
    )

    grade_problem_solving = AgentNode(
        name="grade_problem_solving",
        llm=llm,
        node_prompt=make_dimension_prompt,
        reads=["evaluation_task", "candidate_answer", "dimension_problem_solving"],
        writes={"problem_solving": str},
    )

    grade_ownership = AgentNode(
        name="grade_ownership",
        llm=llm,
        node_prompt=make_dimension_prompt,
        reads=["evaluation_task", "candidate_answer", "dimension_ownership"],
        writes={"ownership": str},
    )

    # 3) Aggregate into an overall summary with a recommendation rationale
    aggregate = AgentNode(
        name="aggregate",
        llm=llm,
        node_prompt=(
            "You are the panel chair. Combine the rubric dimension grades into an overall evaluation.\n\n"
            "Dimension grades (JSON strings):\n"
            "- communication: {communication}\n"
            "- technical: {technical}\n"
            "- problem_solving: {problem_solving}\n"
            "- ownership: {ownership}\n\n"
            "Requirements:\n"
            "- Provide an overall summary in 8-12 sentences.\n"
            "- Compute an overall_score as an integer 0-20 (sum of dimension scores).\n"
            "- Provide key strengths (2 bullets) and key concerns (2 bullets).\n"
            "- Recommend: hire_or_no_hire = either \"HIRE\" or \"NO_HIRE\".\n"
            "- Provide recommendation_reasoning: 3-6 sentences tied to the rubric.\n\n"
            "Return ONLY a single JSON object with keys: overall_score, hire_or_no_hire, recommendation_reasoning."
        ),
        reads=["communication", "technical", "problem_solving", "ownership"],
        writes={"final_evaluation": str},
    )

    # 4) Optional deterministic safety gate: ensure recommendation is consistent with overall score thresholds
    # This prevents malformed outputs from the aggregator from slipping through.
    def score_gate(state):
        # Expect final_evaluation to be a JSON string; if parsing fails, default to NO_HIRE.
        # Deterministic gating uses only presence/format heuristics; model output is still the source of truth.
        text = state.get("final_evaluation", "") or ""
        # Heuristic: if it contains "NO_HIRE" keep it, if "HIRE" but overall_score low, flip.
        hire = '"hire_or_no_hire": "HIRE"' in text
        nohire = '"hire_or_no_hire": "NO_HIRE"' in text
        # naive extraction of overall_score number
        score = None
        import re

        m = re.search(r'"overall_score"\s*:\s*(\d+)', text)
        if m:
            score = int(m.group(1))
        if nohire:
            return "no_hire"
        if hire:
            if score is not None and score >= 14:
                return "hire"
            return "no_hire"
        return "no_hire"

    gate = ConditionNode(
        name="gate",
        condition=score_gate,
        choices=["hire", "no_hire"],
        reads=["final_evaluation"],
        cache_ttl=None,
        retry=False,
    )

    # Terminal nodes: refine final output for downstream consumption
    finalize_hire = AgentNode(
        name="finalize_hire",
        llm=llm,
        node_prompt=(
            "Produce the final grading report for an upstream system.\n"
            "Inputs:\n"
            "- final_evaluation: {final_evaluation}\n\n"
            "Return ONLY JSON with keys: overall_score, recommendation (\"HIRE\"), report.\n"
            "The report should be a concise, recruiter-ready narrative (max 10 sentences)."
        ),
        reads=["final_evaluation"],
        writes={"final_report": str},
    )

    finalize_no_hire = AgentNode(
        name="finalize_no_hire",
        llm=llm,
        node_prompt=(
            "Produce the final grading report for an upstream system.\n"
            "Inputs:\n"
            "- final_evaluation: {final_evaluation}\n\n"
            "Return ONLY JSON with keys: overall_score, recommendation (\"NO_HIRE\"), report.\n"
            "The report should be a concise, recruiter-ready narrative (max 10 sentences)."
        ),
        reads=["final_evaluation"],
        writes={"final_report": str},
    )

    # Wiring
    # sequential: frame then parallel dimension graders then aggregate then gate
    frame > fanout(
        grade_communication,
        grade_technical,
        grade_problem_solving,
        grade_ownership,
    ) > aggregate > gate["hire"] > finalize_hire
    gate["no_hire"] > finalize_no_hire

    return AgenticGraph(start_node=frame, end_nodes={finalize_hire, finalize_no_hire})
