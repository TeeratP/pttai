"""Generated pipeline for task 'interview_grader' (sample 1).

TASK: Grade a candidate's interview answer on several rubric dimensions and give an overall hire / no-hire recommendation.
"""

from pttai import AgentNode, DecisionNode, ConditionNode, AgenticGraph, fanout


def build_graph(llm):
    # --- Stage 1: frame the grading rubric into a sharp decision ---
    frame = AgentNode(
        name="frame",
        llm=llm,
        node_prompt=(
            "You are an interview grader.\n"
            "Task: Grade a candidate's interview answer against multiple rubric dimensions.\n\n"
            "Given the input messages (which include a rubric and the candidate's answer), produce:\n"
            "1) a concise rubric-weighted grading plan,\n"
            "2) a checklist of what evidence to look for per dimension,\n"
            "3) an extraction schema: what to quote or reference from the candidate's answer for each dimension,\n"
            "4) the exact set of dimensions to score if present.\n\n"
            "Output instructions:\n"
            "- Be specific and action-oriented.\n"
            "- Do NOT produce final scores yet.\n"
            "- Keep it compact."
        ),
    )

    # --- Stage 2: parallel rubric critics ---
    clarity = AgentNode(
        name="clarity",
        llm=llm,
        node_prompt=(
            "Rubric dimension: Clarity/Communication.\n"
            "Using the frame plan and the provided rubric + candidate answer, extract evidence and assess:\n"
            "- Does the candidate explain clearly and logically?\n"
            "- Are terms defined and structure used (e.g., steps, bullets, example)?\n\n"
            "Return only:\n"
            "A JSON object with keys: justification (string), score (int 1-5), evidence (list of short quotes)."
        ),
    )

    correctness = AgentNode(
        name="correctness",
        llm=llm,
        node_prompt=(
            "Rubric dimension: Technical Correctness/Reasoning Quality.\n"
            "Using the frame plan and the provided rubric + candidate answer, extract evidence and assess:\n"
            "- Are claims accurate?\n"
            "- Is reasoning sound?\n"
            "- Are assumptions stated and validated?\n\n"
            "Return only:\n"
            "A JSON object with keys: justification (string), score (int 1-5), evidence (list of short quotes)."
        ),
    )

    completeness = AgentNode(
        name="completeness",
        llm=llm,
        node_prompt=(
            "Rubric dimension: Completeness/Depth.\n"
            "Using the frame plan and the provided rubric + candidate answer, extract evidence and assess:\n"
            "- Did they address all aspects of the question?\n"
            "- Did they go deep enough (tradeoffs, edge cases, complexity)?\n"
            "- Did they avoid major omissions?\n\n"
            "Return only:\n"
            "A JSON object with keys: justification (string), score (int 1-5), evidence (list of short quotes)."
        ),
    )

    impact = AgentNode(
        name="impact",
        llm=llm,
        node_prompt=(
            "Rubric dimension: Practical Impact/Execution Mindset.\n"
            "Using the frame plan and the provided rubric + candidate answer, extract evidence and assess:\n"
            "- Are recommendations actionable?\n"
            "- Do they show ownership, outcomes, and measurable thinking?\n"
            "- Do they mention constraints and realistic next steps?\n\n"
            "Return only:\n"
            "A JSON object with keys: justification (string), score (int 1-5), evidence (list of short quotes)."
        ),
    )

    # --- Stage 3: synthesize into final scores and recommendation ---
    verdict = AgentNode(
        name="verdict",
        llm=llm,
        node_prompt=(
            "You are the chair of an interview grading panel.\n"
            "Inputs are:\n"
            "- frame (grading plan)\n"
            "- clarity / correctness / completeness / impact critic outputs (each is a JSON object)\n"
            "Use the provided rubric if it includes weights.\n\n"
            "Produce:\n"
            "1) A final JSON object with keys:\n"
            "   - scores: object mapping dimension -> int 1-5\n"
            "   - dimension_justifications: object mapping dimension -> string\n"
            "   - evidence_quotes: object mapping dimension -> list of short quotes\n"
            "   - overall_recommendation: one of [HIRE, NO_HIRE]\n"
            "   - overall_justification: string (1 paragraph)\n"
            "2) Follow rubric weights if present.\n\n"
            "Return ONLY the final JSON object."
        ),
    )

    # --- Stage 4: ensure output is correct and route into hire/no-hire label ---
    normalize = DecisionNode(
        name="normalize",
        llm=llm,
        node_prompt=(
            "Normalize the grading output.\n"
            "The input is the verdict JSON object from the chair.\n"
            "If it contains overall_recommendation, map it to one of the choices exactly.\n"
            "Also ensure the final JSON is parseable and consistent.\n\n"
            "Return ONLY a selection among the choices; the selection is the field 'decision'."
        ),
        choices=["HIRE", "NO_HIRE"],
    )

    to_hire = AgentNode(
        name="to_hire",
        llm=llm,
        node_prompt=(
            "Produce the final user-facing grading report for HIRE.\n"
            "Use the state messages (which include the chair's JSON) to format:\n"
            "- Overall: HIRE\n"
            "- Scores table (dimensions + 1-5)\n"
            "- Key justifications per dimension\n"
            "- 2-5 evidence quotes supporting the decision\n\n"
            "Keep it concise and professional."
        ),
    )

    to_no_hire = AgentNode(
        name="to_no_hire",
        llm=llm,
        node_prompt=(
            "Produce the final user-facing grading report for NO_HIRE.\n"
            "Use the state messages (which include the chair's JSON) to format:\n"
            "- Overall: NO_HIRE\n"
            "- Scores table (dimensions + 1-5)\n"
            "- Key justifications per dimension\n"
            "- 2-5 evidence quotes supporting the decision\n\n"
            "Keep it concise and professional."
        ),
    )

    # --- Optional deterministic guard: if rubric is missing, treat as NO_HIRE ---
    # This makes the pipeline more robust to bad inputs without adding extra LLM calls.
    has_rubric = ConditionNode(
        name="has_rubric",
        condition=lambda state: "ok"
        if ("rubric" in "".join(m.content for m in state.get("messages", []) if hasattr(m, "content")).lower())
        else "missing",
        choices=["ok", "missing"],
        reads=["messages"],
    )

    # If rubric is missing, immediately return NO_HIRE via a deterministic path.
    missing_rubric = AgentNode(
        name="missing_rubric",
        llm=llm,
        node_prompt=(
            "Rubric is missing from the input. Provide a grading report with overall_recommendation: NO_HIRE.\n"
            "Explain that grading cannot be completed reliably without the rubric.\n"
            "Keep it short."
        ),
    )

    # Wiring:
    # start -> has_rubric -> (missing_rubric OR grading panel)
    # grading panel: frame -> parallel critics -> verdict -> normalize -> hire/no-hire
    #
    # Note: ConditionNode routes by choice; ensure all choices are wired.
    #
    panel = frame > fanout(clarity, correctness, completeness, impact) > verdict > normalize
    panel["HIRE"] > to_hire
    panel["NO_HIRE"] > to_no_hire

    has_rubric["ok"] > panel
    has_rubric["missing"] > missing_rubric

    return AgenticGraph(start_node=has_rubric, end_nodes={to_hire, to_no_hire, missing_rubric})
