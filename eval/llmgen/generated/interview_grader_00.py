"""Generated pipeline for task 'interview_grader' (sample 0).

TASK: Grade a candidate's interview answer on several rubric dimensions and give an overall hire / no-hire recommendation.
"""

from pttai import AgentNode, DecisionNode, AgenticGraph, fanout


def build_graph(llm):
    # 1) Frame the candidate evaluation as a sharp decision task.
    frame = AgentNode(
        name="frame",
        llm=llm,
        node_prompt=(
            "You are an expert interview grader. Convert the interviewer prompt into a checklist of rubric dimensions "
            "and evaluation instructions.\n\n"
            "Return ONLY an explicit, structured rubric description that includes:\n"
            "- technical competence (grounded in specifics)\n"
            "- communication clarity\n"
            "- problem solving / reasoning\n"
            "- collaboration / ownership\n"
            "- role fit / impact\n"
            "- evidence requirements (what counts vs what doesn't)\n"
            "- scoring guidance (e.g., 1-5) and how to handle missing/contradictory evidence\n\n"
            "Input provided in the messages.\n"
        ),
        # Reads messages so the system prompt can see the interview + rubric.
        reads=["messages"],
        writes=["messages"],
    )

    # 2) Concurrent rubric dimension graders.
    tech = AgentNode(
        name="tech",
        llm=llm,
        node_prompt=(
            "Grade the candidate's interview answer for TECHNICAL COMPETENCE using the rubric in the conversation.\n"
            "Output a single concise evaluation with:\n"
            "1) score (1-5)\n"
            "2) 2 strongest technical evidence points (quote or paraphrase specifics)\n"
            "3) 2 biggest gaps/concerns\n"
            "4) any uncertainty due to missing info\n"
        ),
        reads=["messages"],
        writes=["messages"],
    )

    comms = AgentNode(
        name="comms",
        llm=llm,
        node_prompt=(
            "Grade the candidate's interview answer for COMMUNICATION CLARITY using the rubric in the conversation.\n"
            "Output a single concise evaluation with:\n"
            "1) score (1-5)\n"
            "2) 2 strongest clarity/evidence points\n"
            "3) 2 biggest communication gaps (e.g., ambiguity, verbosity, mismatch)\n"
            "4) any uncertainty due to missing info\n"
        ),
        reads=["messages"],
        writes=["messages"],
    )

    reasoning = AgentNode(
        name="reasoning",
        llm=llm,
        node_prompt=(
            "Grade the candidate's interview answer for PROBLEM SOLVING / REASONING using the rubric in the conversation.\n"
            "Output a single concise evaluation with:\n"
            "1) score (1-5)\n"
            "2) 2 strongest reasoning/evidence points (how they structure, tradeoffs, testing)\n"
            "3) 2 biggest gaps/concerns\n"
            "4) any uncertainty due to missing info\n"
        ),
        reads=["messages"],
        writes=["messages"],
    )

    collab = AgentNode(
        name="collab",
        llm=llm,
        node_prompt=(
            "Grade the candidate's interview answer for COLLABORATION / OWNERSHIP using the rubric in the conversation.\n"
            "Output a single concise evaluation with:\n"
            "1) score (1-5)\n"
            "2) 2 strongest collaboration/ownership evidence points\n"
            "3) 2 biggest gaps/concerns\n"
            "4) any uncertainty due to missing info\n"
        ),
        reads=["messages"],
        writes=["messages"],
    )

    role_fit = AgentNode(
        name="role_fit",
        llm=llm,
        node_prompt=(
            "Grade the candidate's interview answer for ROLE FIT / IMPACT using the rubric in the conversation.\n"
            "Output a single concise evaluation with:\n"
            "1) score (1-5)\n"
            "2) 2 strongest role-fit / impact evidence points\n"
            "3) 2 biggest gaps/concerns\n"
            "4) any uncertainty due to missing info\n"
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # 3) Decision gate: require enough evidence; otherwise recommend no-hire or ask for more info.
    evidence_check = DecisionNode(
        name="evidence_check",
        llm=llm,
        node_prompt=(
            "Decide whether the available interview answer provides sufficient evidence to make a reliable hire/no-hire "
            "recommendation across the rubric dimensions.\n\n"
            "Consider:\n"
            "- Were concrete examples given?\n"
            "- Are there specific technical/worked outcomes?\n"
            "- Are there clear indications of reasoning/collaboration?\n"
            "- Or is the answer too generic/underspecified?\n\n"
            "Choose 'enough' if confident; otherwise choose 'not_enough'."
        ),
        choices=["enough", "not_enough"],
        reads=["messages"],
    )

    # 4) When evidence is not enough: produce a 'no-hire' or 'needs more info' style response.
    not_enough = AgentNode(
        name="not_enough",
        llm=llm,
        node_prompt=(
            "Given the graders' outputs (if any) and the conversation rubric, produce the final evaluation.\n\n"
            "The evidence was NOT sufficient. Recommend NO-HIRE (or 'no' decision) due to lack of reliable evidence.\n"
            "Output exactly:\n"
            "- Overall Recommendation: NO-HIRE\n"
            "- Dimension Scores: list each of the five rubric dimensions with its score if available\n"
            "- Summary: 3-5 sentences explaining what is missing and why it prevents a confident hire\n"
            "- Follow-up Questions: 3 targeted questions that would clarify the missing evidence\n"
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # 5) When evidence is enough: weigh graders into an overall recommendation.
    verdict = AgentNode(
        name="verdict",
        llm=llm,
        node_prompt=(
            "You are the hiring panel chair. Use all rubric-dimension evaluations in the conversation to compute an "
            "overall hire/no-hire recommendation.\n\n"
            "Rules:\n"
            "- Dimension scores are 1-5; if any dimension lacks a score, treat it as a weakness.\n"
            "- Provide a balanced summary: strongest strengths + biggest risks.\n"
            "- Ground claims in evidence points mentioned by the graders.\n\n"
            "Output exactly:\n"
            "- Overall Recommendation: HIRE or NO-HIRE\n"
            "- Dimension Scores: list each of the five rubric dimensions with its score\n"
            "- Summary: 3-5 sentences\n"
            "- Key Evidence: 3 bullet points\n"
            "- Risks / Open Questions: 2 bullets\n"
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # Wiring:
    # frame -> parallel graders -> evidence_check -> (enough/not_enough) -> terminal
    frame > fanout(tech, comms, reasoning, collab, role_fit) > evidence_check
    evidence_check["enough"] > verdict
    evidence_check["not_enough"] > not_enough

    return AgenticGraph(start_node=frame, end_nodes={verdict, not_enough})
