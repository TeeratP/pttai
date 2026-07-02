"""Generated pipeline for task 'interview_grader' (sample 3).

TASK: Grade a candidate's interview answer on several rubric dimensions and give an overall hire / no-hire recommendation.
"""

from typing import List

from pttai import AgentNode, AgenticGraph, DecisionNode, ConditionNode


def build_graph(llm):
    # 1) Prepare a single, deterministic grading artifact to reduce variance.
    frame = AgentNode(
        name="frame",
        llm=llm,
        node_prompt=(
            "You are an expert interview grader. You will grade a candidate's answer "
            "to a question across multiple rubric dimensions.\n\n"
            "Given:\n"
            "- question\n"
            "- rubric_dimensions (a list of dimensions, each with criteria)\n"
            "- candidate_answer\n\n"
            "Produce ONE concise grading package in the following format:\n"
            "GradingPackage:\n"
            "1) dimension_scores: a JSON-like object mapping each dimension name to "
            "an integer score from 1 to 5, where 1=poor and 5=excellent.\n"
            "2) dimension_evidence: a JSON-like object mapping each dimension name to "
            "2-4 bullet points of specific evidence from the candidate answer.\n"
            "3) overall_recommendation: one of [hire, no-hire].\n"
            "4) overall_rationale: 3-6 sentences explaining why.\n"
            "Rules:\n"
            "- Be consistent and factual; quote or closely paraphrase the candidate where possible.\n"
            "- If the answer is incomplete or missing required info, score accordingly.\n"
            "- overall_recommendation must align with the rubric and the evidence.\n"
            "Do NOT add anything outside the GradingPackage format."
        ),
        reads=["question", "rubric_dimensions", "candidate_answer"],
        writes={"messages": list},
    )

    # 2) Structured decision: hire/no-hire (LLM-constrained).
    decide = DecisionNode(
        name="decide",
        llm=llm,
        node_prompt=(
            "You are choosing hiring outcome based on a prepared GradingPackage. "
            "Return exactly one choice.\n\n"
            "GradingPackage:\n{messages}"
        ),
        choices=["hire", "no-hire"],
        input_field="messages",
        reads=["messages"],
        writes=None,  # DecisionNode writes to built-in `decision`
    )

    # 3) If hire -> craft final response.
    hire_builder = AgentNode(
        name="hire_builder",
        llm=llm,
        node_prompt=(
            "Produce the final JSON output for hiring=true.\n\n"
            "Use the GradingPackage below as ground truth:\n"
            "{messages}\n\n"
            "Output JSON with this schema (no extra keys):\n"
            "{\n"
            '  "recommendation": "hire",\n'
            '  "hiring": true,\n'
            '  "overall_rationale": string,\n'
            '  "dimension_scores": object<string, number>,\n'
            '  "dimension_evidence": object<string, array<string>>\n'
            "}\n"
            "Return only valid JSON."
        ),
        reads=["messages"],
        writes={"answer": str},
    )

    # 4) If no-hire -> craft final response.
    noh_builder = AgentNode(
        name="noh_builder",
        llm=llm,
        node_prompt=(
            "Produce the final JSON output for hiring=false.\n\n"
            "Use the GradingPackage below as ground truth:\n"
            "{messages}\n\n"
            "Output JSON with this schema (no extra keys):\n"
            "{\n"
            '  "recommendation": "no-hire",\n'
            '  "hiring": false,\n'
            '  "overall_rationale": string,\n'
            '  "dimension_scores": object<string, number>,\n'
            '  "dimension_evidence": object<string, array<string>>\n'
            "}\n"
            "Return only valid JSON."
        ),
        reads=["messages"],
        writes={"answer": str},
    )

    # 5) Optional guardrail: ensure we have an answer string (deterministic).
    # This reads `answer`; if missing/empty, it routes to a re-grader.
    def has_answer(state) -> str:
        a = state.get("answer", "")
        return "ok" if isinstance(a, str) and a.strip() else "missing"

    rerater = AgentNode(
        name="rerater",
        llm=llm,
        node_prompt=(
            "You failed to produce the required final JSON. "
            "Re-emit the final JSON output following the schema exactly.\n\n"
            "GradingPackage:\n{messages}\n"
        ),
        reads=["messages"],
        writes={"answer": str},
    )

    ensure = ConditionNode(
        name="ensure",
        condition=has_answer,
        choices=["ok", "missing"],
        reads=["answer", "messages"],
    )

    # Wiring:
    # frame -> decide (sets `decision`), then hire_builder/noh_builder
    # finally ensure answer; if missing, rerater.
    frame > decide["hire"] > hire_builder > ensure["ok"]
    frame > decide["no-hire"] > noh_builder > ensure["ok"]

    frame > decide["hire"] > hire_builder > ensure["missing"] > rerater
    frame > decide["no-hire"] > noh_builder > ensure["missing"] > rerater

    return AgenticGraph(start_node=frame, end_nodes={ensure, rerater})
