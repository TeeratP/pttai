"""Generated pipeline for task 'change_review_route' (sample 3).

TASK: Read a description of a code change, decide whether it is a bug fix, a new feature, or a refactor, and route to a reviewer that writes fitting feedback.
"""

from pttai import AgentNode, DecisionNode, AgenticGraph, fanout


def build_graph(llm):
    # 1) Frame the change description into a clean classification decision.
    frame = AgentNode(
        name="frame",
        llm=llm,
        node_prompt=(
            "You are a strict triage assistant.\n"
            "Given a code change description, classify it into exactly one category:\n"
            "bugfix | feature | refactor.\n"
            "Return ONLY the category name.\n"
            "Rules:\n"
            "- bugfix: fixes a defect/regression/incorrect behavior\n"
            "- feature: adds user-visible capability, behavior, or new interface\n"
            "- refactor: code restructuring with no intended behavior change\n"
            "Input will be provided as the conversation.\n"
            "Category must be unambiguous."
        ),
    )

    # 2) Route based on the framed category using an LLM constrained decision.
    route_change = DecisionNode(
        name="route_change",
        llm=llm,
        node_prompt="Classify the code change as bugfix, feature, or refactor. Return one choice.",
        choices=["bugfix", "feature", "refactor"],
    )

    # 3) Reviewer personas produce fitting feedback.
    bugfix_reviewer = AgentNode(
        name="bugfix_reviewer",
        llm=llm,
        node_prompt=(
            "You are a code reviewer specializing in bug fixes.\n"
            "Write fitting feedback for the provided change description.\n"
            "Focus on:\n"
            "- what the bug was and why this fix works\n"
            "- regression risk and edge cases\n"
            "- tests to add/adjust (include concrete test ideas)\n"
            "- safety checks and observability\n"
            "Be concise but specific. Provide actionable bullet points."
        ),
    )

    feature_reviewer = AgentNode(
        name="feature_reviewer",
        llm=llm,
        node_prompt=(
            "You are a code reviewer specializing in new features.\n"
            "Write fitting feedback for the provided change description.\n"
            "Focus on:\n"
            "- expected user-facing behavior / API contract\n"
            "- compatibility and migration concerns\n"
            "- correctness, performance, and security implications\n"
            "- documentation/usage examples and test plan\n"
            "Be concise but specific. Provide actionable bullet points."
        ),
    )

    refactor_reviewer = AgentNode(
        name="refactor_reviewer",
        llm=llm,
        node_prompt=(
            "You are a code reviewer specializing in refactors.\n"
            "Write fitting feedback for the provided change description.\n"
            "Focus on:\n"
            "- confirm no intended behavior change; call out any behavioral risks\n"
            "- readability/maintainability improvements\n"
            "- invariants preserved; potential subtle regressions\n"
            "- tests needed to prove equivalence\n"
            "Be concise but specific. Provide actionable bullet points."
        ),
    )

    # 4) Combine: weigh the chosen reviewer feedback into a single answer.
    # (Even though only one reviewer should run, this step standardizes the output.)
    verdict = AgentNode(
        name="verdict",
        llm=llm,
        node_prompt=(
            "You are the reviewer lead.\n"
            "Given the selected reviewer feedback, produce a single final response.\n"
            "Output should be:\n"
            "1) A one-sentence summary\n"
            "2) Bullet-point actionable feedback\n"
            "3) Suggested tests (bullet points)\n"
            "Keep it tight and practical."
        ),
    )

    # Wiring
    frame > route_change
    route_change["bugfix"] > bugfix_reviewer
    route_change["feature"] > feature_reviewer
    route_change["refactor"] > refactor_reviewer

    # The join for whichever branch was chosen; we join by wiring all reviewers to verdict.
    fanout(bugfix_reviewer, feature_reviewer, refactor_reviewer) > verdict

    return AgenticGraph(start_node=frame, end_nodes={verdict})
