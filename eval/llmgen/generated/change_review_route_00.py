"""Generated pipeline for task 'change_review_route' (sample 0).

TASK: Read a description of a code change, decide whether it is a bug fix, a new feature, or a refactor, and route to a reviewer that writes fitting feedback.
"""

from pttai import AgentNode, DecisionNode, AgenticGraph, fanout


def build_graph(llm):
    # Frame the change description into a constrained routing decision.
    frame = AgentNode(
        name="frame",
        llm=llm,
        node_prompt=(
            "You are an expert software reviewer. Given the user's description of a code change, "
            "decide which category it best fits:\n"
            "- bugfix: fixes a defect/regression/security issue\n"
            "- feature: adds user-visible functionality or capability\n"
            "- refactor: restructures code without changing behavior\n\n"
            "Return ONLY the category label: bugfix, feature, or refactor.\n"
            "Description:\n{messages}"
        ),
        # Use the incoming `messages` as prompt content.
        reads=["messages"],
    )

    # Convert frame's category into a structured LLM router choice.
    # We use DecisionNode's structured output to guarantee valid routing keys.
    router = DecisionNode(
        name="router",
        llm=llm,
        node_prompt=(
            "Route the code-change description to the right reviewer.\n"
            "Choose exactly one of these reviewer roles:\n"
            "- bugfix_reviewer\n"
            "- feature_reviewer\n"
            "- refactor_reviewer\n\n"
            "Use the text to infer the correct category.\n"
            "Description:\n{messages}\n\n"
            "Category hint (may be present in the text): {messages}"
        ),
        choices=["bugfix_reviewer", "feature_reviewer", "refactor_reviewer"],
        reads=["messages"],
    )

    bugfix_reviewer = AgentNode(
        name="bugfix_reviewer",
        llm=llm,
        node_prompt=(
            "You are the bug-fix reviewer.\n"
            "Write fitting feedback for the described change:\n"
            "- Identify the specific bug/regression being fixed\n"
            "- Check for edge cases, correctness, and potential unintended side effects\n"
            "- Call out missing tests and suggest what to test\n"
            "- If security/performance is implicated, review that too\n\n"
            "Provide concise, actionable feedback."
        ),
        reads=["messages"],
    )

    feature_reviewer = AgentNode(
        name="feature_reviewer",
        llm=llm,
        node_prompt=(
            "You are the feature reviewer.\n"
            "Write fitting feedback for the described change:\n"
            "- Confirm the new capability and user-visible behavior\n"
            "- Check requirements completeness and acceptance criteria\n"
            "- Review API/UX consistency, backward compatibility, and documentation needs\n"
            "- Suggest tests (happy path + failure modes) and consider observability\n\n"
            "Provide concise, actionable feedback."
        ),
        reads=["messages"],
    )

    refactor_reviewer = AgentNode(
        name="refactor_reviewer",
        llm=llm,
        node_prompt=(
            "You are the refactor reviewer.\n"
            "Write fitting feedback for the described change:\n"
            "- Verify behavior preservation assumptions\n"
            "- Check for readability/maintainability improvements (or regressions)\n"
            "- Identify any subtle semantic changes, performance impacts, and migration risks\n"
            "- Suggest tests/verification steps to ensure functional equivalence\n\n"
            "Provide concise, actionable feedback."
        ),
        reads=["messages"],
    )

    # A final summarizer that ensures the output is just the feedback for the chosen reviewer(s).
    verdict = AgentNode(
        name="verdict",
        llm=llm,
        node_prompt=(
            "You are the chair. Combine the reviewer feedback into a single response.\n"
            "Return ONLY the final feedback text.\n"
        ),
        # Merge any reviewer messages into the final answer.
        reads=["messages"],
    )

    # Wire: frame -> router -> one of the reviewers -> verdict
    # (We run the chosen reviewer; the unused branches simply won't be routed to.)
    frame > router
    router["bugfix_reviewer"] > bugfix_reviewer > verdict
    router["feature_reviewer"] > feature_reviewer > verdict
    router["refactor_reviewer"] > refactor_reviewer > verdict

    return AgenticGraph(start_node=frame, end_nodes={verdict})
