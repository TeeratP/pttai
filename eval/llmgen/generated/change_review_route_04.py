"""Generated pipeline for task 'change_review_route' (sample 4).

TASK: Read a description of a code change, decide whether it is a bug fix, a new feature, or a refactor, and route to a reviewer that writes fitting feedback.
"""

from pttai import AgentNode, DecisionNode, AgenticGraph, fanout


def build_graph(llm):
    """
    Classify a code change description as: bug fix / new feature / refactor,
    then route to the corresponding reviewer persona who writes fitting feedback.
    Expects input on `messages` (default) via graph.invoke(message="...") or invoke({...}).
    Returns the reviewer's final message (in `messages[-1]`).
    """

    frame = AgentNode(
        name="frame",
        llm=llm,
        node_prompt=(
            "You are a careful software engineering triage assistant.\n"
            "Given the user's description of a code change, extract:\n"
            "1) the core intent (what is being changed),\n"
            "2) any behavioral impact (what changes for users/clients),\n"
            "3) whether it fixes a defect, adds capability, or reorganizes code without changing behavior.\n"
            "Write the result as concise bullets."
        ),
        # keep default reads/writes: messages -> messages
    )

    classify = DecisionNode(
        name="classify",
        llm=llm,
        node_prompt=(
            "Classify the code change intent into exactly ONE category.\n\n"
            "Rules:\n"
            "- bug fix: fixes an existing defect / incorrect behavior (correctness, crash, regression).\n"
            "- new feature: adds user-visible capability or meaningful new behavior.\n"
            "- refactor: internal restructuring / cleanup with no intended behavioral change.\n\n"
            "Output only the category."
        ),
        choices=["bug_fix", "new_feature", "refactor"],
    )

    bug_reviewer = AgentNode(
        name="bug_reviewer",
        llm=llm,
        node_prompt=(
            "You are a meticulous reviewer for BUG FIX pull requests.\n\n"
            "Write fitting feedback for the change described so far:\n"
            "- Identify what was broken and why the fix should address it.\n"
            "- Call out edge cases, regression risk, and how to verify.\n"
            "- Suggest targeted tests (unit/integration) and any missing test cases.\n"
            "- Note improvements to error handling, observability, and documentation if relevant.\n\n"
            "Be concise but concrete. Output as review comments/bullets."
        ),
    )

    feature_reviewer = AgentNode(
        name="feature_reviewer",
        llm=llm,
        node_prompt=(
            "You are a thoughtful reviewer for NEW FEATURE pull requests.\n\n"
            "Write fitting feedback for the change described so far:\n"
            "- Explain the user-visible behavior added.\n"
            "- Check for completeness: API/UX, configuration, defaults, and documentation needs.\n"
            "- Call out design tradeoffs, backward compatibility, and performance impacts.\n"
            "- Suggest acceptance criteria and tests (including negative/edge cases).\n\n"
            "Be concise but concrete. Output as review comments/bullets."
        ),
    )

    refactor_reviewer = AgentNode(
        name="refactor_reviewer",
        llm=llm,
        node_prompt=(
            "You are a careful reviewer for REFACTOR pull requests.\n\n"
            "Write fitting feedback for the change described so far:\n"
            "- Confirm the intent is behavior-preserving restructuring.\n"
            "- Point out any potential behavioral differences (even subtle ones).\n"
            "- Suggest how to prove correctness (tests, benchmarks, migration notes if any).\n"
            "- Review readability/maintainability improvements and whether abstractions are justified.\n\n"
            "Be concise but concrete. Output as review comments/bullets."
        ),
    )

    # One-line topology: extract intent -> classify -> route to the matching reviewer.
    frame > fanout(classify)  # fanout kept minimal but valid; joins immediately
    classify["bug_fix"] > bug_reviewer
    classify["new_feature"] > feature_reviewer
    classify["refactor"] > refactor_reviewer

    return AgenticGraph(start_node=frame, end_nodes={bug_reviewer, feature_reviewer, refactor_reviewer})
