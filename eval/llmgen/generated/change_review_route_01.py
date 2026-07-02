"""Generated pipeline for task 'change_review_route' (sample 1).

TASK: Read a description of a code change, decide whether it is a bug fix, a new feature, or a refactor, and route to a reviewer that writes fitting feedback.
"""

from pttai import AgentNode, DecisionNode, AgenticGraph


def build_graph(llm):
    frame = AgentNode(
        name="frame",
        llm=llm,
        node_prompt=(
            "Classify the incoming code change description into ONE of: "
            "bug_fix, new_feature, or refactor.\n\n"
            "Rules:\n"
            "- bug_fix: fixes incorrect behavior, crashes, security issues, or wrong outputs.\n"
            "- new_feature: adds user-visible functionality or capability.\n"
            "- refactor: restructures code without changing externally observable behavior.\n\n"
            "Return the decision as your best judgment; be consistent with the rules."
        ),
        reads=["messages"],
        writes=["messages"],
    )

    router = DecisionNode(
        name="router",
        llm=llm,
        node_prompt=(
            "You are a triage classifier. Based on the conversation, select exactly one label "
            "among {choices} describing the type of code change.\n\n"
            "Label meanings:\n"
            "- bug_fix: fixes incorrect behavior/crashes/security/wrong outputs.\n"
            "- new_feature: adds user-visible functionality/capability.\n"
            "- refactor: code restructuring w/o external behavior change.\n"
            "Respond with one valid choice only."
        ),
        choices=["bug_fix", "new_feature", "refactor"],
        input_field="messages",
    )

    bug_reviewer = AgentNode(
        name="bug_reviewer",
        llm=llm,
        node_prompt=(
            "You are a senior bug-fix reviewer.\n\n"
            "Write fitting feedback for a proposed bug fix:\n"
            "- identify likely root cause(s)\n"
            "- check edge cases and regression risks\n"
            "- verify tests/additions needed\n"
            "- suggest any observability or safety improvements\n\n"
            "Be concrete and specific to the described change."
        ),
        reads=["messages"],
        writes=["messages"],
    )

    feature_reviewer = AgentNode(
        name="feature_reviewer",
        llm=llm,
        node_prompt=(
            "You are a product-focused feature reviewer.\n\n"
            "Write fitting feedback for a proposed new feature:\n"
            "- clarify intended user value and scope\n"
            "- check UX/API consistency and backward compatibility\n"
            "- list acceptance criteria and test plan\n"
            "- call out performance, reliability, and security considerations\n\n"
            "Be concrete and specific to the described change."
        ),
        reads=["messages"],
        writes=["messages"],
    )

    refactor_reviewer = AgentNode(
        name="refactor_reviewer",
        llm=llm,
        node_prompt=(
            "You are a refactor reviewer focused on correctness and maintainability.\n\n"
            "Write fitting feedback for a proposed refactor:\n"
            "- confirm external behavior preservation\n"
            "- highlight risks where semantics might accidentally change\n"
            "- suggest improvements to structure, naming, and documentation\n"
            "- note needed tests to prove behavior parity\n\n"
            "Be concrete and specific to the described change."
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # Frame -> Router -> {choice reviewer}
    frame > router
    router["bug_fix"] > bug_reviewer
    router["new_feature"] > feature_reviewer
    router["refactor"] > refactor_reviewer

    return AgenticGraph(
        start_node=frame,
        end_nodes={bug_reviewer, feature_reviewer, refactor_reviewer},
    )
