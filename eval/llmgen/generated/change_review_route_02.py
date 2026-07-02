"""Generated pipeline for task 'change_review_route' (sample 2).

TASK: Read a description of a code change, decide whether it is a bug fix, a new feature, or a refactor, and route to a reviewer that writes fitting feedback.
"""

from pttai import AgentNode, DecisionNode, AgenticGraph, fanout


def build_graph(llm):
    # 1) Frame the user's change request into a concrete decision:
    #    "bug_fix", "new_feature", or "refactor".
    frame = AgentNode(
        name="frame",
        llm=llm,
        node_prompt=(
            "You are a release triage assistant.\n"
            "Read the following code change description (in the user's message).\n"
            "Decide which category best fits:\n"
            "1) bug_fix: fixes an issue, defect, regression, incorrect behavior, or crash.\n"
            "2) new_feature: adds user-visible capability or significant new behavior.\n"
            "3) refactor: reorganizes internals without changing external behavior.\n"
            "Pick exactly ONE category and justify briefly.\n"
            "Return only the category label: bug_fix / new_feature / refactor."
        ),
    )

    # 2) Use an LLM DecisionNode to constrain routing to valid choices.
    #    It writes to state["decision"] and routes by choice.
    classify = DecisionNode(
        name="classify",
        llm=llm,
        node_prompt=(
            "Classify the code change description into one of:\n"
            "- bug_fix\n"
            "- new_feature\n"
            "- refactor\n\n"
            "Rules:\n"
            "- bug_fix: behavior is corrected or a defect/regression is addressed.\n"
            "- new_feature: new capability or user-visible behavior is introduced.\n"
            "- refactor: restructuring/cleanup with no intended external behavior change.\n\n"
            "Output only one valid label."
        ),
        choices=["bug_fix", "new_feature", "refactor"],
    )

    # 3) Three reviewer personas, run in parallel after classification,
    #    then a verdict reviewer writes the final fitting feedback.
    #    (We run all reviewers concurrently for better coverage; the final
    #     reviewer tailors feedback to the chosen category.)
    reviewer_bug = AgentNode(
        name="reviewer_bug",
        llm=llm,
        node_prompt=(
            "You are a strict bug-fix reviewer.\n"
            "Given the code change request, write fitting review feedback focused on:\n"
            "- root cause analysis (as inferable)\n"
            "- correctness and edge cases\n"
            "- regression risk\n"
            "- test suggestions\n"
            "Be concrete, actionable, and concise.\n"
        ),
    )

    reviewer_feature = AgentNode(
        name="reviewer_feature",
        llm=llm,
        node_prompt=(
            "You are a product-minded feature reviewer.\n"
            "Given the code change request, write fitting review feedback focused on:\n"
            "- expected user-visible behavior\n"
            "- spec clarity and acceptance criteria\n"
            "- backward compatibility\n"
            "- UX/API ergonomics\n"
            "- documentation and test suggestions\n"
            "Be concrete, actionable, and concise.\n"
        ),
    )

    reviewer_refactor = AgentNode(
        name="reviewer_refactor",
        llm=llm,
        node_prompt=(
            "You are an internal refactor reviewer.\n"
            "Given the code change request, write fitting review feedback focused on:\n"
            "- whether external behavior is truly unchanged\n"
            "- maintainability improvements\n"
            "- performance/regression considerations\n"
            "- readability and naming\n"
            "- test coverage needed to prove no behavior change\n"
            "Be concrete, actionable, and concise.\n"
        ),
    )

    # 4) Final synthesis: tailor output based on the chosen category.
    verdict = AgentNode(
        name="verdict",
        llm=llm,
        node_prompt=(
            "You are the reviewer coordinator.\n"
            "A decision label is available in the routing context (state['decision']).\n"
            "Write the final fitting feedback for the chosen category only.\n"
            "You may reference the relevant reviewer draft.\n\n"
            "Requirements:\n"
            "- Start with a one-line summary of the category.\n"
            "- Provide 3-7 bullet points of review feedback.\n"
            "- Include at least one concrete suggestion (tests/docs/code change).\n"
            "- Keep it succinct.\n"
        ),
    )

    # Parallel drafts first; the final node will produce the tailored feedback.
    drafts = fanout(reviewer_bug, reviewer_feature, reviewer_refactor)

    # Wiring:
    # frame -> classify -> (routes decision choice to the corresponding reviewer draft start)
    # But to satisfy "exactly one function" and a working topology, we route the
    # selected category into the matching reviewer draft, while still using fanout
    # for breadth in case of missing context.
    #
    # Here: use classify to drive which draft is used by verdict. Since the final node
    # is an AgentNode, it can read conversation messages; the routed node will update
    # messages and thus provide its draft to verdict.
    classify["bug_fix"] > reviewer_bug
    classify["new_feature"] > reviewer_feature
    classify["refactor"] > reviewer_refactor

    # Ensure drafts exist and then join at verdict.
    # We also run a parallel fanout as an insurance to guarantee reviewer drafts exist,
    # but the classifier routing above will determine which draft is most relevant in messages.
    start = frame > classify
    # Join via verdict after parallel reviewer drafts.
    # Note: classifying routes to one reviewer, but parallel fanout also ensures verdict has material.
    # We wire both by chaining: whichever path arrives first to verdict, verdict will have messages context.
    start > drafts > verdict

    return AgenticGraph(start_node=frame, end_nodes={verdict})
