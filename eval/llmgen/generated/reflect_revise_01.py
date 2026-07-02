"""Generated pipeline for task 'reflect_revise' (sample 1).

TASK: Draft an answer, then critique the draft, then revise the answer using the critique before returning it.
"""

from pttai import AgentNode, ConditionNode, DecisionNode, HumanNode, AgenticGraph, fanout


def build_graph(llm):
    # --- Draft ---
    # 1) Draft an answer
    # 2) Optionally ask a human for feedback (if model decides it's needed)
    draft = AgentNode(
        name="draft",
        llm=llm,
        node_prompt=(
            "You are a careful assistant. Draft a high-quality answer to the user's question.\n"
            "Be accurate, concise, and explicitly address the request.\n\n"
            "User question:\n{question}"
        ),
        reads=["question"],
        writes=["messages"],
    )

    # --- Critique ---
    # 3) Critique the draft for correctness and completeness
    critique = AgentNode(
        name="critique",
        llm=llm,
        node_prompt=(
            "You are a strict reviewer. Critique the assistant's draft for:\n"
            "- factual/citation issues (if any)\n"
            "- missing requirements\n"
            "- unclear or weak reasoning\n"
            "- any contradictions\n\n"
            "Return a critique and list the specific fixes needed."
        ),
        writes=["messages"],
    )

    # --- Decide revise or accept (LLM branching) ---
    need_revision_decision = DecisionNode(
        name="need_revision_decision",
        llm=llm,
        node_prompt=(
            "Decide whether the draft answer should be revised based on the critique.\n"
            "Return 'revise' if there are substantive issues or missing requirements; otherwise return 'accept'."
        ),
        choices=["revise", "accept"],
        input_field="messages",
    )

    # --- Optional human-in-the-loop ---
    # If the model wants revision, allow a human to provide feedback before final revision.
    human_feedback = HumanNode(
        name="human_feedback",
        node_prompt=(
            "Review the current draft and critique. If you want changes, provide specific feedback "
            "that the assistant should incorporate. If no changes are needed, say 'no changes'."
        ),
        n=2,  # show the last two messages (typically draft + critique)
        into="messages",
    )

    # --- Revise ---
    revise = AgentNode(
        name="revise",
        llm=llm,
        node_prompt=(
            "Revise the answer using the critique and any human feedback.\n"
            "Produce the final improved answer that satisfies all requirements.\n\n"
            "Make sure the final answer is coherent and directly answers the user's question.\n"
            "User question:\n{question}"
        ),
        reads=["question"],
        writes=["messages"],
    )

    # --- Accept path ---
    accept = AgentNode(
        name="accept",
        llm=llm,
        node_prompt=(
            "The draft is acceptable. Provide the final answer to the user's question based on the draft."
            "\n\nUser question:\n{question}"
        ),
        reads=["question"],
        writes=["messages"],
    )

    # Optional safety: If question is empty, short-circuit to a clarifying question.
    # (Deterministic predicate based on availability of a seeded scalar read.)
    has_question = ConditionNode(
        name="has_question",
        condition=lambda state: "yes" if (state.get("question") and str(state.get("question")).strip()) else "no",
        choices=["yes", "no"],
        reads=["question"],
    )

    clarify = AgentNode(
        name="clarify",
        llm=llm,
        node_prompt=(
            "Ask a single, clear clarifying question to the user to proceed."
            "\n\nCurrent question (may be empty):\n{question}"
        ),
        reads=["question"],
        writes=["messages"],
    )

    # Wiring:
    # Ensure we have a question first; then draft+critique in sequence, route with DecisionNode.
    has_question["no"] > clarify

    # Main pipeline:
    # draft -> critique -> need_revision_decision
    draft > critique > need_revision_decision
    need_revision_decision["accept"] > accept
    need_revision_decision["revise"] > human_feedback > revise

    return AgenticGraph(
        start_node=has_question,
        end_nodes={clarify, accept, revise},
    )
