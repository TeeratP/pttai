"""Generated pipeline for task 'reflect_revise' (sample 3).

TASK: Draft an answer, then critique the draft, then revise the answer using the critique before returning it.
"""

from pttai import AgentNode, DecisionNode, ConditionNode, AgenticGraph, fanout


def build_graph(llm):
    # 1) Draft the answer
    draft = AgentNode(
        name="draft",
        llm=llm,
        node_prompt=(
            "You are a helpful assistant. Write a draft answer to the user question.\n"
            "Requirements:\n"
            "- Be clear and directly address the question.\n"
            "- If you are uncertain, say so and state what you'd need to verify.\n"
            "- Keep it reasonably concise."
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # 2) Critique the draft (LLM-based judgment)
    critique = AgentNode(
        name="critique",
        llm=llm,
        node_prompt=(
            "You are a meticulous reviewer. Critique the assistant's draft answer.\n"
            "Provide:\n"
            "- The biggest issues (accuracy, completeness, clarity).\n"
            "- Concrete suggestions to improve.\n"
            "Output as a short bullet list."
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # 3) Decide whether to revise
    decide_revise = DecisionNode(
        name="decide_revise",
        llm=llm,
        node_prompt=(
            "Based on the critique, decide whether the draft answer needs a revision.\n"
            "Choose one option only."
        ),
        choices=["revise", "final"],
        reads=["messages"],
        # DecisionNode writes its choice to state['decision'] automatically.
    )

    # 4) Revise using critique
    revise = AgentNode(
        name="revise",
        llm=llm,
        node_prompt=(
            "Revise the previous draft answer using the critique.\n"
            "Requirements:\n"
            "- Fix the identified issues.\n"
            "- Preserve useful content that is already correct.\n"
            "- Output the revised final answer."
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # 5) Finalize without revision (just ensure we end with a final answer)
    finalize = AgentNode(
        name="finalize",
        llm=llm,
        node_prompt=(
            "Prepare the final answer for the user.\n"
            "Use the latest available draft/critique context, but do not do a full rewrite unless necessary.\n"
            "Output only the final answer."
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # Wiring:
    # Draft and critique run sequentially, then a decision routes to revise vs finalize.
    draft > critique > decide_revise
    decide_revise["revise"] > revise
    decide_revise["final"] > finalize

    return AgenticGraph(start_node=draft, end_nodes={revise, finalize})
