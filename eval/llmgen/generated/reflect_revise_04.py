"""Generated pipeline for task 'reflect_revise' (sample 4).

TASK: Draft an answer, then critique the draft, then revise the answer using the critique before returning it.
"""

from pttai import AgentNode, AgenticGraph, fanout


def build_graph(llm):
    # 1) Draft an answer from the user's prompt.
    draft = AgentNode(
        name="draft",
        llm=llm,
        node_prompt=(
            "You are a careful assistant. Draft a high-quality answer to the user's request. "
            "Include key reasoning steps when helpful, be concise but complete, and follow any constraints."
        ),
    )

    # 2) Critique the draft from multiple angles in parallel.
    critique_quality = AgentNode(
        name="critique_quality",
        llm=llm,
        node_prompt=(
            "Critique the provided draft for correctness, completeness, and clarity. "
            "List the top issues as bullet points, and suggest concrete improvements."
        ),
        reads=["messages"],
        input_field="messages",
    )

    critique_assumptions = AgentNode(
        name="critique_assumptions",
        llm=llm,
        node_prompt=(
            "Critique the provided draft for hidden assumptions, missing context, and unstated definitions. "
            "Point out anything that should be clarified or verified."
        ),
        reads=["messages"],
        input_field="messages",
    )

    critique_safety = AgentNode(
        name="critique_safety",
        llm=llm,
        node_prompt=(
            "Critique the provided draft for safety and policy issues: any harmful instructions, "
            "privacy leaks, or rule violations. If none, state that clearly."
        ),
        reads=["messages"],
        input_field="messages",
    )

    # 3) Revise the draft using the critiques.
    revise = AgentNode(
        name="revise",
        llm=llm,
        node_prompt=(
            "You are the editor. You will receive (implicitly, via conversation history) a draft answer "
            "and multiple critiques. Revise the answer to address the critiques. "
            "Return ONLY the revised final answer (no meta-commentary)."
        ),
    )

    # Wiring: draft -> parallel critiques -> revise
    draft > fanout(critique_quality, critique_assumptions, critique_safety) > revise

    return AgenticGraph(start_node=draft, end_nodes={revise})
