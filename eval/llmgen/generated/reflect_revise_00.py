"""Generated pipeline for task 'reflect_revise' (sample 0).

TASK: Draft an answer, then critique the draft, then revise the answer using the critique before returning it.
"""

from pttai import AgentNode, AgenticGraph, fanout


def build_graph(llm):
    draft = AgentNode(
        name="draft",
        llm=llm,
        node_prompt=(
            "You are an excellent writer.\n"
            "Task: Draft a high-quality answer to the user's prompt.\n"
            "Requirements:\n"
            "- Be clear and specific.\n"
            "- Include any needed steps or rationale briefly.\n"
            "- Do NOT include any critique section; only the draft answer."
        ),
        input_field="messages",
        writes="messages",
    )

    critique = AgentNode(
        name="critique",
        llm=llm,
        node_prompt=(
            "You are a meticulous reviewer.\n"
            "Critique the provided draft.\n"
            "Requirements:\n"
            "- Point out concrete issues (missing parts, inaccuracies, unclear wording, structure problems).\n"
            "- Provide actionable improvement suggestions.\n"
            "- Output only the critique."
        ),
        reads=["messages"],
        writes="messages",
    )

    revise = AgentNode(
        name="revise",
        llm=llm,
        node_prompt=(
            "You are the reviser.\n"
            "Given the original user prompt and the draft+critique, produce the final revised answer.\n"
            "Requirements:\n"
            "- Incorporate all key critique fixes.\n"
            "- Improve clarity, correctness, and completeness.\n"
            "- Output ONLY the revised final answer."
        ),
        reads=["messages"],
        writes="messages",
    )

    return AgenticGraph(
        start_node=draft,
        end_nodes={revise},
        # sequential: draft -> critique -> revise
        # (the wiring is expressed by the '>' operator below)
    )
