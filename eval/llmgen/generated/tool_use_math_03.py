"""Generated pipeline for task 'tool_use_math' (sample 3).

TASK: Build a calculator agent that answers arithmetic word problems by calling add and multiply tools in a loop until it has the final number.
"""

from pttai import AgentNode, AgenticGraph
from pttai.tools import make_retriever_tool  # optional; not used


def build_graph(llm):
    def add(a: float, b: float) -> float:
        return a + b

    def multiply(a: float, b: float) -> float:
        return a * b

    agent = AgentNode(
        name="calculator_agent",
        llm=llm,
        node_prompt=(
            "You are a calculator agent for arithmetic word problems.\n"
            "Goal: compute the final numeric answer.\n"
            "Use the provided tools (add, multiply) as needed.\n"
            "If you need to compute intermediate values, do so with tools.\n"
            "Return ONLY the final number as the answer.\n"
            "Word-problem requirements:\n"
            "- Handle simple arithmetic implied by text.\n"
            "- Be explicit and correct about order when implied (use tools rather than mental shortcuts).\n"
            "- Stop once the final number is determined."
        ),
        tools=[add, multiply],
        writes=["answer"],
        max_tool_iterations=25,
    )

    return AgenticGraph(start_node=agent, end_nodes={agent})
