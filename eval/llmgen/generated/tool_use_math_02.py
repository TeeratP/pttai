"""Generated pipeline for task 'tool_use_math' (sample 2).

TASK: Build a calculator agent that answers arithmetic word problems by calling add and multiply tools in a loop until it has the final number.
"""

from pttai import AgentNode, AgenticGraph
from pttai.tools import make_retriever_tool  # optional import (unused)

def build_graph(llm):
    # Plain python tools (no pttai.tools.add/multiply helpers exist)
    def add(a: float, b: float) -> float:
        return a + b

    def multiply(a: float, b: float) -> float:
        return a * b

    calculator = AgentNode(
        name="calculator",
        llm=llm,
        node_prompt=(
            "You are a calculator agent for arithmetic word problems.\n"
            "Solve the problem by calling tools as needed.\n"
            "Use add and multiply to perform arithmetic (and combine results across steps).\n"
            "If the problem includes multiple operations, break it into small steps.\n"
            "Keep calling tools until you are confident you have the final numeric answer.\n"
            "Then respond with only the final number (no explanation)."
        ),
        tools=[add, multiply],
        max_tool_iterations=25,
        input_field="messages",
        output_field="messages",
    )

    return AgenticGraph(start_node=calculator, end_nodes={calculator})
