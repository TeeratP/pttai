"""Generated pipeline for task 'tool_use_math' (sample 1).

TASK: Build a calculator agent that answers arithmetic word problems by calling add and multiply tools in a loop until it has the final number.
"""

from pttai import AgentNode, AgenticGraph


def build_graph(llm):
    def add(a: float, b: float) -> float:
        return a + b

    def multiply(a: float, b: float) -> float:
        return a * b

    # A single AgentNode encapsulates the model + tool-call loop.
    agent = AgentNode(
        name="calculator_agent",
        llm=llm,
        node_prompt=(
            "You are a calculator assistant for arithmetic word problems.\n"
            "Solve the problem by calling tools as needed (use add and multiply).\n"
            "Keep solving until you can produce the final numeric answer.\n"
            "Return ONLY the final number."
        ),
        tools=[add, multiply],
        max_tool_iterations=25,
        input_field="messages",
        output_field="messages",
    )

    return AgenticGraph(start_node=agent, end_nodes={agent})
