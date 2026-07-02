"""Generated pipeline for task 'tool_use_math' (sample 4).

TASK: Build a calculator agent that answers arithmetic word problems by calling add and multiply tools in a loop until it has the final number.
"""

from pttai import AgentNode, AgenticGraph


def build_graph(llm):
    # Define plain python tools (pttai will wrap them as StructuredTool automatically).
    def add(a: float, b: float) -> float:
        return a + b

    def multiply(a: float, b: float) -> float:
        return a * b

    agent = AgentNode(
        name="calculator_agent",
        llm=llm,
        node_prompt=(
            "You are a calculator agent for arithmetic word problems. "
            "Solve the problem by reasoning step-by-step, calling tools when helpful. "
            "Repeat tool calls as needed until you can give the final numeric answer. "
            "Return ONLY the final answer number."
        ),
        tools=[add, multiply],
        max_tool_iterations=25,
        # Keep default behavior: read from `messages` and append final reply to `messages`.
        # The tool loop is built into AgentNode.
    )

    return AgenticGraph(start_node=agent, end_nodes={agent})
