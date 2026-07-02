"""Generated pipeline for task 'tool_use_math' (sample 0).

TASK: Build a calculator agent that answers arithmetic word problems by calling add and multiply tools in a loop until it has the final number.
"""

from pttai import AgentNode, AgenticGraph


# Plain functions wrapped by pttai as StructuredTools via AgentNode(tools=[...])
def add(a: float, b: float) -> float:
    return a + b


def multiply(a: float, b: float) -> float:
    return a * b


def build_graph(llm):
    agent = AgentNode(
        name="calculator_agent",
        llm=llm,
        tools=[add, multiply],
        node_prompt=(
            "You are a calculator agent. Solve the user's arithmetic word problem by "
            "calling the provided tools (add, multiply) as needed. "
            "Keep calling tools in a loop until you can determine the final numeric answer. "
            "If the problem is ambiguous, make the most reasonable assumption and proceed. "
            "Return ONLY the final number (no units, no explanation)."
        ),
        max_tool_iterations=25,
    )

    return AgenticGraph(start_node=agent, end_nodes={agent})
