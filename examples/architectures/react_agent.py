"""ReAct agent — reason + act in a tool-calling loop.

The model interleaves THINKING with tool CALLS until it can answer: call a tool,
read the result, decide the next action, repeat. In nae this whole loop lives
inside ONE ``AgentNode(tools=[...])``.

    user ─▶ [ agent ]──tool_call──▶ (tool) ──result──┐
              ▲                                       │
              └───────────────── loop ────────────────┘
              └──▶ final answer

    python examples/architectures/react_agent.py
"""

import os
import sys

_EX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # examples/
sys.path.insert(0, os.path.dirname(_EX))  # repo root -> `import nae` works from a bare clone
sys.path.insert(0, _EX)  # -> `from _llm import get_llm`
from _llm import get_llm


def search(query: str) -> str:
    """Look up a fact."""
    return f"(stub) top result for {query!r}: 42"


def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression."""
    return str(eval(expression, {"__builtins__": {}}, {}))


def nae_version() -> str:
    from nae import AgentNode, AgenticGraph

    # One node IS the ReAct loop: call model -> run tool_calls -> feed results
    # back -> repeat, until the model returns a plain answer (capped by
    # max_tool_iterations).
    agent = AgentNode(llm=get_llm(), tools=[search, calculator],
                      node_prompt="Reason step by step and use tools to answer.")
    graph = AgenticGraph(start_node=agent, end_nodes={agent})
    return graph.invoke("What is 6 times 7?")["messages"][-1].content


# --- equivalent in raw LangGraph ---
def langgraph_version() -> str:
    from langgraph.graph import StateGraph, MessagesState, START
    from langgraph.prebuilt import ToolNode, tools_condition

    llm_with_tools = get_llm().bind_tools([search, calculator])

    def call_model(state: MessagesState):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", ToolNode([search, calculator]))
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", tools_condition)  # tool_calls? -> tools : END
    builder.add_edge("tools", "call_model")                       # loop back
    graph = builder.compile()

    result = graph.invoke({"messages": [{"role": "user", "content": "What is 6 times 7?"}]})
    return result["messages"][-1].content


if __name__ == "__main__":
    print("[nae]     ", nae_version())
    print("[langgraph] ", langgraph_version())
