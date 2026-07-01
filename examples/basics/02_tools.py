"""02 · Tools — an agent that calls a tool in a loop.

Pass ``tools=[...]`` and the ``AgentNode`` runs the whole model<->tool loop
internally: call the model, execute any tool calls, feed results back, repeat
until the model gives a final answer.

    python examples/basics/02_tools.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _llm import get_llm


def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"{city}: sunny, 72F"


def pttai_version() -> str:
    from pttai import AgentNode, AgenticGraph

    agent = AgentNode(llm=get_llm(), tools=[get_weather],
                      node_prompt="Answer questions using the weather tool.")
    graph = AgenticGraph(start_node=agent, end_nodes={agent})
    return graph.invoke("What's the weather in Paris?")["messages"][-1].content


# --- equivalent in raw LangGraph ---
def langgraph_version() -> str:
    from langgraph.graph import StateGraph, MessagesState, START
    from langgraph.prebuilt import ToolNode, tools_condition

    llm_with_tools = get_llm().bind_tools([get_weather])

    def call_model(state: MessagesState):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", ToolNode([get_weather]))
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", tools_condition)  # tools? -> tools : END
    builder.add_edge("tools", "call_model")                       # loop back
    graph = builder.compile()

    result = graph.invoke({"messages": [{"role": "user", "content": "What's the weather in Paris?"}]})
    return result["messages"][-1].content


if __name__ == "__main__":
    print("[pttai]     ", pttai_version())
    print("[langgraph] ", langgraph_version())
