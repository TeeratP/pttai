"""07 · Map-reduce — ``worker.map("field") > collector``.

``worker.map("items")`` fans the worker out over ``state["items"]`` (one parallel
LangGraph ``Send`` per item), then all replies join into the collector, which
runs ONCE. The spread field is auto-registered as a state channel.

    python examples/basics/07_map_reduce.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _llm import get_llm


def pttai_version() -> list:
    from pttai import AgentNode, AgenticGraph

    dispatch = AgentNode(llm=get_llm(), node_prompt="Kick off the summaries.")
    summarize = AgentNode(llm=get_llm(), node_prompt="Summarize this document.")
    collect = AgentNode(llm=get_llm(), node_prompt="Combine the summaries into a digest.")

    dispatch > summarize.map("docs") > collect        # fan out over state["docs"]

    graph = AgenticGraph(start_node=dispatch, end_nodes={collect})
    out = graph.invoke("Summarize the docs.", docs=["alpha memo", "beta memo", "gamma memo"])
    return out["messages"]


# --- equivalent in raw LangGraph ---
def langgraph_version() -> list:
    from langgraph.graph import StateGraph, MessagesState, START, END
    from langgraph.types import Send
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = get_llm()

    class State(MessagesState):
        docs: list

    def dispatch(state: State):
        return {"messages": [llm.invoke([SystemMessage("Kick off the summaries.")] + state["messages"])]}

    def fan_out(state: State):
        # one Send per doc; each worker invocation gets its own 1-item input
        return [Send("summarize", {"messages": [HumanMessage(d)]}) for d in state["docs"]]

    def summarize(state: State):
        return {"messages": [llm.invoke([SystemMessage("Summarize this document.")] + state["messages"])]}

    def collect(state: State):
        return {"messages": [llm.invoke([SystemMessage("Combine the summaries into a digest.")] + state["messages"])]}

    builder = StateGraph(State)
    builder.add_node("dispatch", dispatch)
    builder.add_node("summarize", summarize)
    builder.add_node("collect", collect, defer=True)
    builder.add_edge(START, "dispatch")
    builder.add_conditional_edges("dispatch", fan_out, ["summarize"])
    builder.add_edge("summarize", "collect")
    builder.add_edge("collect", END)
    graph = builder.compile()

    out = graph.invoke({"messages": [{"role": "user", "content": "Summarize the docs."}],
                        "docs": ["alpha memo", "beta memo", "gamma memo"]})
    return out["messages"]


if __name__ == "__main__":
    print("[pttai]      total messages after map+reduce:", len(pttai_version()))
    print("[langgraph]  total messages after map+reduce:", len(langgraph_version()))
