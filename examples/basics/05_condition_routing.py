"""05 · Condition routing — a plain Python predicate, no LLM.

A ``ConditionNode`` routes on ``condition(state) -> label`` — free, deterministic,
no model call. Use it whenever the branch is decidable in code.

    python examples/basics/05_condition_routing.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _llm import get_llm


def _is_short(state) -> str:
    return "short" if len(state["messages"][-1].content) < 20 else "long"


def pttai_version() -> str:
    from pttai import AgentNode, ConditionNode, AgenticGraph

    route = ConditionNode(condition=_is_short, choices=["short", "long"])
    brief = AgentNode(llm=get_llm(), node_prompt="Give a one-line answer.")
    detailed = AgentNode(llm=get_llm(), node_prompt="Give a thorough answer.")

    route["short"] > brief
    route["long"] > detailed

    graph = AgenticGraph(start_node=route, end_nodes={brief, detailed})
    out = graph.invoke("Hi")
    return f"routed to {out['decision']!r} -> {out['messages'][-1].content}"


# --- equivalent in raw LangGraph ---
def langgraph_version() -> str:
    from langgraph.graph import StateGraph, MessagesState, START, END
    from langchain_core.messages import SystemMessage

    llm = get_llm()

    def brief(state: MessagesState):
        return {"messages": [llm.invoke([SystemMessage("Give a one-line answer.")] + state["messages"])]}

    def detailed(state: MessagesState):
        return {"messages": [llm.invoke([SystemMessage("Give a thorough answer.")] + state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("brief", brief)
    builder.add_node("detailed", detailed)
    # A conditional edge straight off START is the raw-LangGraph way to branch
    # on a predicate without a node.
    builder.add_conditional_edges(START, _is_short, {"short": "brief", "long": "detailed"})
    builder.add_edge("brief", END)
    builder.add_edge("detailed", END)
    graph = builder.compile()

    out = graph.invoke({"messages": [{"role": "user", "content": "Hi"}]})
    label = _is_short(out)
    return f"routed to {label!r} -> {out['messages'][-1].content}"


if __name__ == "__main__":
    print("[pttai]     ", pttai_version())
    print("[langgraph] ", langgraph_version())
