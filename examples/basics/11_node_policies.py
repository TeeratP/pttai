"""11 · Node policies — ``cache_ttl`` / ``retry`` / ``reasoning_effort``.

Per-node knobs, passed straight through to LangGraph (or the model):
  * ``cache_ttl=`` — cache this node's result for N seconds (LangGraph CachePolicy);
    ``AgenticGraph`` auto-provides an in-memory cache when any node sets it.
  * ``retry=True`` — retry this node on exception (LangGraph RetryPolicy).
  * ``reasoning_effort=`` — per-call kwarg for reasoning models (gpt-5.x).

    python examples/basics/11_node_policies.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _llm import get_llm


def pttai_version() -> str:
    from pttai import AgentNode, AgenticGraph

    agent = AgentNode(
        llm=get_llm(),
        node_prompt="Answer concisely.",
        cache_ttl=60,               # cache the result for 60s
        retry=True,                 # retry on transient failure
        reasoning_effort="low",     # per-call reasoning effort (gpt-5.x)
    )
    graph = AgenticGraph(start_node=agent, end_nodes={agent})   # auto in-memory cache
    return graph.invoke("What is 2 + 2?")["messages"][-1].content


# --- equivalent in raw LangGraph ---
def langgraph_version() -> str:
    from langgraph.graph import StateGraph, MessagesState, START, END
    from langgraph.types import CachePolicy, RetryPolicy
    from langgraph.cache.memory import InMemoryCache
    from langchain_core.messages import SystemMessage

    # reasoning_effort is a model kwarg in raw LangGraph — bind it on the model.
    llm = get_llm()

    def call_model(state: MessagesState):
        return {"messages": [llm.invoke([SystemMessage("Answer concisely.")] + state["messages"],
                                        reasoning_effort="low")]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", call_model,
                     cache_policy=CachePolicy(ttl=60), retry_policy=RetryPolicy())
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    graph = builder.compile(cache=InMemoryCache())

    return graph.invoke({"messages": [{"role": "user", "content": "What is 2 + 2?"}]})["messages"][-1].content


if __name__ == "__main__":
    print("[pttai]     ", pttai_version())
    print("[langgraph] ", langgraph_version())
