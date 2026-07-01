"""06 · Parallel fan-out + join — ``start > fanout(a, b) > join``.

``fanout(a, b)`` runs the branches CONCURRENTLY; the join node is deferred and
runs ONCE after every branch finishes. (``start > [a, b] > join`` is the same.)

    python examples/basics/06_parallel_fanout.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _llm import get_llm


def pttai_version() -> list:
    from pttai import AgentNode, AgenticGraph, fanout

    start = AgentNode(llm=get_llm(), node_prompt="Restate the task.")
    pros = AgentNode(llm=get_llm(), node_prompt="List the pros.")
    cons = AgentNode(llm=get_llm(), node_prompt="List the cons.")
    combine = AgentNode(llm=get_llm(), node_prompt="Weigh the pros and cons.")

    start > fanout(pros, cons) > combine

    graph = AgenticGraph(start_node=start, end_nodes={combine})
    return graph.invoke("Should we adopt a 4-day work week?")["log"]


# --- equivalent in raw LangGraph ---
def langgraph_version() -> list:
    import operator
    from typing import Annotated
    from langgraph.graph import StateGraph, MessagesState, START, END
    from langchain_core.messages import SystemMessage

    llm = get_llm()

    class State(MessagesState):
        log: Annotated[list, operator.add]

    def step(name, prompt):
        def node(state: State):
            msg = llm.invoke([SystemMessage(prompt)] + state["messages"])
            return {"messages": [msg], "log": [f"{name}:{msg.content}"]}
        return node

    builder = StateGraph(State)
    builder.add_node("start", step("start", "Restate the task."))
    builder.add_node("pros", step("pros", "List the pros."))
    builder.add_node("cons", step("cons", "List the cons."))
    builder.add_node("combine", step("combine", "Weigh the pros and cons."), defer=True)
    builder.add_edge(START, "start")
    builder.add_edge("start", "pros")     # fan out
    builder.add_edge("start", "cons")
    builder.add_edge("pros", "combine")   # fan in (deferred join)
    builder.add_edge("cons", "combine")
    builder.add_edge("combine", END)
    graph = builder.compile()

    return graph.invoke({"messages": [{"role": "user", "content": "Should we adopt a 4-day work week?"}], "log": []})["log"]


if __name__ == "__main__":
    print("[pttai]      ran nodes:", [line.split(":", 1)[0] for line in pttai_version()])
    print("[langgraph]  ran nodes:", [line.split(":", 1)[0] for line in langgraph_version()])
