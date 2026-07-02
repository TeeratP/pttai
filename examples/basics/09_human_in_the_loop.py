"""09 · Human-in-the-loop — ``HumanNode`` + interrupt/resume.

A ``HumanNode`` pauses the run via LangGraph's ``interrupt()``. With a
checkpointer + a ``thread_id``, the first ``invoke`` stops at the interrupt; a
second ``invoke(Command(resume=...))`` on the same thread feeds the human reply
back in and the run continues.

    python examples/basics/09_human_in_the_loop.py
"""

import os
import sys

_EX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # examples/
sys.path.insert(0, os.path.dirname(_EX))  # repo root -> `import nae` works from a bare clone
sys.path.insert(0, _EX)  # -> `from _llm import get_llm`
from _llm import get_llm


def nae_version() -> str:
    from nae import AgentNode, HumanNode, AgenticGraph
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    draft = AgentNode(llm=get_llm(), node_prompt="Draft a reply.")
    review = HumanNode(node_prompt="Approve or edit this draft:", n=1)
    finalize = AgentNode(llm=get_llm(), node_prompt="Incorporate the human feedback.")

    draft > review > finalize

    graph = AgenticGraph(start_node=draft, end_nodes={finalize},
                         checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "demo-1"}}

    first = graph.invoke("Reply to the customer.", config=config)
    assert "__interrupt__" in first, "expected the graph to pause at the HumanNode"

    final = graph.invoke(Command(resume="Looks good, ship it."), config=config)
    return final["messages"][-1].content


# --- equivalent in raw LangGraph ---
def langgraph_version() -> str:
    from langgraph.graph import StateGraph, MessagesState, START, END
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import interrupt, Command
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = get_llm()

    def draft(state: MessagesState):
        return {"messages": [llm.invoke([SystemMessage("Draft a reply.")] + state["messages"])]}

    def review(state: MessagesState):
        reply = interrupt({"Approve or edit this draft:": state["messages"][-1].content})
        return {"messages": [HumanMessage(reply)]}

    def finalize(state: MessagesState):
        return {"messages": [llm.invoke([SystemMessage("Incorporate the human feedback.")] + state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("draft", draft)
    builder.add_node("review", review)
    builder.add_node("finalize", finalize)
    builder.add_edge(START, "draft")
    builder.add_edge("draft", "review")
    builder.add_edge("review", "finalize")
    builder.add_edge("finalize", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "demo-2"}}

    first = graph.invoke({"messages": [{"role": "user", "content": "Reply to the customer."}]}, config=config)
    assert "__interrupt__" in first
    final = graph.invoke(Command(resume="Looks good, ship it."), config=config)
    return final["messages"][-1].content


if __name__ == "__main__":
    print("[nae]     ", nae_version())
    print("[langgraph] ", langgraph_version())
