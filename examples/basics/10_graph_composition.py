"""10 · Graph composition — an ``AgenticGraph`` used as a node.

A whole graph can be wired into a bigger graph with the same ``>`` operator, so
you can build reusable sub-workflows and compose them.

    python examples/basics/10_graph_composition.py
"""

import os
import sys

_EX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # examples/
sys.path.insert(0, os.path.dirname(_EX))  # repo root -> `import nae` works from a bare clone
sys.path.insert(0, _EX)  # -> `from _llm import get_llm`
from _llm import get_llm


def nae_version() -> str:
    from nae import AgentNode, AgenticGraph

    # --- inner sub-graph: research -> summarize ---
    research = AgentNode(name="research", llm=get_llm(), node_prompt="Research the topic.")
    condense = AgentNode(name="condense", llm=get_llm(), node_prompt="Summarize the findings.")
    research > condense
    inner = AgenticGraph(name="researcher", start_node=research, end_nodes={condense})

    # --- outer graph: use the inner graph as a single node ---
    report = AgentNode(name="report", llm=get_llm(), node_prompt="Write the final report.")
    inner > report

    outer = AgenticGraph(name="pipeline", start_node=inner, end_nodes={report})
    return outer.invoke("Tell me about tidal energy.")["messages"][-1].content


# --- equivalent in raw LangGraph ---
def langgraph_version() -> str:
    from langgraph.graph import StateGraph, MessagesState, START, END
    from langchain_core.messages import SystemMessage

    llm = get_llm()

    def step(prompt):
        def node(state: MessagesState):
            return {"messages": [llm.invoke([SystemMessage(prompt)] + state["messages"])]}
        return node

    # inner sub-graph, compiled
    inner_b = StateGraph(MessagesState)
    inner_b.add_node("research", step("Research the topic."))
    inner_b.add_node("condense", step("Summarize the findings."))
    inner_b.add_edge(START, "research")
    inner_b.add_edge("research", "condense")
    inner_b.add_edge("condense", END)
    inner = inner_b.compile()

    # outer graph adds the compiled sub-graph as a node
    outer_b = StateGraph(MessagesState)
    outer_b.add_node("researcher", inner)
    outer_b.add_node("report", step("Write the final report."))
    outer_b.add_edge(START, "researcher")
    outer_b.add_edge("researcher", "report")
    outer_b.add_edge("report", END)
    outer = outer_b.compile()

    return outer.invoke({"messages": [{"role": "user", "content": "Tell me about tidal energy."}]})["messages"][-1].content


if __name__ == "__main__":
    print("[nae]     ", nae_version())
    print("[langgraph] ", langgraph_version())
