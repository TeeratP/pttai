"""12 · Introspection — ``graph.validate()`` + ``graph.summary()``.

pttai statically checks the graph's dataflow at build time (every key a node
reads is produced upstream; every key it writes is declared) and can print a
Keras-``model.summary()``-style table of each node's reads / writes / available
keys. ``validate()`` returns a report; the build itself raises on hard errors.

    python examples/basics/12_validation_summary.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _llm import get_llm


def pttai_version() -> bool:
    from pttai import AgentNode, AgenticGraph

    outline = AgentNode(llm=get_llm(), node_prompt="Outline the answer.")
    write = AgentNode(llm=get_llm(), node_prompt="Write it up.")
    outline > write

    graph = AgenticGraph(start_node=outline, end_nodes={write})

    report = graph.validate()
    print(f"validate() -> ok={report.ok}, "
          f"{len(report.errors)} error(s), {len(report.warnings)} warning(s)")
    print()
    graph.summary()               # the Keras-style table
    return report.ok


# --- equivalent in raw LangGraph ---
def langgraph_version() -> None:
    # Raw LangGraph has no built-in STATIC state-dataflow validation or a
    # summary() table. The closest analog is the structural view from
    # get_graph() after compile(); state read/write mistakes only surface at
    # runtime. pttai adds the static check + summary on top.
    from langgraph.graph import StateGraph, MessagesState, START, END
    from langchain_core.messages import SystemMessage

    llm = get_llm()

    def step(prompt):
        def node(state: MessagesState):
            return {"messages": [llm.invoke([SystemMessage(prompt)] + state["messages"])]}
        return node

    builder = StateGraph(MessagesState)
    builder.add_node("outline", step("Outline the answer."))
    builder.add_node("write", step("Write it up."))
    builder.add_edge(START, "outline")
    builder.add_edge("outline", "write")
    builder.add_edge("write", END)
    graph = builder.compile()

    nodes = list(graph.get_graph().nodes)
    print("raw LangGraph get_graph().nodes:", nodes)


if __name__ == "__main__":
    print("=== pttai ===")
    pttai_version()
    print("\n=== raw LangGraph ===")
    langgraph_version()
