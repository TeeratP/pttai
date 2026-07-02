"""nae vs. raw LangGraph — the same tool-using agent, side by side.

Both implementations below build the *identical* agent: an LLM that can call two
tools (`add`, `multiply`) in a loop until it has a final answer, then return it.
Ask it "What is 21 + 21, then times 3?" and both print **126**.

The only difference is how much graph plumbing you write:

  * `nae_version()`    — one `AgentNode` with a built-in tool-call loop.
  * `langgraph_version()` — the honest, idiomatic raw-LangGraph equivalent:
    `StateGraph(MessagesState)` + a model node + a `ToolNode` +
    `tools_condition` + a loop-back edge, then `compile()`.

Run it::

    export OPENAI_API_KEY=sk-...
    python examples/vs_langgraph.py            # runs both
    python examples/vs_langgraph.py nae      # runs only the nae version
    python examples/vs_langgraph.py langgraph  # runs only the raw-LangGraph version

(Needs OPENAI_API_KEY — both versions make real model calls.)

Honest line counts of the graph-building code (the part that differs; shared
tool defs, the LLM instantiation, the invoke/print, blanks and comments are not
counted — see the comment block at the bottom of this file):

    nae:         3 lines
    raw LangGraph: 10 lines   (plus one extra import line)
"""

import sys

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

QUESTION = "What is 21 + 21, then times 3?"


# --- the shared tools (identical for both implementations) ------------------

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


# --- the nae way ----------------------------------------------------------

def nae_version() -> str:
    """A tool-using agent in nae. The whole model<->tools loop is one node."""
    from nae import AgentNode, AgenticGraph

    llm = ChatOpenAI(model="gpt-5.4-nano")

    # --- graph code (3 lines) ---
    agent = AgentNode(name="agent", llm=llm, tools=[add, multiply])
    graph = AgenticGraph(start_node=agent, end_nodes={agent})  # schema-free

    return graph.invoke(message=QUESTION)["messages"][-1].content


# --- the raw LangGraph way --------------------------------------------------

def langgraph_version() -> str:
    """The same agent in raw LangGraph: you wire the model/tools sub-graph yourself."""
    from langgraph.graph import StateGraph, MessagesState, START, END
    from langgraph.prebuilt import ToolNode, tools_condition

    llm = ChatOpenAI(model="gpt-5.4-nano")

    # --- graph code (10 lines) ---
    llm_with_tools = llm.bind_tools([add, multiply])

    def call_model(state: MessagesState):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", ToolNode([add, multiply]))
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", tools_condition)  # tools? -> "tools" : END
    builder.add_edge("tools", "call_model")                       # loop back to the model
    graph = builder.compile()

    result = graph.invoke({"messages": [{"role": "user", "content": QUESTION}]})
    return result["messages"][-1].content


# --- runner -----------------------------------------------------------------

def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("nae", "both"):
        print(f"[nae]      {QUESTION} -> {nae_version()}")
    if which in ("langgraph", "both"):
        print(f"[langgraph]  {QUESTION} -> {langgraph_version()}")


if __name__ == "__main__":
    main()


# -- Honest line count (graph-building code only) ---------------------------
#
# nae_version() graph code — 3 lines:
#     agent = AgentNode(name="agent", llm=llm, tools=[add, multiply])
#     graph = AgenticGraph(start_node=agent, end_nodes={agent})
#
# langgraph_version() graph code — 10 lines:
#     llm_with_tools = llm.bind_tools([add, multiply])
#     def call_model(state): ...
#         return {"messages": [llm_with_tools.invoke(state["messages"])]}
#     builder = StateGraph(MessagesState)
#     builder.add_node("call_model", call_model)
#     builder.add_node("tools", ToolNode([add, multiply]))
#     builder.add_edge(START, "call_model")
#     builder.add_conditional_edges("call_model", tools_condition)
#     builder.add_edge("tools", "call_model")
#     graph = builder.compile()
#
# Same agent, same answer (126). nae folds the model node, the ToolNode, the
# conditional edge and the loop-back edge into one AgentNode with an internal
# tool-call loop, and infers the state schema (MessagesState) for you.
