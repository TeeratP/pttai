"""Routing — classify the input, then dispatch to a specialist handler.

A ``DecisionNode`` picks one of several categories (constrained structured
output, so it MUST return a valid label) and routes there. This one first GATHERS
context with a tool, then routes — ``DecisionNode(tools=...)``: tools run in a
first phase, the gathered context feeds the routing call.

    input ─▶ [ triage ]──(lookup tool)──┬─billing──▶ [ billing ]  ─▶ out
                                        ├─technical▶ [ technical ]─▶ out
                                        └─general──▶ [ general ]  ─▶ out

    python examples/architectures/routing.py
"""

import os
import sys

_EX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # examples/
sys.path.insert(0, os.path.dirname(_EX))  # repo root -> `import nae` works from a bare clone
sys.path.insert(0, _EX)  # -> `from _llm import get_llm`
from _llm import get_llm


def lookup_account(user: str) -> str:
    """Look up a customer's plan and status."""
    return f"{user}: Pro plan, payment overdue"


def nae_version() -> str:
    from nae import AgentNode, DecisionNode, AgenticGraph

    # Gather-then-route: the tool runs first, then the model picks a category.
    triage = DecisionNode(
        llm=get_llm(), tools=[lookup_account],
        node_prompt="Look up the account, then classify the ticket.",
        choices=["billing", "technical", "general"],
    )
    billing = AgentNode(llm=get_llm(), node_prompt="Handle the billing question.")
    technical = AgentNode(llm=get_llm(), node_prompt="Handle the technical issue.")
    general = AgentNode(llm=get_llm(), node_prompt="Handle the general query.")

    triage["billing"] > billing
    triage["technical"] > technical
    triage["general"] > general

    graph = AgenticGraph(start_node=triage, end_nodes={billing, technical, general})
    out = graph.invoke("I was charged twice this month!")
    return f"routed to {out['decision_triage']!r} -> {out['messages'][-1].content}"


# --- equivalent in raw LangGraph ---
# The two phases (tool loop, THEN a structured-output route) can't share one
# node, so raw LangGraph needs a `gather` node + a `route` node wired by hand.
def langgraph_version() -> str:
    from typing import Literal
    from langgraph.graph import StateGraph, MessagesState, START, END
    from langchain_core.messages import SystemMessage
    from langgraph.prebuilt import ToolNode, tools_condition
    from pydantic import BaseModel

    class Route(BaseModel):
        choice: Literal["billing", "technical", "general"]

    llm = get_llm()
    gatherer = llm.bind_tools([lookup_account])
    router = llm.with_structured_output(Route)

    def gather(state: MessagesState):  # phase 1: may call the lookup tool
        return {"messages": [gatherer.invoke([SystemMessage("Look up the account.")] + state["messages"])]}

    def route(state: MessagesState):   # phase 2: classify on gathered context
        return router.invoke([SystemMessage("Classify the ticket.")] + state["messages"]).choice

    def handler(prompt):
        def node(state: MessagesState):
            return {"messages": [llm.invoke([SystemMessage(prompt)] + state["messages"])]}
        return node

    builder = StateGraph(MessagesState)
    builder.add_node("gather", gather)
    builder.add_node("tools", ToolNode([lookup_account]))
    builder.add_node("billing", handler("Handle the billing question."))
    builder.add_node("technical", handler("Handle the technical issue."))
    builder.add_node("general", handler("Handle the general query."))
    builder.add_edge(START, "gather")
    # tool_calls? -> run tools and loop; otherwise -> route on gathered context
    builder.add_conditional_edges("gather", tools_condition, {"tools": "tools", END: "__route__"})
    builder.add_node("__route__", lambda s: {})  # pass-through so route() decides the branch
    builder.add_edge("tools", "gather")
    builder.add_conditional_edges("__route__", route,
                                  {"billing": "billing", "technical": "technical", "general": "general"})
    for h in ("billing", "technical", "general"):
        builder.add_edge(h, END)
    graph = builder.compile()

    out = graph.invoke({"messages": [{"role": "user", "content": "I was charged twice this month!"}]})
    return f"routed -> {out['messages'][-1].content}"


if __name__ == "__main__":
    print("[nae]     ", nae_version())
    print("[langgraph] ", langgraph_version())
