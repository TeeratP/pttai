"""Evaluator–optimizer — generate, evaluate, then refine in a loop.

One agent produces a draft, another critiques it, and a gate decides: loop back
to refine, or accept. The gate here caps the loop at 2 rounds (with a real model
you'd also accept early once the critique says "good"), so it always terminates.

    ┌──────────────── refine ────────────────┐
    ▼                                         │
    [ generate ] ─▶ [ evaluate ] ─▶ <gate> ──┘
                                      └──accept──▶ [ finalize ] ─▶ out

    python examples/architectures/evaluator_optimizer.py
"""

import os
import sys

_EX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # examples/
sys.path.insert(0, os.path.dirname(_EX))  # repo root -> `import nae` works from a bare clone
sys.path.insert(0, _EX)  # -> `from _llm import get_llm`
from _llm import get_llm

MAX_ROUNDS = 2


def _gate(state) -> str:
    # Count how many times `generate` has run (each call adds a "generate:..."
    # line to the reduced `log` channel). Accept once we hit the cap — this is
    # the hard guarantee that the loop terminates, even with the offline fake.
    rounds = sum(1 for line in state["log"] if line.startswith("generate"))
    return "accept" if rounds >= MAX_ROUNDS else "refine"


def nae_version() -> str:
    from nae import AgentNode, ConditionNode, AgenticGraph

    generate = AgentNode(name="generate", llm=get_llm(), node_prompt="Write or improve the draft.")
    evaluate = AgentNode(name="evaluate", llm=get_llm(), node_prompt="Critique the draft; list concrete fixes.")
    gate = ConditionNode(name="gate", condition=_gate, choices=["refine", "accept"])
    finalize = AgentNode(name="finalize", llm=get_llm(), node_prompt="Return the polished final draft.")

    generate > evaluate > gate
    gate["refine"] > generate      # loop back to improve
    gate["accept"] > finalize

    graph = AgenticGraph(start_node=generate, end_nodes={finalize})
    return graph.invoke("Write a tagline for a coffee brand.")["messages"][-1].content


# --- equivalent in raw LangGraph ---
def langgraph_version() -> str:
    from typing import Annotated
    import operator
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
    from langchain_core.messages import SystemMessage, AnyMessage
    from typing_extensions import TypedDict

    llm = get_llm()

    class State(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]
        rounds: Annotated[int, operator.add]   # incremented each generate

    def generate(state: State):
        msg = llm.invoke([SystemMessage("Write or improve the draft.")] + state["messages"])
        return {"messages": [msg], "rounds": 1}

    def evaluate(state: State):
        return {"messages": [llm.invoke([SystemMessage("Critique the draft; list concrete fixes.")] + state["messages"])]}

    def finalize(state: State):
        return {"messages": [llm.invoke([SystemMessage("Return the polished final draft.")] + state["messages"])]}

    def gate(state: State):
        return "accept" if state["rounds"] >= MAX_ROUNDS else "refine"

    builder = StateGraph(State)
    builder.add_node("generate", generate)
    builder.add_node("evaluate", evaluate)
    builder.add_node("finalize", finalize)
    builder.add_edge(START, "generate")
    builder.add_edge("generate", "evaluate")
    builder.add_conditional_edges("evaluate", gate, {"refine": "generate", "accept": "finalize"})
    builder.add_edge("finalize", END)
    graph = builder.compile()

    out = graph.invoke({"messages": [{"role": "user", "content": "Write a tagline for a coffee brand."}], "rounds": 0})
    return out["messages"][-1].content


if __name__ == "__main__":
    print("[nae]     ", nae_version())
    print("[langgraph] ", langgraph_version())
