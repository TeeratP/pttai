"""Supervisor — a router agent delegates to worker agents, in a loop.

A supervisor ``DecisionNode`` picks which worker handles the next step; the
worker runs and reports back; a gate loops to the supervisor or finishes. The
gate caps the loop (here at 2 delegations) so it always terminates — with the
offline fake the supervisor always routes to the first worker, so the cap is
what guarantees the run ends.

    ┌─────────────── continue ────────────────┐
    ▼                                          │
    [ supervisor ] ─┬─researcher─▶ [ researcher ]─┐
                    └─writer─────▶ [ writer ]────┴▶ <gate>
                                                     └─done─▶ [ report ] ─▶ out

    python examples/architectures/supervisor.py
"""

import os
import sys

_EX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # examples/
sys.path.insert(0, os.path.dirname(_EX))  # repo root -> `import nae` works from a bare clone
sys.path.insert(0, _EX)  # -> `from _llm import get_llm`
from _llm import get_llm

MAX_TURNS = 2


def _gate(state) -> str:
    # The supervisor logs a "supervisor:<choice>" line each time it runs. Finish
    # once we hit the delegation cap — the guarantee the loop terminates.
    turns = sum(1 for line in state["log"] if line.startswith("supervisor"))
    return "done" if turns >= MAX_TURNS else "continue"


def nae_version() -> str:
    from nae import AgentNode, DecisionNode, ConditionNode, AgenticGraph

    supervisor = DecisionNode(name="supervisor", llm=get_llm(),
                              node_prompt="Pick the next worker for this task.",
                              choices=["researcher", "writer"])
    researcher = AgentNode(name="researcher", llm=get_llm(), node_prompt="Research and report findings.")
    writer = AgentNode(name="writer", llm=get_llm(), node_prompt="Write a section from the findings.")
    gate = ConditionNode(name="gate", condition=_gate, choices=["continue", "done"])
    report = AgentNode(name="report", llm=get_llm(), node_prompt="Summarize the team's work.")

    supervisor["researcher"] > researcher
    supervisor["writer"] > writer
    researcher > gate                 # workers report back to the gate
    writer > gate
    gate["continue"] > supervisor     # loop back to delegate again
    gate["done"] > report

    graph = AgenticGraph(start_node=supervisor, end_nodes={report})
    return graph.invoke("Produce a short briefing on Mars.")["messages"][-1].content


# --- equivalent in raw LangGraph ---
def langgraph_version() -> str:
    from typing import Annotated, Literal
    import operator
    from langgraph.graph import StateGraph, MessagesState, START, END
    from langchain_core.messages import SystemMessage
    from pydantic import BaseModel

    llm = get_llm()

    class Which(BaseModel):
        choice: Literal["researcher", "writer"]

    router = llm.with_structured_output(Which)

    class State(MessagesState):
        next: str
        turns: Annotated[int, operator.add]   # counts supervisor visits

    def supervisor(state: State):
        choice = router.invoke([SystemMessage("Pick the next worker.")] + state["messages"]).choice
        return {"next": choice, "turns": 1}

    def worker(prompt):
        def node(state: State):
            return {"messages": [llm.invoke([SystemMessage(prompt)] + state["messages"])]}
        return node

    def route(state: State):  # cap the loop, else dispatch to the chosen worker
        return END if state["turns"] >= MAX_TURNS else state["next"]

    builder = StateGraph(State)
    builder.add_node("supervisor", supervisor)
    builder.add_node("researcher", worker("Research and report findings."))
    builder.add_node("writer", worker("Write a section from the findings."))
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", route,
                                  {"researcher": "researcher", "writer": "writer", END: END})
    builder.add_edge("researcher", "supervisor")   # report back
    builder.add_edge("writer", "supervisor")
    graph = builder.compile()

    out = graph.invoke({"messages": [{"role": "user", "content": "Produce a short briefing on Mars."}], "turns": 0})
    return out["messages"][-1].content


if __name__ == "__main__":
    print("[nae]     ", nae_version())
    print("[langgraph] ", langgraph_version())
