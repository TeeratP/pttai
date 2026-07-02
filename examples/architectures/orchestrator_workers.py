"""Orchestrator–workers — fan a task out over sub-tasks, then synthesize.

An orchestrator breaks the job into sub-tasks; a worker runs on EACH sub-task in
parallel (``worker.map("subtasks")``, one LangGraph ``Send`` per item); a
synthesizer joins the results once. The number of workers is data-driven.

    [ plan ] ─▶ worker.map("subtasks") ═══▶ [ worker ] × N ═══▶ [ synthesize ] ─▶ out

    python examples/architectures/orchestrator_workers.py
"""

import os
import sys

_EX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # examples/
sys.path.insert(0, os.path.dirname(_EX))  # repo root -> `import nae` works from a bare clone
sys.path.insert(0, _EX)  # -> `from _llm import get_llm`
from _llm import get_llm

# Sub-tasks are seeded at invoke here so the example runs offline; with a real
# model the orchestrator would GENERATE them (e.g. writes={"subtasks": list}).
SUBTASKS = ["research the market", "draft the intro", "outline the pricing"]


def nae_version() -> list:
    from nae import AgentNode, AgenticGraph

    plan = AgentNode(llm=get_llm(), node_prompt="Break the request into sub-tasks.")
    worker = AgentNode(llm=get_llm(), node_prompt="Complete this one sub-task.")
    synthesize = AgentNode(llm=get_llm(), node_prompt="Combine the sub-task results into one answer.")

    plan > worker.map("subtasks") > synthesize     # dynamic fan-out over the list

    graph = AgenticGraph(start_node=plan, end_nodes={synthesize})
    out = graph.invoke("Produce a launch plan.", subtasks=SUBTASKS)
    return out["messages"]


# --- equivalent in raw LangGraph ---
def langgraph_version() -> list:
    from langgraph.graph import StateGraph, MessagesState, START, END
    from langgraph.types import Send
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = get_llm()

    class State(MessagesState):
        subtasks: list

    def plan(state: State):
        return {"messages": [llm.invoke([SystemMessage("Break the request into sub-tasks.")] + state["messages"])]}

    def assign(state: State):  # one Send per sub-task -> parallel workers
        return [Send("worker", {"messages": [HumanMessage(t)]}) for t in state["subtasks"]]

    def worker(state: State):
        return {"messages": [llm.invoke([SystemMessage("Complete this one sub-task.")] + state["messages"])]}

    def synthesize(state: State):
        return {"messages": [llm.invoke([SystemMessage("Combine the sub-task results into one answer.")] + state["messages"])]}

    builder = StateGraph(State)
    builder.add_node("plan", plan)
    builder.add_node("worker", worker)
    builder.add_node("synthesize", synthesize, defer=True)  # join once, after all workers
    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", assign, ["worker"])
    builder.add_edge("worker", "synthesize")
    builder.add_edge("synthesize", END)
    graph = builder.compile()

    out = graph.invoke({"messages": [{"role": "user", "content": "Produce a launch plan."}], "subtasks": SUBTASKS})
    return out["messages"]


if __name__ == "__main__":
    print("[nae]      messages after map+synthesize:", len(nae_version()))
    print("[langgraph]  messages after map+synthesize:", len(langgraph_version()))
