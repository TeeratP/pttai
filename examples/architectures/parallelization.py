"""Parallelization — sectioning / voting: run branches CONCURRENTLY, then join.

A question is framed once, fanned out to three rival personas that argue IN
PARALLEL, then a chair weighs all three into one verdict. This is the pattern in
``examples/panel.py``, folded here into the nae-vs-LangGraph format.

    ┌▶ [ optimist ]  ─┐
    [ frame ] ─┼▶ [ skeptic ]   ─┼▶ [ verdict ] ─▶ out
    └▶ [ pragmatist ]─┘   (join deferred until all three finish)

    python examples/architectures/parallelization.py
"""

import os
import sys

_EX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # examples/
sys.path.insert(0, os.path.dirname(_EX))  # repo root -> `import nae` works from a bare clone
sys.path.insert(0, _EX)  # -> `from _llm import get_llm`
from _llm import get_llm

QUESTION = ("Should an early-stage SaaS startup rewrite its monolith into "
            "microservices to win enterprise customers?")


def nae_version() -> str:
    from nae import AgentNode, AgenticGraph, fanout

    frame = AgentNode(llm=get_llm(), node_prompt="Restate the question as ONE concrete decision.")
    optimist = AgentNode(llm=get_llm(), node_prompt="Argue FOR the bold move. Two sentences.")
    skeptic = AgentNode(llm=get_llm(), node_prompt="Argue AGAINST. Name the biggest risks. Two sentences.")
    pragmatist = AgentNode(llm=get_llm(), node_prompt="Propose the smallest de-risking next step.")
    verdict = AgentNode(llm=get_llm(), node_prompt="Weigh all three and give a one-paragraph verdict.")

    # The three personas run IN PARALLEL, then join at `verdict`.
    frame > fanout(optimist, skeptic, pragmatist) > verdict

    graph = AgenticGraph(start_node=frame, end_nodes={verdict})
    return graph.invoke(QUESTION)["messages"][-1].content


# --- equivalent in raw LangGraph ---
def langgraph_version() -> str:
    from langgraph.graph import StateGraph, MessagesState, START, END
    from langchain_core.messages import SystemMessage

    llm = get_llm()

    def step(prompt):
        def node(state: MessagesState):
            return {"messages": [llm.invoke([SystemMessage(prompt)] + state["messages"])]}
        return node

    builder = StateGraph(MessagesState)
    builder.add_node("frame", step("Restate the question as ONE concrete decision."))
    builder.add_node("optimist", step("Argue FOR the bold move. Two sentences."))
    builder.add_node("skeptic", step("Argue AGAINST. Name the biggest risks. Two sentences."))
    builder.add_node("pragmatist", step("Propose the smallest de-risking next step."))
    # defer=True makes verdict fire ONCE, after all three branches finish.
    builder.add_node("verdict", step("Weigh all three and give a one-paragraph verdict."), defer=True)
    builder.add_edge(START, "frame")
    for persona in ("optimist", "skeptic", "pragmatist"):
        builder.add_edge("frame", persona)     # fan out
        builder.add_edge(persona, "verdict")   # fan in (join)
    builder.add_edge("verdict", END)
    graph = builder.compile()

    out = graph.invoke({"messages": [{"role": "user", "content": QUESTION}]})
    return out["messages"][-1].content


if __name__ == "__main__":
    print("[nae]     ", nae_version())
    print("[langgraph] ", langgraph_version())
