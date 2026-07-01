"""04 · Decision routing — LLM picks the branch.

A ``DecisionNode`` calls the model with constrained (structured) output so it
MUST return one of ``choices``, then routes to the handler wired to that choice
via ``decision["label"] > handler``. The label goes to a dedicated ``decision``
channel, never into the conversation.

    python examples/basics/04_decision_routing.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _llm import get_llm


def pttai_version() -> str:
    from pttai import AgentNode, DecisionNode, AgenticGraph

    classify = DecisionNode(
        llm=get_llm(),
        node_prompt="Classify the sentiment of the message.",
        choices=["positive", "negative"],
    )
    praise = AgentNode(llm=get_llm(), node_prompt="Thank the happy customer.")
    apologize = AgentNode(llm=get_llm(), node_prompt="Apologize to the unhappy customer.")

    classify["positive"] > praise        # wire each choice to its handler
    classify["negative"] > apologize

    graph = AgenticGraph(start_node=classify, end_nodes={praise, apologize})
    out = graph.invoke("I absolutely love this product!")
    return f"routed to {out['decision']!r} -> {out['messages'][-1].content}"


# --- equivalent in raw LangGraph ---
def langgraph_version() -> str:
    from typing import Literal
    from langgraph.graph import StateGraph, MessagesState, START, END
    from langchain_core.messages import SystemMessage
    from pydantic import BaseModel

    class Route(BaseModel):
        choice: Literal["positive", "negative"]

    llm = get_llm()
    router = llm.with_structured_output(Route)

    def classify(state: MessagesState):
        prompt = [SystemMessage("Classify the sentiment of the message.")] + state["messages"]
        return {"choice": router.invoke(prompt).choice}

    def praise(state: MessagesState):
        return {"messages": [llm.invoke([SystemMessage("Thank the happy customer.")] + state["messages"])]}

    def apologize(state: MessagesState):
        return {"messages": [llm.invoke([SystemMessage("Apologize to the unhappy customer.")] + state["messages"])]}

    class State(MessagesState):
        choice: str

    builder = StateGraph(State)
    builder.add_node("classify", classify)
    builder.add_node("praise", praise)
    builder.add_node("apologize", apologize)
    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", lambda s: s["choice"],
                                  {"positive": "praise", "negative": "apologize"})
    builder.add_edge("praise", END)
    builder.add_edge("apologize", END)
    graph = builder.compile()

    out = graph.invoke({"messages": [{"role": "user", "content": "I absolutely love this product!"}]})
    return f"routed to {out['choice']!r} -> {out['messages'][-1].content}"


if __name__ == "__main__":
    print("[pttai]     ", pttai_version())
    print("[langgraph] ", langgraph_version())
