"""Document triage — classify an incoming document, route to a handler.

An NLP routing pipeline: a ``DecisionNode`` classifies the document into one of
its ``choices`` using constrained structured output (the model MUST return a
valid label), then routes to the matching specialized handler via
``triage["label"] > handler``. The label goes to the dedicated per-node
``decision_triage`` channel, never into the conversation.

    python examples/nlp/doc_triage.py
"""

import os
import sys

_EX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # examples/
sys.path.insert(0, os.path.dirname(_EX))  # repo root -> `import nae` works from a bare clone
sys.path.insert(0, _EX)  # -> `from _llm import get_llm`
from _llm import get_llm

DOCUMENT = "The export button throws a 500 error every time I click it. Please fix."


def nae_version() -> str:
    from nae import AgentNode, DecisionNode, AgenticGraph

    triage = DecisionNode(
        llm=get_llm(),
        node_prompt="Classify this incoming document by its intent.",
        choices=["bug_report", "feature_request", "question"],
    )
    bug = AgentNode(llm=get_llm(), node_prompt="Acknowledge the bug and ask for reproduction steps.")
    feature = AgentNode(llm=get_llm(), node_prompt="Thank the user and log the feature request.")
    question = AgentNode(llm=get_llm(), node_prompt="Answer the user's question helpfully.")

    triage["bug_report"] > bug              # wire each class to its handler
    triage["feature_request"] > feature
    triage["question"] > question

    graph = AgenticGraph(start_node=triage, end_nodes={bug, feature, question})
    out = graph.invoke(DOCUMENT)
    return f"routed to {out['decision_triage']!r} -> {out['messages'][-1].content}"


# --- equivalent in raw LangGraph ---
def langgraph_version() -> str:
    from typing import Literal
    from langgraph.graph import StateGraph, MessagesState, START, END
    from langchain_core.messages import SystemMessage
    from pydantic import BaseModel

    class Route(BaseModel):
        choice: Literal["bug_report", "feature_request", "question"]

    llm = get_llm()
    router = llm.with_structured_output(Route)

    def triage(state: MessagesState):
        prompt = [SystemMessage("Classify this incoming document by its intent.")] + state["messages"]
        return {"choice": router.invoke(prompt).choice}

    def handler(prompt):
        def node(state: MessagesState):
            return {"messages": [llm.invoke([SystemMessage(prompt)] + state["messages"])]}
        return node

    class State(MessagesState):
        choice: str

    builder = StateGraph(State)
    builder.add_node("triage", triage)
    builder.add_node("bug", handler("Acknowledge the bug and ask for reproduction steps."))
    builder.add_node("feature", handler("Thank the user and log the feature request."))
    builder.add_node("question", handler("Answer the user's question helpfully."))
    builder.add_edge(START, "triage")
    builder.add_conditional_edges("triage", lambda s: s["choice"],
                                  {"bug_report": "bug", "feature_request": "feature", "question": "question"})
    for h in ("bug", "feature", "question"):
        builder.add_edge(h, END)
    graph = builder.compile()

    out = graph.invoke({"messages": [{"role": "user", "content": DOCUMENT}]})
    return f"routed to {out['choice']!r} -> {out['messages'][-1].content}"


if __name__ == "__main__":
    print("[nae]     ", nae_version())
    print("[langgraph] ", langgraph_version())
