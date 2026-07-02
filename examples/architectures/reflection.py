"""Reflection — generate, self-critique, then revise (one pass).

The model drafts an answer, critiques its OWN draft, then rewrites using the
critique. A single, always-terminating pass (unlike evaluator_optimizer, which
loops). Chain three agents; each sees everything before it.

    [ draft ] ─▶ [ critique ] ─▶ [ revise ] ─▶ out

    python examples/architectures/reflection.py
"""

import os
import sys

_EX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # examples/
sys.path.insert(0, os.path.dirname(_EX))  # repo root -> `import nae` works from a bare clone
sys.path.insert(0, _EX)  # -> `from _llm import get_llm`
from _llm import get_llm


def nae_version() -> str:
    from nae import AgentNode, AgenticGraph

    draft = AgentNode(llm=get_llm(), node_prompt="Draft an answer to the question.")
    critique = AgentNode(llm=get_llm(), node_prompt="Critique the draft above: what is weak or missing?")
    revise = AgentNode(llm=get_llm(), node_prompt="Rewrite the draft, fixing every point in the critique.")

    draft > critique > revise

    graph = AgenticGraph(start_node=draft, end_nodes={revise})
    return graph.invoke("Explain why the sky is blue.")["messages"][-1].content


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
    builder.add_node("draft", step("Draft an answer to the question."))
    builder.add_node("critique", step("Critique the draft above: what is weak or missing?"))
    builder.add_node("revise", step("Rewrite the draft, fixing every point in the critique."))
    builder.add_edge(START, "draft")
    builder.add_edge("draft", "critique")
    builder.add_edge("critique", "revise")
    builder.add_edge("revise", END)
    graph = builder.compile()

    out = graph.invoke({"messages": [{"role": "user", "content": "Explain why the sky is blue."}]})
    return out["messages"][-1].content


if __name__ == "__main__":
    print("[nae]     ", nae_version())
    print("[langgraph] ", langgraph_version())
