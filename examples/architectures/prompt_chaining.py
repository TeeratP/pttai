"""Prompt chaining — decompose a task into a fixed sequence of steps.

Each step's output feeds the next (``a > b > c``). An optional GATE between
steps can reject early — the canonical Anthropic "prompt chaining with a gate".

    input ─▶ [ extract ] ─▶ <gate?> ──pass──▶ [ expand ] ─▶ [ finalize ] ─▶ out
                                └──fail──▶ [ reject ] ─▶ out

    python examples/architectures/prompt_chaining.py
"""

import os
import sys

_EX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # examples/
sys.path.insert(0, os.path.dirname(_EX))  # repo root -> `import nae` works from a bare clone
sys.path.insert(0, _EX)  # -> `from _llm import get_llm`
from _llm import get_llm


def _gate(state) -> str:
    # Deterministic quality gate: proceed only if the extracted text is short
    # enough to be a clean spec. (A real pipeline might gate on a validator.)
    return "pass" if len(state["messages"][-1].content) < 300 else "fail"


def nae_version() -> str:
    from nae import AgentNode, ConditionNode, AgenticGraph

    extract = AgentNode(llm=get_llm(), node_prompt="Extract the core request as a one-line spec.")
    gate = ConditionNode(condition=_gate, choices=["pass", "fail"])
    expand = AgentNode(llm=get_llm(), node_prompt="Expand the spec into 3 bullet points.")
    finalize = AgentNode(llm=get_llm(), node_prompt="Write the final answer from the bullets.")
    reject = AgentNode(llm=get_llm(), node_prompt="Politely say the request was unclear.")

    extract > gate                    # chain the steps, then branch at the gate
    gate["pass"] > expand > finalize
    gate["fail"] > reject

    graph = AgenticGraph(start_node=extract, end_nodes={finalize, reject})
    return graph.invoke("Draft a tweet announcing our new pricing.")["messages"][-1].content


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
    builder.add_node("extract", step("Extract the core request as a one-line spec."))
    builder.add_node("expand", step("Expand the spec into 3 bullet points."))
    builder.add_node("finalize", step("Write the final answer from the bullets."))
    builder.add_node("reject", step("Politely say the request was unclear."))
    builder.add_edge(START, "extract")
    builder.add_conditional_edges("extract", _gate, {"pass": "expand", "fail": "reject"})
    builder.add_edge("expand", "finalize")
    builder.add_edge("finalize", END)
    builder.add_edge("reject", END)
    graph = builder.compile()

    out = graph.invoke({"messages": [{"role": "user", "content": "Draft a tweet announcing our new pricing."}]})
    return out["messages"][-1].content


if __name__ == "__main__":
    print("[nae]     ", nae_version())
    print("[langgraph] ", langgraph_version())
