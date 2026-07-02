"""13 · Free observability — the ``token`` and ``log`` channels.

Every pttai graph tracks two things for you with **zero wiring**:

  * ``state['token']`` — total token spend across EVERY LLM call in the run
    (including tool-loop calls and parallel fan-out), as
    ``{model_name: usage_metadata}``, deep-summed per model.
  * ``state['log']``   — a per-node trace line for every node/tool call.

In raw LangGraph you'd hand-wire a usage callback + a custom reducer channel and
read ``response.usage_metadata`` in every node yourself (see the block below).
pttai does it automatically — the value prop of this example.

    python examples/basics/13_token_and_log.py

Offline (no OPENAI_API_KEY) the scripted fake model reports NO usage_metadata,
so ``token`` is an empty ``{}`` — the accounting mechanism runs, there's just
nothing to count. With OPENAI_API_KEY set, the SAME code fills ``token`` with
real per-model counts (shape shown in the printout below).
"""

import os
import sys

_EX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # examples/
sys.path.insert(0, os.path.dirname(_EX))  # repo root -> `import pttai` works from a bare clone
sys.path.insert(0, _EX)  # -> `from _llm import get_llm`
from _llm import get_llm


def pttai_version() -> dict:
    from pttai import AgentNode, AgenticGraph

    # Naming is optional: `outline` gets its name inferred from the variable it's
    # assigned to, while `draft` is named explicitly. Both work — name a node when
    # you want a stable label in the log/summary, skip it otherwise.
    outline = AgentNode(llm=get_llm(), node_prompt="Outline the answer.")
    draft = AgentNode(name="draft", llm=get_llm(), node_prompt="Write a draft from the outline.")

    outline > draft

    graph = AgenticGraph(start_node=outline, end_nodes={draft})
    # Seed `log: []` so the trace channel comes back; `token` accrues on its own.
    state = graph.invoke({"messages": ["Explain what a monad is."], "log": []})
    return {"token": state.get("token", {}), "log": state["log"]}


# --- equivalent in raw LangGraph ---
# To get the SAME token total you must: (1) declare a custom reduced channel with
# a summing reducer, and (2) read `response.usage_metadata` in EVERY node and
# return it. Miss one node and your total is silently wrong. pttai's LLMNode does
# all of this for you.
def langgraph_version() -> dict:
    import operator
    from typing import Annotated

    from langchain_core.messages import AIMessage, SystemMessage
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages

    def sum_tokens(left: dict, right: dict) -> dict:
        out = dict(left or {})
        for k, v in (right or {}).items():
            out[k] = out.get(k, 0) + v
        return out

    class State(dict):
        messages: Annotated[list, add_messages]
        log: Annotated[list, operator.add]
        token: Annotated[dict, sum_tokens]  # custom channel — you own the reducer

    llm = get_llm()

    def step(name, prompt):
        def node(state):
            resp = llm.invoke([SystemMessage(prompt)] + state["messages"])
            delta = {"messages": [resp], "log": [f"{name}:{resp.content}"]}
            usage = getattr(resp, "usage_metadata", None)  # you must remember this
            if usage:
                delta["token"] = {"total_tokens": usage.get("total_tokens", 0)}
            return delta
        return node

    builder = StateGraph(State)
    builder.add_node("outline", step("outline", "Outline the answer."))
    builder.add_node("draft", step("draft", "Write a draft from the outline."))
    builder.add_edge(START, "outline")
    builder.add_edge("outline", "draft")
    builder.add_edge("draft", END)
    graph = builder.compile()

    state = graph.invoke(
        {"messages": [{"role": "user", "content": "Explain what a monad is."}], "log": [], "token": {}}
    )
    return {"token": state.get("token", {}), "log": state["log"]}


if __name__ == "__main__":
    result = pttai_version()

    print("[pttai] token usage (total across the whole run):")
    print("   ", result["token"] or "{}  <- empty offline; the fake reports no usage_metadata")
    print("    With a real model this looks like:")
    print("    {'gpt-5.4-nano': {'input_tokens': 42, 'output_tokens': 88, 'total_tokens': 130, ...}}")
    print()
    print("[pttai] per-node trace (state['log']):")
    for line in result["log"]:
        print("   ", line)

    print()
    print("[langgraph] same two channels, hand-wired:")
    lg = langgraph_version()
    print("    token:", lg["token"] or "{}  <- empty offline (same reason)")
    for line in lg["log"]:
        print("   ", line)
