"""Headline result: build-time bugs pttai's validator catches that raw LangGraph does NOT.

For each bug class we (1) construct the offending pttai graph and CONFIRM it
fails at BUILD time (construction), capturing the verbatim error and timing the
build; then (2) construct the equivalent raw-LangGraph graph and observe what
actually happens — compiles silently, fails at compile, or only fails at RUNTIME
after >=1 model call. Model calls are counted with an offline fake LLM (no API
key) so we can quantify wasted calls before LangGraph fails.

Nothing here is asserted-into-success blindly: if a bug class turns out NOT to be
caught at build (by pttai) or IS also caught by LangGraph, that is printed as-is.

    PYTHONPATH=. .venv/bin/python eval/validator_bugs.py
"""

import os
import sys
import time
import traceback
from typing import Literal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))

from _llm import get_llm  # offline fake unless OPENAI_API_KEY is set
from pttai import AgentNode, DecisionNode, AgenticGraph
from pttai.validation import GraphValidationError


class CountingLLM:
    """Wraps the offline fake LLM and counts every ``.invoke`` (including bound /
    structured-output variants), so we can report model calls wasted before a raw
    LangGraph graph fails at runtime. The counter is shared across all derived
    (bind_tools / with_structured_output) copies."""

    def __init__(self, inner, counter):
        self._inner, self._counter = inner, counter

    def invoke(self, *a, **k):
        self._counter[0] += 1
        return self._inner.invoke(*a, **k)

    def bind_tools(self, tools):
        return CountingLLM(self._inner.bind_tools(tools), self._counter)

    def with_structured_output(self, model):
        return CountingLLM(self._inner.with_structured_output(model), self._counter)


def _counting_llm():
    counter = [0]
    return CountingLLM(get_llm(), counter), counter


def _build_pttai(build):
    """Run a pttai graph constructor; return (raised, error_type, message, ms)."""
    t0 = time.perf_counter()
    try:
        build()
        ms = (time.perf_counter() - t0) * 1000
        return False, None, "", ms
    except Exception as e:  # noqa: BLE001 — we WANT to inspect any build failure
        ms = (time.perf_counter() - t0) * 1000
        return True, type(e).__name__, str(e), ms


# --------------------------------------------------------------------------
# Bug (a): read-before-write of a state key
# --------------------------------------------------------------------------
def bug_read_before_write():
    llm = get_llm()

    def build():
        write = AgentNode(name="write", llm=llm, node_prompt="use {plan}",
                          reads=["plan"], writes=["messages"])
        planner = AgentNode(name="planner", llm=llm, node_prompt="plan",
                            reads=["messages"], writes={"plan": str})
        write > planner                       # producer of `plan` runs LAST
        AgenticGraph(start_node=write, end_nodes={planner})

    raised, etype, msg, ms = _build_pttai(build)

    # Raw LangGraph equivalent: `write` reads state["plan"] before its producer
    # (`planner`) runs. Compiles fine; with `plan` seeded empty (None) at invoke,
    # the read does NOT even error — the node silently runs on a null plan and
    # produces wrong output (a latent correctness bug LangGraph never surfaces).
    from langgraph.graph import StateGraph, MessagesState, START, END
    from langchain_core.messages import SystemMessage

    class State(MessagesState):
        plan: str

    llm2, counter = _counting_llm()

    def write_node(state: State):
        prompt = f"use {state['plan']}"       # KeyError here at runtime
        return {"messages": [llm2.invoke([SystemMessage(prompt)] + state["messages"])]}

    def planner_node(state: State):
        return {"plan": "a plan"}

    b = StateGraph(State)
    b.add_node("write", write_node)
    b.add_node("planner", planner_node)
    b.add_edge(START, "write")
    b.add_edge("write", "planner")
    b.add_edge("planner", END)
    lg = _run_langgraph(b, {"messages": [{"role": "user", "content": "hi"}], "plan": None})
    # note: reader runs first, so 0 model calls precede the failure here.
    lg["wasted_calls"] = counter[0]
    return _result("read-before-write of a state key", raised, etype, msg, ms, lg)


# --------------------------------------------------------------------------
# Bug (b): dangling decision choice (a choice with no .child)
# --------------------------------------------------------------------------
def bug_dangling_choice():
    llm = get_llm()

    def build():
        d = DecisionNode(name="d", llm=llm, node_prompt="pick", choices=["a", "b"])
        h = AgentNode(name="h", llm=llm, node_prompt="handle")
        d["a"] > h                            # choice "b" is never wired
        AgenticGraph(start_node=d, end_nodes={h})

    raised, etype, msg, ms = _build_pttai(build)

    # Raw LangGraph equivalent: a structured-output router + conditional edges
    # whose path map omits the "b" branch. Compiles fine. The fake router
    # deterministically returns the FIRST Literal, so we order it "b" first to
    # deterministically hit the UNWIRED branch -> runtime error after 1 call.
    from langgraph.graph import StateGraph, MessagesState, START, END
    from langchain_core.messages import SystemMessage
    from pydantic import BaseModel

    class Route(BaseModel):
        choice: Literal["b", "a"]             # fake picks "b" (first) -> unwired

    llm2, counter = _counting_llm()
    router = llm2.with_structured_output(Route)

    def decide(state: MessagesState):
        c = router.invoke([SystemMessage("pick")] + state["messages"]).choice
        return {"messages": state["messages"], "_route": c}

    def route_fn(state):
        return state["_route"]

    def handler(state: MessagesState):
        return {"messages": [llm2.invoke(state["messages"])]}

    class S(MessagesState):
        _route: str

    b = StateGraph(S)
    b.add_node("d", decide)
    b.add_node("h", handler)
    b.add_edge(START, "d")
    b.add_conditional_edges("d", route_fn, {"a": "h"})   # "b" omitted
    b.add_edge("h", END)
    lg = _run_langgraph(b, {"messages": [{"role": "user", "content": "hi"}]})
    lg["wasted_calls"] = counter[0]
    return _result("dangling decision choice (no .child)", raised, etype, msg, ms, lg)


# --------------------------------------------------------------------------
# Bug (c): a non-end node whose .child is None
# --------------------------------------------------------------------------
def bug_dangling_nonend():
    llm = get_llm()

    def build():
        a = AgentNode(name="a", llm=llm, node_prompt="one")
        bb = AgentNode(name="b", llm=llm, node_prompt="two")
        a > bb                                # forgot to declare b as an end node
        AgenticGraph(start_node=a, end_nodes=set())

    raised, etype, msg, ms = _build_pttai(build)

    # Raw LangGraph equivalent: node "b" has no outgoing edge / no edge to END.
    from langgraph.graph import StateGraph, MessagesState, START
    from langchain_core.messages import SystemMessage

    llm2, counter = _counting_llm()

    def step(prompt):
        def node(state: MessagesState):
            return {"messages": [llm2.invoke([SystemMessage(prompt)] + state["messages"])]}
        return node

    b = StateGraph(MessagesState)
    b.add_node("a", step("one"))
    b.add_node("b", step("two"))
    b.add_edge(START, "a")
    b.add_edge("a", "b")                       # "b" dead-ends: no edge to END
    lg = _run_langgraph(b, {"messages": [{"role": "user", "content": "hi"}]})
    lg["wasted_calls"] = counter[0]
    return _result("non-end node whose .child is None", raised, etype, msg, ms, lg)


# --------------------------------------------------------------------------
# Bug (d): duplicate node names
# --------------------------------------------------------------------------
def bug_duplicate_names():
    llm = get_llm()

    def build():
        a = AgentNode(name="dup", llm=llm, node_prompt="one")
        bb = AgentNode(name="dup", llm=llm, node_prompt="two")
        a > bb
        AgenticGraph(start_node=a, end_nodes={bb})

    raised, etype, msg, ms = _build_pttai(build)

    # Raw LangGraph equivalent: add_node("dup", ...) twice.
    from langgraph.graph import StateGraph, MessagesState, START, END
    from langchain_core.messages import SystemMessage

    llm2, counter = _counting_llm()

    def step(prompt):
        def node(state: MessagesState):
            return {"messages": [llm2.invoke([SystemMessage(prompt)] + state["messages"])]}
        return node

    lg = {"phase": None, "error_type": None, "message": ""}
    try:
        b = StateGraph(MessagesState)
        b.add_node("dup", step("one"))
        b.add_node("dup", step("two"))         # duplicate name
        b.add_edge(START, "dup")
        b.add_edge("dup", END)
        compiled = b.compile()
        try:
            compiled.invoke({"messages": [{"role": "user", "content": "hi"}]})
            lg["phase"] = "silent (no error at build or runtime)"
        except Exception as e:  # noqa: BLE001
            lg["phase"] = "runtime"
            lg["error_type"] = type(e).__name__
            lg["message"] = str(e).splitlines()[0]
    except Exception as e:  # noqa: BLE001
        lg["phase"] = "build (add_node/compile)"
        lg["error_type"] = type(e).__name__
        lg["message"] = str(e).splitlines()[0]
    lg["wasted_calls"] = counter[0]
    return _result("duplicate node names", raised, etype, msg, ms, lg)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _run_langgraph(builder, invoke_input):
    """Compile + invoke a raw-LangGraph builder; report the phase it fails in."""
    out = {"phase": None, "error_type": None, "message": ""}
    try:
        compiled = builder.compile()
    except Exception as e:  # noqa: BLE001
        out["phase"] = "build (compile)"
        out["error_type"] = type(e).__name__
        out["message"] = str(e).splitlines()[0]
        return out
    try:
        compiled.invoke(invoke_input)
        out["phase"] = "silent (no error at build or runtime)"
    except Exception as e:  # noqa: BLE001
        out["phase"] = "runtime"
        out["error_type"] = type(e).__name__
        out["message"] = str(e).splitlines()[0]
    return out


def _result(bug, raised, etype, msg, ms, lg):
    return {
        "bug": bug,
        "pttai_raised": raised,
        "pttai_error_type": etype,
        "pttai_message": msg,
        "pttai_build_ms": ms,
        "lg": lg,
    }


def main():
    print("=" * 78)
    print("Validator bug-catching: pttai (build time) vs. raw LangGraph")
    print(f"LLM backend: {'REAL OpenAI' if os.environ.get('OPENAI_API_KEY') else 'offline fake'}")
    print("=" * 78)

    results = [
        bug_read_before_write(),
        bug_dangling_choice(),
        bug_dangling_nonend(),
        bug_duplicate_names(),
    ]

    for r in results:
        print(f"\n### {r['bug']}")
        if r["pttai_raised"]:
            print(f"  pttai: caught at BUILD in {r['pttai_build_ms']:.2f} ms "
                  f"({r['pttai_error_type']})")
            print("  verbatim error:")
            for line in r["pttai_message"].splitlines():
                print(f"    | {line}")
        else:
            print(f"  pttai: DID NOT catch it at build "
                  f"(built in {r['pttai_build_ms']:.2f} ms) — REPORTING HONESTLY")
        lg = r["lg"]
        print(f"  LangGraph: fails at -> {lg['phase']}"
              + (f" ({lg['error_type']}: {lg['message']})" if lg["error_type"] else ""))
        print(f"  LangGraph model calls before failure: {lg['wasted_calls']}")

    # Summary table
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    hdr = ("bug class", "pttai", "build ms", "LangGraph", "wasted calls")
    w = (34, 14, 9, 40, 12)

    def row(cols):
        return "  ".join(str(c).ljust(w[i]) for i, c in enumerate(cols))

    print(row(hdr))
    print("-" * (sum(w) + 2 * (len(w) - 1)))
    for r in results:
        lg = r["lg"]
        pttai_cell = "caught@build" if r["pttai_raised"] else "NOT caught"
        lg_cell = lg["phase"]
        if lg["error_type"]:
            lg_cell += f" [{lg['error_type']}]"
        print(row((r["bug"], pttai_cell, f"{r['pttai_build_ms']:.2f}",
                   lg_cell, lg["wasted_calls"])))

    caught = sum(1 for r in results if r["pttai_raised"])
    print(f"\npttai caught {caught}/{len(results)} bug classes at build time.")
    lg_build = sum(1 for r in results if r["lg"]["phase"] and r["lg"]["phase"].startswith("build"))
    lg_runtime = sum(1 for r in results if r["lg"]["phase"] == "runtime")
    lg_silent = sum(1 for r in results if r["lg"]["phase"] and r["lg"]["phase"].startswith("silent"))
    print(f"raw LangGraph: {lg_build} caught at build, {lg_runtime} only at runtime, "
          f"{lg_silent} silent.")


if __name__ == "__main__":
    main()
