"""Labeled dataflow-bug corpus for LLM agent graphs.

Each BUGGY item is a real graph containing exactly one dataflow bug of a known
class, plus the equivalent raw-LangGraph graph so we can measure where the bug
surfaces in each framework. Each CLEAN item is a real, working pipeline from
``examples/`` (the false-positive set — the validator must NOT flag these).

Honesty notes baked into the labels (`category`):
  * ``duplicate-node-names`` is caught at BUILD by *both* frameworks — raw
    LangGraph's ``add_node`` raises ``Node `x` already present.`` So it is NOT a
    pttai differentiator (``category="both"``).
  * ``dead-end-node`` is NOT a LangGraph bug. A node with no outgoing edge is a
    *legal implicit terminal* in LangGraph — it compiles and runs to a normal
    halt (verified empirically on LangGraph 1.2.x: ``compile()`` does not raise
    and ``invoke()`` returns without error). pttai rejects it only because its
    own DSL *requires* every terminal to be declared in ``end_nodes``. So it is
    pttai DSL strictness, not a LangGraph defect: ``category="dsl-strictness"``.
    It does NOT count toward the pttai-only differentiator headline or the
    wasted-call total.
  * ``cyclic-read-before-write`` is a genuine pttai-only-at-build catch (issue
    #34): a graph with a cycle where a node reads a key produced ONLY by a
    downstream node that loops back. On the FIRST iteration the key is absent,
    so raw LangGraph KeyErrors at runtime; pttai's back-edge-aware availability
    fixpoint rejects it at build.
  * The validator checks state-key PRESENCE / availability, not Python value
    types — there is no true "type mismatch" detection. The closest firing
    check is ``prompt-placeholder-mismatch`` (a ``{name}`` in a node_prompt with
    no matching scalar read -> guaranteed runtime KeyError), included as its own
    class.

Note on the LangGraph "wasted calls" figure: it is a SIMULATED, worst-case
measurement, not real-model cost. Every LangGraph baseline runs against a
counting *fake* LLM (offline), and the dangling-choice baselines deterministically
hit the omitted branch because the fake ``with_structured_output`` picks the
first ``Literal`` value and the corpus orders the unwired choice first. That is a
worst-case ordering, disclosed here honestly — not an observation of a real model.
"""

import os
from dataclasses import dataclass
from typing import Callable, List, Literal, Optional

from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from pydantic import BaseModel

from pttai import AgentNode, DecisionNode, ConditionNode, AgenticGraph
from pttai.state import AgenticState

from _common import counting_llm, step, run_langgraph, HELLO, get_llm


@dataclass
class Item:
    id: str
    bug_class: str
    label: str                       # "buggy" | "clean"
    category: str                    # buggy: "differentiator" | "both" | "dsl-strictness"
    build_pttai: Callable            # constructs the AgenticGraph (raises if invalid)
    langgraph: Optional[Callable] = None   # buggy only: () -> phase dict w/ wasted_calls

    @property
    def pttai_only(self) -> bool:
        """A genuine pttai-only-at-build differentiator (excludes ``both`` and
        ``dsl-strictness``). Kept for the ``results.csv`` column and downstream
        chart code."""
        return self.category == "differentiator"


# ===========================================================================
# BUG CLASS 1 — read-before-write of a computed state key  (pttai-only)
# ===========================================================================
def _rbw1_pttai():
    llm = get_llm()
    write = AgentNode(name="write", llm=llm, node_prompt="use {plan}",
                      reads=["plan"], writes=["messages"])
    planner = AgentNode(name="planner", llm=llm, node_prompt="plan",
                        writes={"plan": str})
    write > planner                                   # producer of `plan` runs LAST
    AgenticGraph(start_node=write, end_nodes={planner})


def _rbw1_lg():
    llm, counter = counting_llm()

    class S(MessagesState):
        plan: str

    def build():
        b = StateGraph(S)
        b.add_node("write", lambda s: {"messages": [llm.invoke(
            [SystemMessage(f"use {s['plan']}")] + s["messages"])]})  # KeyError: 'plan'
        b.add_node("planner", lambda s: {"plan": "a plan"})
        b.add_edge(START, "write")
        b.add_edge("write", "planner")
        b.add_edge("planner", END)
        return b

    out = run_langgraph(build, HELLO)
    out["wasted_calls"] = counter[0]
    return out


def _rbw2_pttai():
    llm = get_llm()
    start = AgentNode(name="start", llm=llm, node_prompt="start")
    summarize = AgentNode(name="summarize", llm=llm, node_prompt="summarize {draft}",
                          reads=["draft"], writes=["messages"])
    drafter = AgentNode(name="drafter", llm=llm, node_prompt="draft",
                        writes={"draft": str})
    start > summarize > drafter                       # summarize reads `draft` too early
    AgenticGraph(start_node=start, end_nodes={drafter})


def _rbw2_lg():
    llm, counter = counting_llm()

    class S(MessagesState):
        draft: str

    def build():
        b = StateGraph(S)
        b.add_node("start", step(llm, "start"))
        b.add_node("summarize", lambda s: {"messages": [llm.invoke(
            [SystemMessage(f"summarize {s['draft']}")] + s["messages"])]})  # KeyError
        b.add_node("drafter", lambda s: {"draft": "a draft"})
        b.add_edge(START, "start")
        b.add_edge("start", "summarize")
        b.add_edge("summarize", "drafter")
        b.add_edge("drafter", END)
        return b

    out = run_langgraph(build, HELLO)
    out["wasted_calls"] = counter[0]
    return out


def _rbw3_pttai():
    llm = get_llm()
    a = AgentNode(name="a", llm=llm, node_prompt="start")
    scorer = AgentNode(name="scorer", llm=llm, node_prompt="score", writes={"score": str})
    reader = AgentNode(name="reader", llm=llm, node_prompt="use {score}", reads=["score"])
    join = AgentNode(name="join", llm=llm, node_prompt="join")
    (a > [scorer, reader]) > join                     # reader reads `score` from its sibling
    AgenticGraph(start_node=a, end_nodes={join})


def _rbw3_lg():
    llm, counter = counting_llm()

    class S(MessagesState):
        score: str

    def build():
        b = StateGraph(S)
        b.add_node("a", step(llm, "start"))
        b.add_node("scorer", lambda s: {"score": "9"})
        b.add_node("reader", lambda s: {"messages": [llm.invoke(
            [SystemMessage(f"use {s['score']}")] + s["messages"])]})  # KeyError: 'score'
        b.add_node("join", step(llm, "join"))
        b.add_edge(START, "a")
        b.add_edge("a", "scorer")
        b.add_edge("a", "reader")
        b.add_edge("scorer", "join")
        b.add_edge("reader", "join")
        b.add_edge("join", END)
        return b

    out = run_langgraph(build, HELLO)
    out["wasted_calls"] = counter[0]
    return out


# ===========================================================================
# BUG CLASS 1b — CYCLIC read-before-write: a node reads a key produced ONLY by
#                a DOWNSTREAM node that loops back (loop-carried). On the first
#                iteration the key is absent, so raw LangGraph KeyErrors at
#                runtime. pttai's back-edge-aware availability fixpoint (issue
#                #34) rejects it at build.  (pttai-only)
# ===========================================================================
class _SummaryState(AgenticState):
    summary: str                                      # produced only inside the loop


class _TallyState(AgenticState):
    tally: str                                        # produced only inside the loop


def _crbw1_pttai():
    llm = get_llm()
    gen = AgentNode(name="gen", llm=llm, node_prompt="generate",
                    reads=["messages", "summary"])            # loop-carried read
    writer = AgentNode(name="writer", llm=llm, node_prompt="summarize",
                       output_field="summary")                # sole producer, DOWNSTREAM
    gate = ConditionNode(name="gate", condition=lambda s: "refine",
                         choices=["refine", "accept"])
    fin = AgentNode(name="fin", llm=llm, node_prompt="done")
    gen > writer > gate
    gate["refine"] > gen                                      # back-edge closes the cycle
    gate["accept"] > fin
    AgenticGraph(state=_SummaryState, start_node=gen, end_nodes={fin})


def _crbw1_lg():
    llm, counter = counting_llm()

    class S(MessagesState):
        summary: str

    def build():
        b = StateGraph(S)
        b.add_node("gen", lambda s: {"messages": [llm.invoke(
            [SystemMessage(f"generate {s['summary']}")] + s["messages"])]})  # KeyError iter 1
        b.add_node("writer", lambda s: {"summary": "a summary"})
        b.add_node("fin", step(llm, "done"))
        b.add_edge(START, "gen")
        b.add_edge("gen", "writer")
        b.add_conditional_edges("writer", lambda s: "refine",
                                {"refine": "gen", "accept": "fin"})  # loops back to gen
        b.add_edge("fin", END)
        return b

    out = run_langgraph(build, HELLO)
    out["wasted_calls"] = counter[0]
    return out


def _crbw2_pttai():
    llm = get_llm()
    start = AgentNode(name="start", llm=llm, node_prompt="start")
    loop = AgentNode(name="loop", llm=llm, node_prompt="iterate",
                     reads=["messages", "tally"])             # loop-carried read
    acc = AgentNode(name="acc", llm=llm, node_prompt="accumulate",
                    output_field="tally")                     # sole producer, DOWNSTREAM
    gate = ConditionNode(name="gate", condition=lambda s: "again",
                         choices=["again", "done"])
    fin = AgentNode(name="fin", llm=llm, node_prompt="done")
    start > loop > acc > gate
    gate["again"] > loop                                      # back-edge closes the cycle
    gate["done"] > fin
    AgenticGraph(state=_TallyState, start_node=start, end_nodes={fin})


def _crbw2_lg():
    llm, counter = counting_llm()

    class S(MessagesState):
        tally: str

    def build():
        b = StateGraph(S)
        b.add_node("start", step(llm, "start"))
        b.add_node("loop", lambda s: {"messages": [llm.invoke(
            [SystemMessage(f"iterate {s['tally']}")] + s["messages"])]})  # KeyError iter 1
        b.add_node("acc", lambda s: {"tally": "1"})
        b.add_node("fin", step(llm, "done"))
        b.add_edge(START, "start")
        b.add_edge("start", "loop")
        b.add_edge("loop", "acc")
        b.add_conditional_edges("acc", lambda s: "again",
                                {"again": "loop", "done": "fin"})  # loops back to loop
        b.add_edge("fin", END)
        return b

    out = run_langgraph(build, HELLO)
    out["wasted_calls"] = counter[0]
    return out


# ===========================================================================
# BUG CLASS 2 — dangling decision/condition choice (a choice with no .child)
#               (pttai-only)
# ===========================================================================
def _dc1_pttai():
    llm = get_llm()
    start = AgentNode(name="start", llm=llm, node_prompt="start")
    d = DecisionNode(name="d", llm=llm, node_prompt="pick", choices=["a", "b"])
    h = AgentNode(name="h", llm=llm, node_prompt="handle")
    start > d
    d["a"] > h                                        # choice "b" never wired
    AgenticGraph(start_node=start, end_nodes={h})


def _router_lg(unwired_first: List[str], wired_paths: dict):
    """A raw-LangGraph structured-output router whose path map omits a branch.
    The fake ``with_structured_output`` picks the FIRST Literal value, so we
    order the unwired choice first -> it deterministically hits the omitted
    branch -> runtime error after 1 model call."""
    llm, counter = counting_llm()

    class Route(BaseModel):
        choice: Literal[tuple(unwired_first)]  # type: ignore[valid-type]

    router = llm.with_structured_output(Route)

    def build():
        class S(MessagesState):
            _route: str

        b = StateGraph(S)
        b.add_node("d", lambda s: {"_route": router.invoke(
            [SystemMessage("pick")] + s["messages"]).choice})
        for name, prompt in wired_paths.items():
            b.add_node(name, step(llm, prompt))
            b.add_edge(name, END)
        b.add_edge(START, "d")
        b.add_conditional_edges("d", lambda s: s["_route"],
                                {c: c for c in wired_paths})  # unwired choice omitted
        return b

    out = run_langgraph(build, HELLO)
    out["wasted_calls"] = counter[0]
    return out


def _dc1_lg():
    return _router_lg(["b", "a"], {"a": "handle"})


def _dc2_pttai():
    llm = get_llm()
    start = AgentNode(name="start", llm=llm, node_prompt="start")
    d = DecisionNode(name="d", llm=llm, node_prompt="pick", choices=["low", "med", "high"])
    lo = AgentNode(name="lo", llm=llm, node_prompt="low")
    hi = AgentNode(name="hi", llm=llm, node_prompt="high")
    start > d
    d["low"] > lo
    d["high"] > hi                                    # "med" never wired
    AgenticGraph(start_node=start, end_nodes={lo, hi})


def _dc2_lg():
    return _router_lg(["med", "low", "high"], {"low": "low", "high": "high"})


def _dc3_pttai():
    llm = get_llm()
    start = AgentNode(name="start", llm=llm, node_prompt="start")
    g = ConditionNode(name="g", condition=lambda s: "fail", choices=["pass", "fail"])
    h = AgentNode(name="h", llm=llm, node_prompt="handle")
    start > g
    g["pass"] > h                                     # "fail" never wired
    AgenticGraph(start_node=start, end_nodes={h})


def _dc3_lg():
    llm, counter = counting_llm()

    def build():
        b = StateGraph(MessagesState)
        b.add_node("start", step(llm, "start"))
        b.add_node("h", step(llm, "handle"))
        b.add_edge(START, "start")
        b.add_conditional_edges("start", lambda s: "fail", {"pass": "h"})  # "fail" omitted
        b.add_edge("h", END)
        return b

    out = run_langgraph(build, HELLO)
    out["wasted_calls"] = counter[0]
    return out


# ===========================================================================
# BUG CLASS 3 — non-end node whose .child is None (a dead-end)
#   NOT a LangGraph bug: a node with no outgoing edge is a LEGAL implicit
#   terminal in LangGraph (verified empirically — compile() does not raise and
#   invoke() returns without error). pttai rejects it only because its DSL
#   REQUIRES every terminal to be declared in end_nodes. So this is pttai DSL
#   strictness, category="dsl-strictness" — excluded from the pttai-only
#   differentiator headline and the wasted-call total.
# ===========================================================================
def _de1_pttai():
    llm = get_llm()
    a = AgentNode(name="a", llm=llm, node_prompt="one")
    b = AgentNode(name="b", llm=llm, node_prompt="two")
    a > b                                             # forgot to make b an end node
    AgenticGraph(start_node=a, end_nodes=set())


def _de1_lg():
    llm, counter = counting_llm()

    def build():
        b = StateGraph(MessagesState)
        b.add_node("a", step(llm, "one"))
        b.add_node("b", step(llm, "two"))
        b.add_edge(START, "a")
        b.add_edge("a", "b")                          # "b" dead-ends: no edge to END
        return b

    out = run_langgraph(build, HELLO)
    out["wasted_calls"] = counter[0]
    return out


def _de2_pttai():
    llm = get_llm()
    a = AgentNode(name="a", llm=llm, node_prompt="one")
    b = AgentNode(name="b", llm=llm, node_prompt="two")
    c = AgentNode(name="c", llm=llm, node_prompt="three")
    a > b > c
    AgenticGraph(start_node=a, end_nodes=set())        # c dead-ends


def _de2_lg():
    llm, counter = counting_llm()

    def build():
        b = StateGraph(MessagesState)
        b.add_node("a", step(llm, "one"))
        b.add_node("b", step(llm, "two"))
        b.add_node("c", step(llm, "three"))
        b.add_edge(START, "a")
        b.add_edge("a", "b")
        b.add_edge("b", "c")                          # "c" dead-ends
        return b

    out = run_langgraph(build, HELLO)
    out["wasted_calls"] = counter[0]
    return out


def _de3_pttai():
    llm = get_llm()
    a = AgentNode(name="a", llm=llm, node_prompt="one")
    b = AgentNode(name="b", llm=llm, node_prompt="two")
    c = AgentNode(name="c", llm=llm, node_prompt="three")
    a > [b, c]                                        # c is not an end node
    AgenticGraph(start_node=a, end_nodes={b})


def _de3_lg():
    llm, counter = counting_llm()

    def build():
        b = StateGraph(MessagesState)
        b.add_node("a", step(llm, "one"))
        b.add_node("b", step(llm, "two"))
        b.add_node("c", step(llm, "three"))
        b.add_edge(START, "a")
        b.add_edge("a", "b")
        b.add_edge("a", "c")
        b.add_edge("b", END)                          # "c" dead-ends
        return b

    out = run_langgraph(build, HELLO)
    out["wasted_calls"] = counter[0]
    return out


# ===========================================================================
# BUG CLASS 4 — duplicate node names  (caught by BOTH frameworks at build)
# ===========================================================================
def _dup1_pttai():
    llm = get_llm()
    a = AgentNode(name="dup", llm=llm, node_prompt="one")
    b = AgentNode(name="dup", llm=llm, node_prompt="two")
    a > b
    AgenticGraph(start_node=a, end_nodes={b})


def _dup1_lg():
    llm, counter = counting_llm()

    def build():
        b = StateGraph(MessagesState)
        b.add_node("dup", step(llm, "one"))
        b.add_node("dup", step(llm, "two"))           # ValueError at build
        b.add_edge(START, "dup")
        b.add_edge("dup", END)
        return b

    out = run_langgraph(build, HELLO)
    out["wasted_calls"] = counter[0]
    return out


def _dup2_pttai():
    llm = get_llm()
    a = AgentNode(name="worker", llm=llm, node_prompt="one")
    b = AgentNode(name="mid", llm=llm, node_prompt="two")
    c = AgentNode(name="worker", llm=llm, node_prompt="three")
    a > b > c
    AgenticGraph(start_node=a, end_nodes={c})


def _dup2_lg():
    llm, counter = counting_llm()

    def build():
        b = StateGraph(MessagesState)
        b.add_node("worker", step(llm, "one"))
        b.add_node("mid", step(llm, "two"))
        b.add_node("worker", step(llm, "three"))      # ValueError at build
        b.add_edge(START, "worker")
        b.add_edge("worker", "mid")
        b.add_edge("mid", END)
        return b

    out = run_langgraph(build, HELLO)
    out["wasted_calls"] = counter[0]
    return out


# ===========================================================================
# BUG CLASS 5 — concurrent write to a reducer-less key across parallel branches
#               (pttai-only)
# ===========================================================================
class _ResultState(AgenticState):
    result: str                                       # plain: no reducer


def _cw1_pttai():
    llm = get_llm()
    a = AgentNode(name="a", llm=llm, node_prompt="start")
    b = AgentNode(name="b", llm=llm, node_prompt="b", output_field="result")
    c = AgentNode(name="c", llm=llm, node_prompt="c", output_field="result")
    j = AgentNode(name="j", llm=llm, node_prompt="join", reads=["result"])
    (a > [b, c]) > j                                  # b and c both write `result`
    AgenticGraph(state=_ResultState, start_node=a, end_nodes={j})


def _cw_lg(n_writers: int):
    llm, counter = counting_llm()

    class S(MessagesState):
        result: str                                   # plain: no reducer

    def build():
        b = StateGraph(S)
        b.add_node("a", step(llm, "start"))
        writers = [f"w{i}" for i in range(n_writers)]
        for w in writers:
            b.add_node(w, lambda s, w=w: {"result": w})
            b.add_edge("a", w)
            b.add_edge(w, "j")
        b.add_node("j", step(llm, "join"))
        b.add_edge(START, "a")
        b.add_edge("j", END)
        return b

    out = run_langgraph(build, HELLO)
    out["wasted_calls"] = counter[0]
    return out


def _cw1_lg():
    return _cw_lg(2)


def _cw2_pttai():
    llm = get_llm()
    a = AgentNode(name="a", llm=llm, node_prompt="start")
    b = AgentNode(name="b", llm=llm, node_prompt="b", output_field="result")
    c = AgentNode(name="c", llm=llm, node_prompt="c", output_field="result")
    d = AgentNode(name="d", llm=llm, node_prompt="d", output_field="result")
    j = AgentNode(name="j", llm=llm, node_prompt="join", reads=["result"])
    (a > [b, c, d]) > j
    AgenticGraph(state=_ResultState, start_node=a, end_nodes={j})


def _cw2_lg():
    return _cw_lg(3)


# ===========================================================================
# BUG CLASS 6 — prompt placeholder with no matching scalar read  (pttai-only)
#               (a guaranteed runtime KeyError; the closest thing to a
#                "schema mismatch" the validator actually detects)
# ===========================================================================
def _pp1_pttai():
    llm = get_llm()
    prod = AgentNode(name="prod", llm=llm, node_prompt="produce", writes={"topic": str})
    reader = AgentNode(name="reader", llm=llm, node_prompt="Write about {subject}",
                       reads=["topic"])                # {subject} is not a declared read
    prod > reader
    AgenticGraph(start_node=prod, end_nodes={reader})


def _pp1_lg():
    llm, counter = counting_llm()

    class S(MessagesState):
        topic: str

    def build():
        b = StateGraph(S)
        b.add_node("prod", lambda s: {"topic": "cats"})
        b.add_node("reader", lambda s: {"messages": [llm.invoke(
            [SystemMessage("Write about {subject}".format(**{"topic": s["topic"]}))]
            + s["messages"])]})                       # KeyError: 'subject'
        b.add_edge(START, "prod")
        b.add_edge("prod", "reader")
        b.add_edge("reader", END)
        return b

    out = run_langgraph(build, HELLO)
    out["wasted_calls"] = counter[0]
    return out


def _pp2_pttai():
    llm = get_llm()
    prod = AgentNode(name="prod", llm=llm, node_prompt="produce", writes={"rubric": str})
    scorer = AgentNode(name="scorer", llm=llm, node_prompt="Score against {criteria}",
                       reads=["rubric"])               # {criteria} is not a declared read
    prod > scorer
    AgenticGraph(start_node=prod, end_nodes={scorer})


def _pp2_lg():
    llm, counter = counting_llm()

    class S(MessagesState):
        rubric: str

    def build():
        b = StateGraph(S)
        b.add_node("prod", lambda s: {"rubric": "clarity"})
        b.add_node("scorer", lambda s: {"messages": [llm.invoke(
            [SystemMessage("Score against {criteria}".format(**{"rubric": s["rubric"]}))]
            + s["messages"])]})                       # KeyError: 'criteria'
        b.add_edge(START, "prod")
        b.add_edge("prod", "scorer")
        b.add_edge("scorer", END)
        return b

    out = run_langgraph(build, HELLO)
    out["wasted_calls"] = counter[0]
    return out


# ===========================================================================
# BUGGY corpus
# ===========================================================================
BUGGY: List[Item] = [
    Item("rbw-1", "read-before-write", "buggy", "differentiator", _rbw1_pttai, _rbw1_lg),
    Item("rbw-2", "read-before-write", "buggy", "differentiator", _rbw2_pttai, _rbw2_lg),
    Item("rbw-3", "read-before-write", "buggy", "differentiator", _rbw3_pttai, _rbw3_lg),
    Item("crbw-1", "cyclic-read-before-write", "buggy", "differentiator", _crbw1_pttai, _crbw1_lg),
    Item("crbw-2", "cyclic-read-before-write", "buggy", "differentiator", _crbw2_pttai, _crbw2_lg),
    Item("dc-1", "dangling-choice", "buggy", "differentiator", _dc1_pttai, _dc1_lg),
    Item("dc-2", "dangling-choice", "buggy", "differentiator", _dc2_pttai, _dc2_lg),
    Item("dc-3", "dangling-choice", "buggy", "differentiator", _dc3_pttai, _dc3_lg),
    # dead-end: pttai DSL strictness, NOT a LangGraph bug (legal implicit terminal).
    Item("de-1", "dead-end-node", "buggy", "dsl-strictness", _de1_pttai, _de1_lg),
    Item("de-2", "dead-end-node", "buggy", "dsl-strictness", _de2_pttai, _de2_lg),
    Item("de-3", "dead-end-node", "buggy", "dsl-strictness", _de3_pttai, _de3_lg),
    # duplicate names: caught at build by BOTH frameworks — not a differentiator.
    Item("dup-1", "duplicate-node-names", "buggy", "both", _dup1_pttai, _dup1_lg),
    Item("dup-2", "duplicate-node-names", "buggy", "both", _dup2_pttai, _dup2_lg),
    Item("cw-1", "concurrent-write-no-reducer", "buggy", "differentiator", _cw1_pttai, _cw1_lg),
    Item("cw-2", "concurrent-write-no-reducer", "buggy", "differentiator", _cw2_pttai, _cw2_lg),
    Item("pp-1", "prompt-placeholder-mismatch", "buggy", "differentiator", _pp1_pttai, _pp1_lg),
    Item("pp-2", "prompt-placeholder-mismatch", "buggy", "differentiator", _pp2_pttai, _pp2_lg),
]


# ===========================================================================
# CLEAN corpus — the real working pipelines from examples/ (false-positive set)
# ===========================================================================
CLEAN_EXAMPLES = [
    ("architectures", "prompt_chaining"),
    ("architectures", "routing"),
    ("architectures", "react_agent"),
    ("architectures", "parallelization"),
    ("architectures", "evaluator_optimizer"),
    ("architectures", "orchestrator_workers"),
    ("architectures", "reflection"),
    ("architectures", "supervisor"),
    ("basics", "01_single_agent"),
    ("basics", "02_tools"),
    ("basics", "03_sequential"),
    ("basics", "04_decision_routing"),
    ("basics", "05_condition_routing"),
    ("basics", "06_parallel_fanout"),
    ("basics", "07_map_reduce"),
    ("basics", "08_typed_io"),
    ("basics", "10_graph_composition"),
    ("basics", "11_node_policies"),
    ("basics", "12_validation_summary"),
]


def _clean_builder(pkg, mod):
    """Return a no-arg callable that builds+runs the example's pttai graph. A
    GraphValidationError / build ValueError raised here is a FALSE POSITIVE."""
    import importlib
    import sys

    def build():
        m = importlib.import_module(f"{pkg}.{mod}")
        # Silence example console output at the fd level (some examples call
        # graph.summary(file=sys.stdout) whose default is bound at import, so a
        # plain redirect_stdout misses it). Build+validate happens regardless.
        devnull = os.open(os.devnull, os.O_WRONLY)
        saved = os.dup(1)
        try:
            os.dup2(devnull, 1)
            m.pttai_version()                         # builds (validates) + runs offline
        finally:
            sys.stdout.flush()                        # flush to devnull, not to the restored fd
            os.dup2(saved, 1)
            os.close(saved)
            os.close(devnull)

    return build


CLEAN: List[Item] = [
    Item(f"clean-{pkg}-{mod}", f"valid/{pkg}", "clean", "clean", _clean_builder(pkg, mod))
    for pkg, mod in CLEAN_EXAMPLES
]


CORPUS: List[Item] = BUGGY + CLEAN
