"""Regression tests for the build-time analysis / robustness fixes (issue #12).

All offline — these exercise the static dataflow + graph-build checks, so the
nodes' LLMs are never invoked (a bare FakeLLM suffices). Covers:

  1. convergence cap — a key produced early, read far down a long chain.
  2. mid-loop mutation — a valid diamond with a plain key written on a branch
     AND the join (no spurious concurrency error), plus a real-concurrency guard.
  3. unreachable detection — an orphan hung off an end node warns.
  4. map-off-router — a clear, actionable error message.
  5. duplicate default subgraph name — two unnamed subgraphs get distinct names.
"""

import pytest

from nae import AgentNode, AgenticGraph, fanout
from nae.nodes import ConditionNode
from nae.validation import GraphValidationError


# --- Fix 1: convergence cap -------------------------------------------------

def test_deep_chain_key_produced_early_read_deep(t):
    """20-node chain: `extract` writes `topic`, a deep `use` node reads it.
    More nodes than distinct state keys, so the old fixed iteration bound
    under-converged and falsely reported `topic` as never produced."""
    llm = t.FakeLLM()
    extract = AgentNode(name="extract", llm=llm, writes={"topic": str})
    chain = [extract]
    for i in range(20):
        n = AgentNode(name=f"step{i}", llm=llm)
        chain[-1] > n
        chain.append(n)
    use = AgentNode(name="use", llm=llm, node_prompt="{topic}", reads=["topic"])
    chain[-1] > use
    g = AgenticGraph(start_node=extract, end_nodes={use})  # must not raise
    assert g.validate().ok


# --- Fix 2: mid-loop mutation in _concurrency_pairs -------------------------

def test_diamond_branch_and_join_share_plain_key_builds(t):
    """Diamond `a > [b, c] > d` where `b` (a branch) and `d` (the join) both
    write a plain key `summary`. `d` is strictly downstream of `b`, so they are
    NOT concurrent — this must build on every hash seed."""
    llm = t.FakeLLM()
    a = AgentNode(name="a", llm=llm)
    b = AgentNode(name="b", llm=llm, writes={"summary": str})
    c = AgentNode(name="c", llm=llm)
    d = AgentNode(name="d", llm=llm, writes={"summary": str})
    a > fanout(b, c) > d
    g = AgenticGraph(start_node=a, end_nodes={d})  # must not raise
    assert g.validate().ok


def test_diamond_two_parallel_branches_plain_key_still_errors(t):
    """Guard against over-suppression: `b` and `c` ARE parallel branches; both
    writing a plain (non-reduced) key is a genuine concurrent-write conflict and
    must still be a hard error."""
    llm = t.FakeLLM()
    a = AgentNode(name="a", llm=llm)
    b = AgentNode(name="b", llm=llm, writes={"summary": str})
    c = AgentNode(name="c", llm=llm, writes={"summary": str})
    d = AgentNode(name="d", llm=llm)
    a > fanout(b, c) > d
    with pytest.raises(GraphValidationError, match="no reducer"):
        AgenticGraph(start_node=a, end_nodes={d})


# --- Fix 3: unreachable detection -------------------------------------------

def test_orphan_off_end_node_warns(t):
    """A node hung off an end node's children is never built (the end node
    terminates the graph), so it is unreachable — warn (never raise)."""
    llm = t.FakeLLM()
    a = AgentNode(name="a", llm=llm)
    orphan = AgentNode(name="orphan", llm=llm)
    a > orphan          # a is the end node, so `orphan` is never reached
    g = AgenticGraph(start_node=a, end_nodes={a})
    report = g.validate()
    assert report.ok    # warning only, no error
    msgs = [str(w) for w in report.warnings]
    assert any("orphan" in m and "unreachable from the start node" in m
               for m in msgs), msgs


# --- Fix 4: map-off-router clear message ------------------------------------

def test_map_off_router_raises_clear_message(t):
    """Mapping directly off a decision/condition branch is unsupported; the
    Spread guard must raise an actionable message naming the passthrough
    workaround. (Unreachable via the public wiring API — `Choice.__gt__` blocks
    `choice > spread` first — so drive the build path directly.)"""
    llm = t.FakeLLM()
    a = AgentNode(name="a", llm=llm)
    g = AgenticGraph(start_node=a, end_nodes={a})

    router = ConditionNode(name="router", condition=lambda s: "x", choices=["x"])
    worker = AgentNode(name="w", llm=llm)
    collector = AgentNode(name="coll", llm=llm)
    spread = worker.map("items")
    spread > collector

    with pytest.raises(ValueError, match="insert an intermediate node"):
        g._build_graph(spread, router)


# --- Fix 5: duplicate default subgraph name ---------------------------------

def test_two_unnamed_subgraphs_get_distinct_names(t):
    """Two subgraphs both defaulting to name 'graph' must auto-suffix (graph,
    graph_1) rather than collide as duplicate explicit names."""
    a = AgentNode(name="a", llm=t.FakeLLM())
    g0 = AgenticGraph(start_node=a, end_nodes={a})
    b = AgentNode(name="b", llm=t.FakeLLM())
    g1 = AgenticGraph(start_node=b, end_nodes={b})

    assert g0.name == "graph" and g1.name == "graph"  # each standalone -> "graph"

    g0 > g1
    parent = AgenticGraph(start_node=g0, end_nodes={g1})  # must not raise
    assert {g0.name, g1.name} == {"graph", "graph_1"}


def test_named_subgraph_never_suffixed(t):
    """An explicitly-named subgraph keeps its name (no suffix)."""
    a = AgentNode(name="a", llm=t.FakeLLM())
    inner = AgenticGraph(start_node=a, end_nodes={a}, name="inner")
    assert inner.name == "inner"
    assert inner._auto_name is False
