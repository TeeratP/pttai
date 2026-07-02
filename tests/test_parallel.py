"""Parallel fan-out/join: `a > [b, c]` wiring, diamond edges, and defer.

NOTE on syntax: `a > [b, c] > d` is a Python *chained comparison*
(`(a > [b,c]) and ([b,c] > d)`). The join half `[b,c] > d` is `list.__gt__(d)`
-> NotImplemented -> reflected `d.__lt__([b,c])`, which `Node.__lt__` handles to
wire the join. So the clean no-paren one-liner works. The explicit-Branch form
`branch = a > [b, c]; branch > d` (and `(a > [b, c]) > d`) wires identically.
"""

import pytest

from nae import fanout
from nae.graph import AgenticGraph
from nae.nodes import AgentNode, DecisionNode
from nae.state import AgenticState


def _edges(graph):
    return {(e.source, e.target) for e in graph.compiled_graph.get_graph().edges}


def _nodes(graph):
    return set(graph.compiled_graph.get_graph().nodes.keys())


def _agent(t, name, content=None):
    return AgentNode(name=name, llm=t.FakeLLM(responses=[t.ai(content or name)]))


def test_fanout_sets_children(t):
    a = _agent(t, "a")
    b = _agent(t, "b")
    c = _agent(t, "c")
    a > [b, c]
    assert a.children == [b, c]


def test_fanout_edges(t):
    a = _agent(t, "a")
    b = _agent(t, "b")
    c = _agent(t, "c")
    a > [b, c]
    g = AgenticGraph(state=AgenticState, start_node=a, end_nodes={b, c})

    edges = _edges(g)
    assert ("__start__", "a") in edges
    assert ("a", "b") in edges
    assert ("a", "c") in edges
    assert ("b", "__end__") in edges
    assert ("c", "__end__") in edges


def test_chained_oneliner_wires_join(t):
    # The clean no-paren one-liner wires via Node.__lt__ (reflected list>node).
    a = _agent(t, "a")
    b = _agent(t, "b")
    c = _agent(t, "c")
    d = _agent(t, "d")
    a > [b, c] > d
    assert a.children == [b, c]
    assert b.children == [d]
    assert c.children == [d]
    assert d._is_join is True


def test_fanout_helper_wires_join(t):
    # `a > fanout(b, c) > d` wires identically to the `a > [b, c] > d` list form.
    a = _agent(t, "a")
    b = _agent(t, "b")
    c = _agent(t, "c")
    d = _agent(t, "d")
    a > fanout(b, c) > d
    assert a.children == [b, c]
    assert b.children == [d]
    assert c.children == [d]
    assert d._is_join is True

    g_fan = AgenticGraph(state=AgenticState, start_node=a, end_nodes={d})

    # Same topology built via the list form must produce identical edges.
    a2 = _agent(t, "a")
    b2 = _agent(t, "b")
    c2 = _agent(t, "c")
    d2 = _agent(t, "d")
    a2 > [b2, c2] > d2
    g_list = AgenticGraph(state=AgenticState, start_node=a2, end_nodes={d2})

    assert _edges(g_fan) == _edges(g_list)


def test_diamond_join_edges(t):
    a = _agent(t, "a")
    b = _agent(t, "b")
    c = _agent(t, "c")
    d = _agent(t, "d")
    a > [b, c] > d  # clean no-paren one-liner
    g = AgenticGraph(state=AgenticState, start_node=a, end_nodes={d})

    edges = _edges(g)
    assert ("b", "d") in edges  # fan-in fix: both branches reach the join
    assert ("c", "d") in edges
    assert ("d", "__end__") in edges


def test_diamond_join_runs_once(t):
    # Explicit-Branch form wires identically to the no-paren one-liner.
    a = _agent(t, "a", "a-out")
    b = _agent(t, "b", "b-out")
    c = _agent(t, "c", "c-out")
    d = _agent(t, "d", "d-out")
    branch = a > [b, c]
    branch > d
    g = AgenticGraph(state=AgenticState, start_node=a, end_nodes={d})

    out = g.invoke({"messages": ["hi"], "log": []})
    log = out["log"]
    assert sum(1 for line in log if line.startswith("d:")) == 1  # join ran once
    assert "b:b-out" in log and "c:c-out" in log  # both branches contributed


def test_unbalanced_join_runs_once(t):
    # a fans out to a long arm (b1->b2->d) and a short arm (c->d). Without
    # defer, the short arm reaches d in an earlier super-step and d fires twice.
    a = _agent(t, "a", "a-out")
    b1 = _agent(t, "b1", "b1-out")
    b2 = _agent(t, "b2", "b2-out")
    c = _agent(t, "c", "c-out")
    d = _agent(t, "d", "d-out")
    a > [b1, c]
    b1 > b2 > d
    c > d
    g = AgenticGraph(state=AgenticState, start_node=a, end_nodes={d})

    assert d._defer is True  # in-degree 2 -> deferred

    out = g.invoke({"messages": ["hi"], "log": []})
    assert sum(1 for line in out["log"] if line.startswith("d:")) == 1


def test_multinode_branch_join(t):
    a = _agent(t, "a", "a-out")
    b = _agent(t, "b", "b-out")
    c = _agent(t, "c", "c-out")
    b2 = _agent(t, "b2", "b2-out")
    c2 = _agent(t, "c2", "c2-out")
    d = _agent(t, "d", "d-out")
    a > [b, c]
    b > b2
    c > c2
    b2 > d
    c2 > d
    g = AgenticGraph(state=AgenticState, start_node=a, end_nodes={d})

    edges = _edges(g)
    assert ("b2", "d") in edges
    assert ("c2", "d") in edges

    out = g.invoke({"messages": ["hi"], "log": []})
    assert sum(1 for line in out["log"] if line.startswith("d:")) == 1


def test_decision_merge_regression(t):
    # Two choices of a DecisionNode route to the SAME merge node: no crash,
    # decision edges are conditional (not static joins), merge still reachable.
    d = DecisionNode(name="d", llm=t.FakeLLM(structured_value="x"),
                     node_prompt="p", choices=["x", "y"])
    merge = _agent(t, "merge")
    d["x"] > merge
    d["y"] > merge
    g = AgenticGraph(state=AgenticState, start_node=d, end_nodes={merge})

    assert "merge" in _nodes(g)
    assert ("merge", "__end__") in _edges(g)
    # Reached only via conditional decision edges -> not flagged as a static join.
    assert getattr(merge, "_defer", False) is False


def test_fanout_chain_branches(t):
    # Multi-node branch CHAINS inline. `b>c>d>e` evaluates to its tail `e`, but
    # head-tracking fans `a` out to the chain HEADS (b, f), not the tails.
    a = _agent(t, "a", "a-out")
    b = _agent(t, "b", "b-out")
    c = _agent(t, "c", "c-out")
    d = _agent(t, "d", "d-out")
    e = _agent(t, "e", "e-out")
    f = _agent(t, "f", "f-out")
    g = _agent(t, "g", "g-out")
    h = _agent(t, "h", "h-out")
    a > fanout(b > c > d > e, f > g) > h

    assert a.children == [b, f]  # fans out to HEADS, not tails (e, g)
    assert h._is_join is True

    graph = AgenticGraph(state=AgenticState, start_node=a, end_nodes={h})
    assert h._defer is True  # in-degree 2 (from e and g)

    edges = _edges(graph)
    expected = {
        ("__start__", "a"),
        ("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "h"),
        ("a", "f"), ("f", "g"), ("g", "h"),
        ("h", "__end__"),
    }
    assert edges == expected

    out = graph.invoke({"messages": ["hi"], "log": []})
    log = out["log"]
    assert sum(1 for line in log if line.startswith("h:")) == 1  # join ran once
    for name in ("b", "c", "d", "e", "f", "g"):
        assert f"{name}:{name}-out" in log  # every chain node executed


def test_fanout_chain_list_form(t):
    # Same graph via the bracket-list form must produce identical edges.
    a = _agent(t, "a")
    b = _agent(t, "b")
    c = _agent(t, "c")
    d = _agent(t, "d")
    e = _agent(t, "e")
    f = _agent(t, "f")
    g = _agent(t, "g")
    h = _agent(t, "h")
    a > [b > c > d > e, f > g] > h
    graph_list = AgenticGraph(state=AgenticState, start_node=a, end_nodes={h})

    a2 = _agent(t, "a")
    b2 = _agent(t, "b")
    c2 = _agent(t, "c")
    d2 = _agent(t, "d")
    e2 = _agent(t, "e")
    f2 = _agent(t, "f")
    g2 = _agent(t, "g")
    h2 = _agent(t, "h")
    a2 > fanout(b2 > c2 > d2 > e2, f2 > g2) > h2
    graph_fan = AgenticGraph(state=AgenticState, start_node=a2, end_nodes={h2})

    assert _edges(graph_list) == _edges(graph_fan)


def test_fanout_into_fanout_not_supported(t):
    # Known ceiling: `[..] > [..]` is Python's element-wise list richcompare,
    # NOT our join wiring. It compares the first differing pair via `b > d`
    # (wiring only b->d and returning that node), leaves the rest unwired, and
    # never marks a join. So no proper fan-out->fan-out: put a node between them.
    b = _agent(t, "b")
    c = _agent(t, "c")
    d = _agent(t, "d")
    e = _agent(t, "e")
    [b, c] > [d, e]
    assert c.children == []                     # only the first pair touched
    assert getattr(d, "_is_join", False) is False  # never a real join


# --- map-reduce (Spread) -------------------------------------------------

def test_map_wiring(t):
    # `a > worker.map("items") > collector` registers the worker, fans out from
    # `a` via a conditional (Send) edge, and joins into a deferred collector.
    a = _agent(t, "a")
    worker = _agent(t, "worker")
    collector = _agent(t, "collector")
    a > worker.map("items") > collector
    g = AgenticGraph(state=AgenticState, start_node=a, end_nodes={collector})

    nodes = _nodes(g)
    assert "worker" in nodes and "collector" in nodes
    assert ("worker", "collector") in _edges(g)  # static worker->collector edge
    assert ("collector", "__end__") in _edges(g)
    assert collector._is_join is True
    assert collector._defer is True
    assert not getattr(worker, "_defer", False)  # worker fans out, never deferred


def test_map_runs_per_item(t):
    # Core proof: worker runs once per item IN PARALLEL, collector runs once after.
    class MapState(AgenticState):
        items: list  # the field map() spreads over (must be a real state channel)

    worker_llm = t.FakeLLM(echo=True)  # distinguishable reply per item
    a = _agent(t, "a", "a-out")
    worker = AgentNode(name="worker", llm=worker_llm)
    collector = _agent(t, "collector", "collector-out")
    a > worker.map("items") > collector
    g = AgenticGraph(state=MapState, start_node=a, end_nodes={collector})

    out = g.invoke({"messages": ["hi"], "log": [], "items": ["x", "y", "z"]})

    # worker invoked exactly 3 times (once per item)
    assert worker_llm.invoke_count == 3
    # collector ran exactly once, AFTER all workers
    assert sum(1 for line in out["log"] if line.startswith("collector:")) == 1
    # all three worker replies accumulated in messages before the collector
    contents = [m.content for m in out["messages"]]
    assert "reply:x" in contents
    assert "reply:y" in contents
    assert "reply:z" in contents


def test_map_sole_child_guard(t):
    # The Spread must be its predecessor's ONLY child — it can't share a source.
    a = _agent(t, "a")
    worker = _agent(t, "worker")
    collector = _agent(t, "collector")
    other = _agent(t, "other")
    spread = worker.map("items")
    spread > collector
    a.children = [spread, other]  # illegal: spread shares the source with `other`
    with pytest.raises(ValueError, match="only child"):
        AgenticGraph(state=AgenticState, start_node=a, end_nodes={collector, other})


def test_map_as_choice_child_raises(t):
    # A decision choice cannot route directly into a `.map`/fan-out.
    d = DecisionNode(name="d", llm=t.FakeLLM(structured_value="pos"),
                     node_prompt="p", choices=["pos", "neg"])
    worker = _agent(t, "worker")
    with pytest.raises(ValueError, match="wrap it in an AgenticGraph"):
        d["pos"] > worker.map("items")
