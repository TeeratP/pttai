"""Graph construction from `>` wiring: edges, duplicate-name guard, composition."""

import pytest

from nae.graph import AgenticGraph
from nae.nodes import AgentNode, DecisionNode
from nae.state import AgenticState


def _edges(graph):
    return {(e.source, e.target) for e in graph.compiled_graph.get_graph().edges}


def _nodes(graph):
    return set(graph.compiled_graph.get_graph().nodes.keys())


def test_linear_chain_edges(t):
    a = AgentNode(name="a", llm=t.FakeLLM())
    b = AgentNode(name="b", llm=t.FakeLLM())
    c = AgentNode(name="c", llm=t.FakeLLM())
    a > b > c
    g = AgenticGraph(state=AgenticState, start_node=a, end_nodes={c})

    edges = _edges(g)
    assert ("__start__", "a") in edges
    assert ("a", "b") in edges
    assert ("b", "c") in edges
    assert ("c", "__end__") in edges


def test_decision_branches_reach_both_handlers(t):
    d = DecisionNode(name="clf", llm=t.FakeLLM(structured_value="positive"),
                     node_prompt="p", choices=["positive", "negative"])
    pos = AgentNode(name="pos", llm=t.FakeLLM())
    neg = AgentNode(name="neg", llm=t.FakeLLM())
    d["positive"] > pos
    d["negative"] > neg
    g = AgenticGraph(state=AgenticState, start_node=d, end_nodes={pos, neg})

    nodes = _nodes(g)
    assert {"clf", "pos", "neg"}.issubset(nodes)


def test_duplicate_name_raises(t):
    a = AgentNode(name="dup", llm=t.FakeLLM())
    b = AgentNode(name="dup", llm=t.FakeLLM())
    a > b
    with pytest.raises(ValueError, match="Duplicate node name"):
        AgenticGraph(state=AgenticState, start_node=a, end_nodes={b})


def test_same_node_revisited_is_ok(t):
    # Diamond: both decision branches converge on the same `merge` object.
    d = DecisionNode(name="d", llm=t.FakeLLM(structured_value="x"),
                     node_prompt="p", choices=["x", "y"])
    merge = AgentNode(name="merge", llm=t.FakeLLM())
    d["x"] > merge
    d["y"] > merge
    g = AgenticGraph(state=AgenticState, start_node=d, end_nodes={merge})  # must not raise
    assert "merge" in _nodes(g)


def test_graph_as_node_composition(t):
    a1 = AgentNode(name="a1", llm=t.FakeLLM())
    a2 = AgentNode(name="a2", llm=t.FakeLLM())
    a1 > a2
    g0 = AgenticGraph(name="g0", state=AgenticState, start_node=a1, end_nodes=a2)

    b1 = AgentNode(name="b1", llm=t.FakeLLM())
    b2 = AgentNode(name="b2", llm=t.FakeLLM())
    b1 > b2
    g1 = AgenticGraph(name="g1", state=AgenticState, start_node=b1, end_nodes=b2)

    g0 > g1
    big = AgenticGraph(name="big", state=AgenticState, start_node=g0, end_nodes=g1)

    nodes = _nodes(big)
    assert {"g0", "g1"}.issubset(nodes)
