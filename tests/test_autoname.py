"""Node name inference (issue #11): `reviewer = AgentNode()` auto-names the node
"reviewer" by reading the assignment target at the call site. Inference reads the
CALLER's source, so the constructions below are written as plain assignments —
that is exactly what's being tested."""

import io

import pytest

from nae import AgenticGraph, AgentNode


def test_simple_inference(t):
    # 1. `reviewer = AgentNode(...)` -> name "reviewer".
    reviewer = AgentNode(llm=t.FakeLLM(responses=[t.ai("x")]))
    AgenticGraph(start_node=reviewer, end_nodes={reviewer})
    assert reviewer.name == "reviewer"


def test_multiline_constructor_infers(t):
    # 2. A constructor spanning several lines still infers the LHS name.
    agent = AgentNode(
        llm=t.FakeLLM(responses=[t.ai("y")]),
        node_prompt="hi",
    )
    AgenticGraph(start_node=agent, end_nodes={agent})
    assert agent.name == "agent"


def test_attribute_assignment_infers_attr_name(t):
    # 3. `self.x = AgentNode(...)` -> attribute name "x".
    class Holder:
        def __init__(self):
            self.x = AgentNode(llm=t.FakeLLM(responses=[t.ai("x")]))

    h = Holder()
    AgenticGraph(start_node=h.x, end_nodes={h.x})
    assert h.x.name == "x"


def test_tuple_unpack_falls_back(t):
    # 4. Tuple-unpack can't infer a single LHS name -> both fall back to numbered
    # type names by traversal order.
    a, b = AgentNode(llm=t.FakeLLM(responses=[t.ai("a")])), AgentNode(llm=t.FakeLLM(responses=[t.ai("b")]))
    a > b
    AgenticGraph(start_node=a, end_nodes={b})
    assert a.name == "agentnode"
    assert b.name == "agentnode_1"


def test_same_inferred_name_suffixed(t):
    # 5. Two nodes that infer the SAME name collide -> reviewer, reviewer_1.
    reviewer = AgentNode(llm=t.FakeLLM(responses=[t.ai("a")]))
    first = reviewer
    reviewer = AgentNode(llm=t.FakeLLM(responses=[t.ai("b")]))
    second = reviewer
    first > second
    AgenticGraph(start_node=first, end_nodes={second})
    assert {first.name, second.name} == {"reviewer", "reviewer_1"}


def test_duplicate_explicit_names_raise(t):
    # 6. Two EXPLICIT duplicate names in one graph still raise.
    a = AgentNode(name="dup", llm=t.FakeLLM(responses=[t.ai("a")]))
    b = AgentNode(name="dup", llm=t.FakeLLM(responses=[t.ai("b")]))
    a > b
    with pytest.raises(ValueError, match="Duplicate node name"):
        AgenticGraph(start_node=a, end_nodes={b})


def test_explicit_positional_name_wins(t):
    # 7. An explicit positional name overrides inference.
    x = AgentNode("custom", llm=t.FakeLLM(responses=[t.ai("z")]))
    AgenticGraph(start_node=x, end_nodes={x})
    assert x.name == "custom"


def test_summary_shows_inferred_names(t):
    # 8. summary() renders the resolved inferred names.
    reviewer = AgentNode(llm=t.FakeLLM(responses=[t.ai("x")]))
    g = AgenticGraph(start_node=reviewer, end_nodes={reviewer})
    buf = io.StringIO()
    g.summary(file=buf)
    assert "reviewer" in buf.getvalue()


def test_explicit_named_graph_regression(t):
    # 9. An existing-style graph with explicit names builds + invokes unchanged.
    a = AgentNode(name="alpha", llm=t.FakeLLM(responses=[t.ai("hello")]))
    out = AgenticGraph(start_node=a, end_nodes={a}).invoke("hi")
    assert a.name == "alpha"
    assert any(m.content == "hello" for m in out["messages"])
