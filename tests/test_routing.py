"""DecisionNode writes to its `decision_{name}` field (not messages) and routes off it."""

import pytest
from langchain_core.messages import HumanMessage

from nae.nodes import AgentNode, DecisionNode


def _decision(t, value="positive"):
    return DecisionNode(name="clf", llm=t.FakeLLM(structured_value=value),
                        node_prompt="p", choices=["positive", "negative"])


def test_returns_decision_field_not_message(t):
    d = _decision(t, "positive")
    delta = d({"messages": [HumanMessage(content="hi")], "log": []})
    assert delta == {"decision_clf": "positive", "log": ["clf:positive"]}
    assert "messages" not in delta  # label must not pollute conversation history


def test_route_picks_correct_branch(t):
    d = _decision(t, "positive")
    pos = AgentNode(name="pos", llm=t.FakeLLM())
    neg = AgentNode(name="neg", llm=t.FakeLLM())
    d["positive"] > pos
    d["negative"] > neg
    assert d.route({"decision_clf": "positive"}) == "pos"
    assert d.route({"decision_clf": "negative"}) == "neg"


def test_route_unknown_choice_raises(t):
    d = _decision(t)
    d["positive"] > AgentNode(name="pos", llm=t.FakeLLM())
    with pytest.raises(ValueError, match="not found"):
        d.route({"decision_clf": "bogus"})


def test_route_childless_choice_raises(t):
    d = _decision(t)
    # no edges created from either choice
    with pytest.raises(ValueError, match="does not have a child"):
        d.route({"decision_clf": "positive"})


def _lookup(sentiment: str) -> str:
    """Look up the sentiment of a subject."""
    return sentiment


def test_decision_with_tools_gathers_then_routes(t):
    # Two-phase: the tool-bound model calls a tool (queued responses), then the
    # route model returns the structured choice (structured_value). Tools and
    # structured output never share a call.
    llm = t.FakeLLM(
        responses=[
            t.tool_call_msg("_lookup", {"sentiment": "positive"}),
            t.ai("the sentiment is positive"),
        ],
        structured_value="positive",
    )
    d = DecisionNode(name="clf", llm=llm, node_prompt="classify",
                     choices=["positive", "negative"], tools=[_lookup])
    delta = d({"messages": [HumanMessage(content="great job!")], "log": []})

    assert delta["decision_clf"] == "positive"
    assert "messages" not in delta  # routing label never pollutes the conversation
    # the tool loop ran (its trace lines precede the decision line) and routing
    # resolves to a real branch
    assert any("tools:_lookup" in line for line in delta["log"])
    assert delta["log"][-1] == "clf:positive"
    d["positive"] > AgentNode(name="pos", llm=t.FakeLLM())
    assert d.route({"decision_clf": "positive"}) == "pos"
