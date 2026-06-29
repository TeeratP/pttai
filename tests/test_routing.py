"""DecisionNode writes to the `decision` field (not messages) and routes off it."""

import pytest
from langchain_core.messages import HumanMessage

from pttai.nodes import AgentNode, DecisionNode


def _decision(t, value="positive"):
    return DecisionNode(name="clf", llm=t.FakeLLM(structured_value=value),
                        node_prompt="p", choices=["positive", "negative"])


def test_returns_decision_field_not_message(t):
    d = _decision(t, "positive")
    delta = d({"messages": [HumanMessage(content="hi")], "log": [], "decision": ""})
    assert delta == {"decision": "positive", "log": ["clf:positive"]}
    assert "messages" not in delta  # label must not pollute conversation history


def test_route_picks_correct_branch(t):
    d = _decision(t, "positive")
    pos = AgentNode(name="pos", llm=t.FakeLLM())
    neg = AgentNode(name="neg", llm=t.FakeLLM())
    d["positive"] > pos
    d["negative"] > neg
    assert d.route({"decision": "positive"}) == "pos"
    assert d.route({"decision": "negative"}) == "neg"


def test_route_unknown_choice_raises(t):
    d = _decision(t)
    d["positive"] > AgentNode(name="pos", llm=t.FakeLLM())
    with pytest.raises(ValueError, match="not found"):
        d.route({"decision": "bogus"})


def test_route_childless_choice_raises(t):
    d = _decision(t)
    # no edges created from either choice
    with pytest.raises(ValueError, match="does not have a child"):
        d.route({"decision": "positive"})
