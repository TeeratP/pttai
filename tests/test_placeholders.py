"""Issue #4: scalar reads must line up with node_prompt {placeholders}."""

import operator
from typing import Annotated, TypedDict

import pytest
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from nae.graph import AgenticGraph
from nae.nodes import AgentNode
from nae.validation import GraphValidationError


class TopicState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    log: Annotated[list[str], operator.add]
    topic: str       # scalar input (read, no writer)
    audience: str     # scalar input (read, no writer)


def _agent(t, name, **kw):
    return AgentNode(name=name, llm=t.FakeLLM(responses=[t.ai(name)]), **kw)


def test_used_but_undeclared_raises(t):
    # `{audience}` is a declared scalar read (so .format_map runs), but `{topic}`
    # is referenced with no matching read -> guaranteed KeyError -> hard error.
    n = _agent(t, "n", node_prompt="{topic} for {audience}", reads=["messages", "audience"])
    with pytest.raises(GraphValidationError) as ei:
        AgenticGraph(state=TopicState, start_node=n, end_nodes=n)
    msg = str(ei.value)
    assert "topic" in msg
    assert "KeyError" in msg


def test_declared_but_unused_warns(t):
    # `topic` is a scalar read but the prompt never interpolates {topic} -> dead
    # read -> warning (not fatal).
    n = _agent(t, "n", node_prompt="just classify the conversation", reads=["messages", "topic"])
    g = AgenticGraph(state=TopicState, start_node=n, end_nodes=n)
    report = g.validate()
    assert report.ok is True
    assert any("topic" in w.message and "dead read" in w.message for w in report.warnings)


def test_braces_no_scalar_reads_passes(t):
    # JSON-brace-heavy prompt with NO scalar reads: .format_map is skipped at
    # runtime, so the placeholder check is skipped too -> clean.
    n = _agent(t, "n", node_prompt='return {"k": 1}', reads=["messages"])
    g = AgenticGraph(start_node=n, end_nodes=n)
    report = g.validate()
    assert report.ok is True
    assert report.warnings == []


def test_classify_example_passes(t):
    # The canonical valid case: reads topic, prompt interpolates {topic}.
    classify = AgentNode(
        name="classify",
        llm=t.FakeLLM(structured_value={"sentiment": "pos", "score": "5"}),
        node_prompt="Rate the {topic} discussion. Return a sentiment and a 1-10 score.",
        reads=["messages", "topic"],
        writes=["sentiment", "score"],
    )
    g = AgenticGraph(state=TopicState, start_node=classify, end_nodes=classify)
    report = g.validate()
    assert report.ok is True
    # topic is interpolated -> no dead-read warning for it
    assert not any("topic" in w.message for w in report.warnings)
