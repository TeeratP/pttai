"""Pins the reducer semantics the delta-returning node design relies on (risk R1)."""

import operator

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import add_messages


def test_add_messages_appends_distinct():
    merged = add_messages(
        [HumanMessage(content="hi", id="1")],
        [AIMessage(content="yo", id="2")],
    )
    assert [m.content for m in merged] == ["hi", "yo"]


def test_add_messages_replaces_on_matching_id():
    merged = add_messages(
        [AIMessage(content="old", id="x")],
        [AIMessage(content="new", id="x")],
    )
    assert len(merged) == 1
    assert merged[0].content == "new"


def test_add_messages_coerces_bare_string():
    merged = add_messages([], ["hello"])
    assert len(merged) == 1
    assert isinstance(merged[0], HumanMessage)
    assert merged[0].content == "hello"


def test_operator_add_concatenates_log():
    assert operator.add(["a"], ["b", "c"]) == ["a", "b", "c"]
