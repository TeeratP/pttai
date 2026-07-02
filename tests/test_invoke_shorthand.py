"""Shorthand run-method inputs: a plain str / list of messages / full dict, plus
``**extra`` state keys, are normalized into a full state dict with auto-seeded
boilerplate channels (issue #6)."""

import asyncio

from langchain_core.messages import HumanMessage

from nae.graph import AgenticGraph, _normalize_input
from nae.nodes import AgentNode
from nae.state import AgenticState


def _single_node_graph(t, **node_kwargs):
    """A one-node graph (start == end) plus its node and scripted LLM."""
    llm = t.FakeLLM(responses=[t.ai("r1")])
    a = AgentNode(name="a", llm=llm,
                  node_prompt=node_kwargs.pop("node_prompt", "p"), **node_kwargs)
    g = AgenticGraph(state=AgenticState, start_node=a, end_nodes=a)
    return g, a, llm


# --- _normalize_input unit behavior ----------------------------------------

def test_normalize_str_wraps_humanmessage():
    s = _normalize_input("some text")
    assert s["log"] == []
    assert len(s["messages"]) == 1
    assert isinstance(s["messages"][0], HumanMessage)
    assert s["messages"][0].content == "some text"


def test_normalize_list_passes_messages_through():
    msgs = [HumanMessage(content="a"), HumanMessage(content="b")]
    s = _normalize_input(msgs)
    assert s["messages"] == msgs
    assert s["messages"] is not msgs  # copied, not aliased
    assert s["log"] == []


def test_normalize_dict_is_shallow_copy_and_seeds_log():
    original = {"messages": [HumanMessage(content="hi")], "topic": "x"}
    s = _normalize_input(original)
    assert s is not original
    assert s["log"] == []          # auto-seeded
    assert s["topic"] == "x"       # preserved
    assert "log" not in original   # did not mutate caller's dict


def test_normalize_extra_kwargs_merge():
    s = _normalize_input("hi", topic="product")
    assert s["topic"] == "product"
    assert s["messages"][0].content == "hi"
    assert s["log"] == []


def test_normalize_dict_keeps_explicit_log():
    s = _normalize_input({"messages": [], "log": ["pre"]})
    assert s["log"] == ["pre"]  # setdefault does not clobber


# --- run methods normalize -------------------------------------------------

def test_invoke_str(t):
    g, a, llm = _single_node_graph(t)
    out = g.invoke("hello world")
    # the node's LLM saw a HumanMessage carrying the raw text
    assert any(isinstance(m, HumanMessage) and m.content == "hello world"
               for m in llm.last_messages)
    assert out["messages"][0].content == "hello world"
    assert out["log"] == ["a:r1"]


def test_invoke_list_seeds_all_messages(t):
    g, a, llm = _single_node_graph(t)
    out = g.invoke([HumanMessage(content="a"), HumanMessage(content="b")])
    assert [m.content for m in out["messages"][:2]] == ["a", "b"]


def test_invoke_extra_kwargs_seed_state(t):
    llm = t.FakeLLM(responses=[t.ai("r1")])
    # No custom schema: `topic` is auto-registered as an input because the node
    # reads it (and no node writes it) on the default AgenticState.
    a = AgentNode(name="a", llm=llm, node_prompt="{topic}", reads=["topic"])
    g = AgenticGraph(start_node=a, end_nodes=a)
    g.invoke("hi", topic="product")
    # {topic} placeholder interpolated from the extra-seeded state key
    assert llm.last_messages[0].content == "product"


def test_invoke_full_dict_autoseeds_log_when_omitted(t):
    g, a, llm = _single_node_graph(t)
    out = g.invoke({"messages": [HumanMessage(content="hi")]})  # no log key
    assert out["log"] == ["a:r1"]


def test_invoke_full_dict_unchanged_back_compat(t):
    g, a, llm = _single_node_graph(t)
    out = g.invoke({"messages": [HumanMessage(content="hi")], "log": []})
    assert [m.content for m in out["messages"]] == ["hi", "r1"]
    assert out["log"] == ["a:r1"]


def test_stream_normalizes_str(t):
    g, a, llm = _single_node_graph(t)
    chunks = list(g.stream("streamed text"))
    assert chunks  # at least one update emitted
    assert any(isinstance(m, HumanMessage) and m.content == "streamed text"
               for m in llm.last_messages)


def test_invoke_message_keyword(t):
    g, a, llm = _single_node_graph(t)
    out = g.invoke(message="research something")
    assert any(isinstance(m, HumanMessage) and m.content == "research something"
               for m in llm.last_messages)
    assert out["messages"][0].content == "research something"
    assert out["log"] == ["a:r1"]


def test_invoke_message_keyword_with_extra(t):
    llm = t.FakeLLM(responses=[t.ai("r1")])
    a = AgentNode(name="a", llm=llm, node_prompt="{topic}", reads=["topic"])
    g = AgenticGraph(start_node=a, end_nodes=a)  # default state + auto-registered topic
    g.invoke(message="x", topic="product")
    assert llm.last_messages[0].content == "product"


def test_message_keyword_list_form():
    s = _normalize_input(message=[HumanMessage(content="a"), HumanMessage(content="b")])
    assert [m.content for m in s["messages"]] == ["a", "b"]
    assert s["log"] == []


def test_positional_and_message_both_raises():
    import pytest
    with pytest.raises(ValueError, match="not both"):
        _normalize_input("hi", message="bye")


def test_neither_input_nor_message_raises():
    import pytest
    with pytest.raises(ValueError):
        _normalize_input()


def test_ainvoke_normalizes_str_with_extra(t):
    llm = t.FakeLLM(responses=[t.ai("r1")])
    a = AgentNode(name="a", llm=llm, node_prompt="{topic}", reads=["topic"])
    g = AgenticGraph(start_node=a, end_nodes=a)  # default state + auto-registered topic
    asyncio.run(g.ainvoke("hi", topic="async-topic"))
    assert llm.last_messages[0].content == "async-topic"
