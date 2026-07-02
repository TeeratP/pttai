"""Multi-key state IO: reads/writes (Phase 3)."""

import operator
from typing import Annotated, TypedDict

import pytest
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages

from pttai.graph import AgenticGraph
from pttai.nodes import AgentNode


class MultiState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    log: Annotated[list[str], operator.add]
    sentiment: str
    score: str


def double(x: int) -> int:
    """Double x."""
    return x * 2


def test_multi_read_interpolates_prompt(t):
    llm = t.FakeLLM(responses=[t.ai("ok")])
    node = AgentNode(name="n", llm=llm, node_prompt="about {topic} for {audience}",
                     reads=["topic", "audience"])
    node({"topic": "cats", "audience": "kids", "log": []})
    assert llm.last_messages[0].content == "about cats for kids"


def test_multi_read_messages_plus_scalar(t):
    llm = t.FakeLLM(responses=[t.ai("ok")])
    node = AgentNode(name="n", llm=llm, node_prompt="t {topic}", reads=["messages", "topic"])
    node({"messages": [HumanMessage(content="hi")], "topic": "x", "log": []})
    assert llm.last_messages[0].content == "t x"        # scalar interpolated
    assert llm.last_messages[1].content == "hi"          # history preserved


def test_multi_write_structured_returns_keys(t):
    llm = t.FakeLLM(structured_value={"sentiment": "pos", "score": "5"})
    node = AgentNode(name="n", llm=llm, node_prompt="p", writes=["sentiment", "score"])
    delta = node({"messages": [HumanMessage(content="hi")], "log": []})
    assert delta["sentiment"] == "pos"
    assert delta["score"] == "5"
    assert "messages" not in delta
    assert "log" in delta


def test_multi_write_e2e_through_graph(t):
    n = AgentNode(name="n", llm=t.FakeLLM(structured_value={"sentiment": "pos", "score": "5"}),
                  node_prompt="p", writes=["sentiment", "score"])
    g = AgenticGraph(state=MultiState, start_node=n, end_nodes=n)
    out = g.invoke({"messages": [HumanMessage(content="hi")], "log": [],
                    "sentiment": "", "score": ""})
    assert out["sentiment"] == "pos"
    assert out["score"] == "5"


def test_multi_write_with_tools_raises(t):
    with pytest.raises(ValueError, match="cannot be combined with tools"):
        AgentNode(name="n", llm=t.FakeLLM(responses=[t.ai("x")]),
                  node_prompt="p", writes=["a", "b"], tools=[double])


def test_braces_prompt_without_reads_untouched(t):
    llm = t.FakeLLM(responses=[t.ai("ok")])
    node = AgentNode(name="n", llm=llm, node_prompt='return {"k": 1}')
    node({"messages": [HumanMessage(content="hi")], "log": []})
    assert llm.last_messages[0].content == 'return {"k": 1}'


def test_single_scalar_write_equals_output_field(t):
    a = AgentNode(name="a", llm=t.FakeLLM(responses=[t.ai("sum")]),
                  node_prompt="p", writes=["summary"])
    da = a({"messages": [HumanMessage(content="x")], "log": []})
    b = AgentNode(name="b", llm=t.FakeLLM(responses=[t.ai("sum")]),
                  node_prompt="p", output_field="summary")
    db = b({"messages": [HumanMessage(content="x")], "log": []})
    assert set(da) == set(db)
    assert da["summary"] == db["summary"] == "sum"
    assert "messages" not in da


def test_mixed_messages_and_scalar_writes_raises(t):
    node = AgentNode(name="n", llm=t.FakeLLM(responses=[t.ai("x")]),
                     node_prompt="p", writes=["messages", "x"])
    with pytest.raises(ValueError, match="not both"):
        node({"messages": [HumanMessage(content="hi")], "log": []})


# --- typed writes (dict[str, type]) ---------------------------------------

class TypedState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    log: Annotated[list[str], operator.add]
    a: int
    b: bool


def test_typed_dict_write_roundtrips_native_types(t):
    # FakeLLM scripts native int/bool through with_structured_output ->
    # SimpleNamespace, so getattr returns the real types, not strings.
    llm = t.FakeLLM(structured_value={"a": 7, "b": True})
    node = AgentNode(name="n", llm=llm, node_prompt="p", writes={"a": int, "b": bool})
    delta = node({"messages": [HumanMessage(content="hi")], "log": []})
    assert delta["a"] == 7 and isinstance(delta["a"], int)
    assert delta["b"] is True
    assert "messages" not in delta and "log" in delta


def test_typed_dict_write_e2e_through_graph(t):
    n = AgentNode(name="n", llm=t.FakeLLM(structured_value={"a": 9, "b": False}),
                  node_prompt="p", writes={"a": int, "b": bool})
    g = AgenticGraph(state=TypedState, start_node=n, end_nodes=n)
    out = g.invoke({"messages": [HumanMessage(content="hi")], "log": [], "a": 0, "b": True})
    assert out["a"] == 9 and isinstance(out["a"], int)
    assert out["b"] is False


def test_list_writes_still_yield_strings(t):
    # list[str] form unchanged: both fields typed str.
    n = AgentNode(name="n", llm=t.FakeLLM(structured_value={"a": "1", "b": "2"}),
                  node_prompt="p", writes=["a", "b"])
    assert n.writes == {"a": str, "b": str}
    delta = n({"messages": [HumanMessage(content="hi")], "log": []})
    assert delta["a"] == "1" and delta["b"] == "2"


def test_single_key_dict_uses_structured_mode(t):
    # A 1-key dict forces structured/typed mode (vs list-of-one .content write).
    llm = t.FakeLLM(structured_value={"x": 42})
    node = AgentNode(name="n", llm=llm, node_prompt="p", writes={"x": int})
    delta = node({"messages": [HumanMessage(content="hi")], "log": []})
    assert delta["x"] == 42 and isinstance(delta["x"], int)
    assert "messages" not in delta


def test_typed_dict_write_with_tools_raises(t):
    with pytest.raises(ValueError, match="cannot be combined with tools"):
        AgentNode(name="n", llm=t.FakeLLM(responses=[t.ai("x")]),
                  node_prompt="p", writes={"x": int}, tools=[double])
