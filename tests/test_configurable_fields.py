"""Configurable per-node input/output state fields."""

import operator
from typing import Annotated, TypedDict

import pytest
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages

from nae.graph import AgenticGraph
from nae.nodes import AgentNode


class SummaryState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    log: Annotated[list[str], operator.add]
    summary: str


def test_output_field_writes_custom_key(t):
    s = AgentNode(name="s", llm=t.FakeLLM(responses=[t.ai("a short summary")]),
                  node_prompt="summarize", output_field="summary")
    g = AgenticGraph(state=SummaryState, start_node=s, end_nodes=s)
    out = g.invoke({"messages": [HumanMessage(content="long text")], "log": [], "summary": ""})

    assert out["summary"] == "a short summary"
    # conversation history is left untouched when output goes to a custom field
    assert [m.content for m in out["messages"]] == ["long text"]


def test_input_field_read(t):
    node = AgentNode(name="n", llm=t.FakeLLM(responses=[t.ai("ok")]),
                     node_prompt="p", input_field="docs")
    delta = node({"docs": [HumanMessage(content="hi")], "log": []})
    assert delta["messages"][-1].content == "ok"


def test_missing_input_field_raises(t):
    node = AgentNode(name="n", llm=t.FakeLLM(responses=[t.ai("x")]), input_field="docs")
    with pytest.raises(ValueError, match="docs"):
        node({"messages": [], "log": []})
