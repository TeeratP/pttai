"""AgentNode tool-call loop accumulates one delta and does not mutate input state."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from nae.nodes import AgentNode


def double(x: int) -> int:
    """Double x."""
    return x * 2


def test_tool_loop_accumulates_single_delta(t):
    tc = t.tool_call_msg("double", {"x": 3})
    final = t.ai("the answer is 6")
    node = AgentNode(name="agent", llm=t.FakeLLM(responses=[tc, final]), node_prompt="p",
                     tools=[double])

    state = {"messages": [HumanMessage(content="double 3")], "log": []}
    delta = node(state)

    msgs = delta["messages"]
    assert len(msgs) == 3
    assert msgs[0] is tc
    assert isinstance(msgs[1], ToolMessage)
    assert msgs[1].content == "6"  # json.dumps(6)
    assert msgs[2] is final
    assert len(delta["log"]) == 3
    # input state must be untouched (no in-place mutation)
    assert len(state["messages"]) == 1


def test_no_tools_returns_single_message_delta(t):
    final = t.ai("hello")
    node = AgentNode(name="agent", llm=t.FakeLLM(responses=[final]), node_prompt="p")
    delta = node({"messages": [HumanMessage(content="hi")], "log": []})
    assert delta["messages"] == [final]
    assert delta["log"] == ["agent:hello"]


def test_tool_loop_respects_max_iterations(t):
    tc = t.tool_call_msg("double", {"x": 1})
    node = AgentNode(name="agent", llm=t.FakeLLM(responses=[tc], repeat=True),
                     node_prompt="p", max_tool_iterations=3, tools=[double])
    state = {"messages": [HumanMessage(content="go")], "log": []}
    with pytest.raises(RuntimeError, match="max_tool_iterations"):
        node(state)


def test_missing_messages_key_raises(t):
    node = AgentNode(name="agent", llm=t.FakeLLM(responses=[t.ai("x")]), node_prompt="p")
    try:
        node({"log": []})
        assert False, "expected ValueError"
    except ValueError as e:
        assert "messages" in str(e)
