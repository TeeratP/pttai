"""Offline regression tests for the node-runtime bug fixes (issue #13).

All tests use a fake LLM exposing invoke/bind_tools/with_structured_output;
no API calls.
"""
import datetime

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from nae import AgentNode, DecisionNode
from nae.node import _infer_name  # noqa: F401  (kept for clarity of what's exercised)
from nae.nodes._fields import partition_reads


# --- fakes ---------------------------------------------------------------

class FakeLLM:
    """Minimal fake: always returns a fixed AIMessage."""
    def __init__(self, content="ok"):
        self._content = content

    def invoke(self, messages, **kwargs):
        return AIMessage(content=self._content, id="x")

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, model):
        return self


class ScriptedLLM:
    """Returns a queued list of responses, one per invoke()."""
    def __init__(self, responses):
        self._responses = list(responses)

    def invoke(self, messages, **kwargs):
        return self._responses.pop(0)

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, model):
        return self


# --- Fix 8: DecisionNode without llm -> clear ValueError -----------------

def test_decision_node_without_llm_raises_value_error():
    with pytest.raises(ValueError, match="DecisionNode requires an llm"):
        DecisionNode(name="r", node_prompt="p", choices=["a", "b"])


# --- Fix 9: unknown tool name -> error ToolMessage, run completes --------

def test_unknown_tool_name_does_not_crash():
    bad_call = AIMessage(
        content="",
        id="a1",
        tool_calls=[{"name": "no_such_tool", "args": {}, "id": "tc1"}],
    )
    final = AIMessage(content="done", id="a2")

    def real_tool(x: str) -> str:
        """a real tool"""
        return x

    node = AgentNode(name="agent", llm=ScriptedLLM([bad_call, final]), tools=[real_tool])
    delta = node({"messages": [HumanMessage(content="hi")]})

    msgs = delta["messages"]
    err = [m for m in msgs if isinstance(m, ToolMessage)]
    assert len(err) == 1
    assert err[0].tool_call_id == "tc1"
    assert "unknown tool" in err[0].content
    # the run finished with the model's normal answer
    assert msgs[-1].content == "done"


# --- Fix 10: non-JSON tool return -> stringified, run completes ----------

def test_non_json_tool_return_does_not_crash():
    call = AIMessage(
        content="",
        id="a1",
        tool_calls=[{"name": "now", "args": {}, "id": "tc1"}],
    )
    final = AIMessage(content="done", id="a2")

    fixed = datetime.datetime(2026, 6, 30, 12, 0, 0)

    def now() -> datetime.datetime:
        """returns a datetime (non-JSON-serializable)"""
        return fixed

    node = AgentNode(name="agent", llm=ScriptedLLM([call, final]), tools=[now])
    delta = node({"messages": [HumanMessage(content="hi")]})

    tool_msgs = [m for m in delta["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert isinstance(tool_msgs[0].content, str)
    assert str(fixed) in tool_msgs[0].content


# --- Fix 11: empty-list read classifies as history, not scalar ----------

def test_empty_list_read_is_history_not_scalar():
    state = {"messages": [], "extra": []}
    history, scalars = partition_reads(state, ["extra"])
    assert history == []
    assert scalars == {}  # empty list did NOT land in scalars


def test_non_empty_non_message_list_is_scalar():
    state = {"items": [1, 2, 3]}
    history, scalars = partition_reads(state, ["items"])
    assert history == []
    assert scalars == {"items": [1, 2, 3]}


def test_empty_read_with_literal_brace_prompt_does_not_raise():
    # node_prompt contains a literal brace; an empty-list read must classify as
    # history so .format_map is never run over the prompt.
    node = AgentNode(
        name="agent",
        llm=FakeLLM(),
        node_prompt="respond with JSON like {\"k\": 1}",
        reads=["history"],
    )
    delta = node({"history": []})  # empty list -> history, no interpolation
    assert "messages" in delta


# --- Fix 12: HumanNode(n=k) with fewer than k messages -> no IndexError --

def test_human_node_n_larger_than_history(monkeypatch):
    import nae.nodes.human_node as hn

    monkeypatch.setattr(hn, "interrupt", lambda payload: "reply")
    node = hn.HumanNode(name="human", n=5)
    state = {"messages": [HumanMessage(content="one"), HumanMessage(content="two")]}
    delta = node(state)  # must not raise IndexError
    assert delta["messages"][0].content == "reply"


# --- Fix 13: _infer_name memoization still infers distinct names ---------

def test_infer_name_distinct_from_different_source():
    reviewer = AgentNode(llm=FakeLLM())
    writer = AgentNode(llm=FakeLLM())
    assert reviewer.name == "reviewer"
    assert writer.name == "writer"
