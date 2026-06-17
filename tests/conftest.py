"""Shared test fixtures: a scripted fake LLM and message helpers (no real API)."""

import itertools
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage


class _Structured:
    """Mimics a pydantic structured-output result with a `.choice` attribute."""

    def __init__(self, choice):
        self.choice = choice


class FakeLLM:
    """Scripted stand-in for a chat model.

    - `responses`: list of AIMessages returned one-per-`.invoke()` call.
    - `structured_value`: if set, every `.invoke()` returns an object whose
      `.choice` is this value (used for DecisionNode).
    `bind_tools`/`with_structured_output` return self so the scripted behaviour
    survives the wrapping that AgentNode/DecisionNode perform.
    """

    def __init__(self, responses=None, structured_value=None):
        self._responses = list(responses or [])
        self._i = 0
        self._structured_value = structured_value

    def invoke(self, messages):
        if self._structured_value is not None:
            return _Structured(self._structured_value)
        resp = self._responses[self._i]
        self._i += 1
        return resp

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, model):
        return self


_ids = itertools.count()


def ai(content="", tool_calls=None):
    # Distinct id per message so add_messages appends instead of dedupes.
    return AIMessage(content=content, tool_calls=tool_calls or [], id=f"ai-{next(_ids)}")


def tool_call_msg(name, args, call_id="call-1"):
    return ai(tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


@pytest.fixture
def t():
    """Toolkit: FakeLLM + message constructors."""
    return SimpleNamespace(FakeLLM=FakeLLM, ai=ai, tool_call_msg=tool_call_msg)
