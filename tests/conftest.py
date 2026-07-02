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
    - `structured_value`: if set, `.invoke()` returns a structured object once the
      `responses` queue is exhausted. A scalar yields an object with `.choice`
      (DecisionNode); a dict yields an object exposing those fields (multi-field
      structured writes). Queuing `responses` (tool-call AIMessages) alongside a
      `structured_value` scripts DecisionNode's two-phase tool use: the tool loop
      drains the queue, then the route call gets the structured choice.
    `bind_tools`/`with_structured_output` return self so the scripted behaviour
    survives the wrapping that AgentNode/DecisionNode perform.
    """

    def __init__(self, responses=None, structured_value=None, repeat=False, fail_times=0, echo=False):
        self._responses = list(responses or [])
        self._i = 0
        self._structured_value = structured_value
        self._repeat = repeat  # when True, keep returning the last response
        self._fail_times = fail_times  # raise on the first N invokes (retry tests)
        self._echo = echo  # when True, reply echoing the last human message (parallel map)
        self.invoke_count = 0  # total invokes (cache/retry assertions)
        self.last_kwargs = {}  # kwargs from the most recent invoke (reasoning_effort)
        self.last_messages = None  # prompt from the most recent invoke (interpolation)

    def invoke(self, messages, **kwargs):
        self.last_kwargs = kwargs
        self.last_messages = messages
        self.invoke_count += 1
        if self.invoke_count <= self._fail_times:
            raise ConnectionError("simulated transient failure")
        if self._echo:
            last_human = messages[-1].content
            return ai(f"reply:{last_human}")
        # The response queue (tool-loop AIMessages) takes precedence; a
        # structured_value is returned once the queue is exhausted.
        if self._responses:
            if self._repeat and self._i >= len(self._responses):
                return self._responses[-1]
            if self._i < len(self._responses):
                resp = self._responses[self._i]
                self._i += 1
                return resp
        if self._structured_value is not None:
            if isinstance(self._structured_value, dict):
                return SimpleNamespace(**self._structured_value)
            return _Structured(self._structured_value)
        resp = self._responses[self._i]
        self._i += 1
        return resp

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, model):
        return self


class ChatOpenAI(FakeLLM):
    """FakeLLM whose class is *named* ChatOpenAI so nae's duck-typed OpenAI
    detection (`type(llm).__name__ == 'ChatOpenAI'`) treats it as OpenAI."""


_ids = itertools.count()


def ai(content="", tool_calls=None, usage_metadata=None, model_name=None):
    # Distinct id per message so add_messages appends instead of dedupes.
    # Optional usage_metadata / response_metadata.model_name feed the `token`
    # channel (per-model token tally); omit them to mimic providers/fakes that
    # report no usage (the node then contributes nothing to `token`).
    kwargs = {}
    if usage_metadata is not None:
        kwargs["usage_metadata"] = usage_metadata
    if model_name is not None:
        kwargs["response_metadata"] = {"model_name": model_name}
    return AIMessage(content=content, tool_calls=tool_calls or [],
                     id=f"ai-{next(_ids)}", **kwargs)


def tool_call_msg(name, args, call_id="call-1"):
    return ai(tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


@pytest.fixture
def t():
    """Toolkit: FakeLLM + message constructors."""
    return SimpleNamespace(FakeLLM=FakeLLM, ChatOpenAI=ChatOpenAI, ai=ai,
                           tool_call_msg=tool_call_msg)
