"""Shared LLM helper for the examples — real OpenAI when a key is set, else an
offline fake so every example runs with NO ``OPENAI_API_KEY``.

    from _llm import get_llm      # (with a sys.path shim, see the example files)
    llm = get_llm()

``get_llm()`` returns:
  * ``ChatOpenAI(model="gpt-5.4-nano")`` when ``OPENAI_API_KEY`` is set, or
  * a scripted fake chat model otherwise.

The fake supports everything the examples exercise:
  * ``.invoke(messages)``            -> an ``AIMessage`` with plausible content;
  * ``.bind_tools(tools)``           -> a variant that emits ONE tool call on the
                                        first turn, then a final text answer (so
                                        an ``AgentNode`` tool loop terminates);
  * ``.with_structured_output(Model)`` -> a valid ``Model`` instance (for a
                                        ``Literal`` field it picks the FIRST
                                        allowed value; other fields get a
                                        type-appropriate value, e.g. int -> 7).

It is intentionally small — enough to make the examples runnable and readable,
not a full model simulator.
"""

import inspect
import itertools
import os
from typing import Literal, get_args, get_origin

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

_ids = itertools.count()


def _nid() -> str:
    # A distinct id per message so the add_messages reducer appends (never dedupes).
    return f"fake-{next(_ids)}"


def _last_human_text(messages) -> str:
    """The most recent human/user text in the prompt, for an echo-y reply."""
    for m in reversed(messages):
        content = getattr(m, "content", None)
        if isinstance(m, HumanMessage) and content:
            return content
    # fall back to the last message that has any string content
    for m in reversed(messages):
        content = getattr(m, "content", None)
        if isinstance(content, str) and content:
            return content
    return ""


def _tool_name(tool) -> str:
    """Tool name for either a LangChain tool (``.name``) or a plain callable."""
    return getattr(tool, "name", None) or getattr(tool, "__name__", "tool")


def _fake_tool_args(tool) -> dict:
    """Build type-appropriate placeholder args so the tool validates and runs.

    Handles a LangChain tool (JSON-schema ``.args``) and a plain callable (bound
    directly in the raw-LangGraph blocks) via its signature annotations.
    """
    by_type = {"integer": 7, "number": 7.0, "boolean": True, "array": [], "object": {}}
    schema = getattr(tool, "args", None)
    if isinstance(schema, dict):
        return {name: by_type.get(spec.get("type"), "sample") for name, spec in schema.items()}
    by_pytype = {int: 7, float: 7.0, bool: True, list: [], dict: {}}
    args = {}
    for name, param in inspect.signature(tool).parameters.items():
        args[name] = by_pytype.get(param.annotation, "sample")
    return args


def _value_for(annotation):
    """A valid value for one structured-output field annotation."""
    if get_origin(annotation) is Literal:
        return get_args(annotation)[0]            # first allowed choice
    if annotation is int:
        return 7
    if annotation is float:
        return 7.0
    if annotation is bool:
        return True
    if annotation is list or get_origin(annotation) is list:
        return []
    return "..."                                   # str and anything else


def _instantiate(model):
    """Construct a valid pydantic ``model`` instance with placeholder field values."""
    return model(**{name: _value_for(field.annotation)
                    for name, field in model.model_fields.items()})


class _StructuredFake:
    """What ``.with_structured_output(Model)`` returns: ``.invoke()`` yields a
    valid ``Model`` instance."""

    def __init__(self, model):
        self._model = model

    def invoke(self, messages, **kwargs):
        return _instantiate(self._model)

    def bind_tools(self, tools):
        return self


class _FakeLLM:
    """Scripted offline chat model (see the module docstring)."""

    def __init__(self, tools=None):
        self._tools = list(tools) if tools else None
        self.invoke_count = 0

    def invoke(self, messages, **kwargs):
        self.invoke_count += 1
        last = messages[-1] if messages else None
        # Tool phase: on the first turn (last message is the human ask, not yet a
        # ToolMessage) request one tool call; once a ToolMessage is present, the
        # tool has run — answer for real. This is stateless, so re-runs behave.
        if self._tools and not isinstance(last, ToolMessage):
            tool = self._tools[0]
            return AIMessage(
                content="",
                id=_nid(),
                tool_calls=[{
                    "name": _tool_name(tool),
                    "args": _fake_tool_args(tool),
                    "id": "call_1",
                    "type": "tool_call",
                }],
            )
        text = _last_human_text(messages)
        content = f"(offline fake) responding to: {text}" if text else "(offline fake) hello!"
        return AIMessage(content=content, id=_nid())

    def bind_tools(self, tools):
        return _FakeLLM(tools=tools)

    def with_structured_output(self, model):
        return _StructuredFake(model)


def get_llm():
    """Return a chat model: real OpenAI if ``OPENAI_API_KEY`` is set, else a fake."""
    if os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-5.4-nano")
    return _FakeLLM()
