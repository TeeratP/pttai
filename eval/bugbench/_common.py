"""Shared helpers for the dataflow-bug benchmark (offline, no API key).

``CountingLLM`` wraps the scripted fake model from ``examples/_llm.py`` and
counts every ``.invoke`` (including ``bind_tools`` / ``with_structured_output``
variants), so "LLM calls a raw-LangGraph graph burns before the bug surfaces"
is a real measurement, not an estimate. ``run_langgraph`` compiles + invokes a
raw-LangGraph builder and reports the phase the bug surfaces in.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))

from _llm import get_llm  # offline fake unless OPENAI_API_KEY is set  # noqa: E402
from langchain_core.messages import SystemMessage  # noqa: E402


class CountingLLM:
    """Wraps a chat model and counts every ``.invoke``. The counter list is
    shared across all derived (bind_tools / with_structured_output) copies, so
    the count survives across a whole run."""

    def __init__(self, inner, counter):
        self._inner, self._counter = inner, counter

    def invoke(self, *a, **k):
        self._counter[0] += 1
        return self._inner.invoke(*a, **k)

    def bind_tools(self, tools):
        return CountingLLM(self._inner.bind_tools(tools), self._counter)

    def with_structured_output(self, model):
        return CountingLLM(self._inner.with_structured_output(model), self._counter)


def counting_llm():
    """Return ``(CountingLLM, counter)`` where ``counter[0]`` is the live call count."""
    counter = [0]
    return CountingLLM(get_llm(), counter), counter


def step(llm, prompt):
    """A raw-LangGraph node that prepends a SystemMessage and calls the model —
    the same shape used throughout ``examples/`` for the LangGraph side."""

    def node(state):
        return {"messages": [llm.invoke([SystemMessage(prompt)] + state["messages"])]}

    return node


def run_langgraph(build_fn, invoke_input):
    """Build (add_node/add_edge), compile, then invoke a raw-LangGraph graph and
    report where a bug surfaces: ``build`` (construction/compile raised),
    ``runtime`` (invoke raised, after >=0 model calls), or ``silent`` (no error
    at all — a latent wrong-output bug LangGraph never surfaces)."""
    out = {"phase": None, "error_type": None, "message": ""}
    try:
        builder = build_fn()          # may raise at add_node (e.g. duplicate name)
        compiled = builder.compile()  # may raise at compile (structural check)
    except Exception as e:            # noqa: BLE001 — we WANT to inspect it
        out["phase"] = "build"
        out["error_type"] = type(e).__name__
        out["message"] = str(e).splitlines()[0]
        return out
    try:
        compiled.invoke(invoke_input)
        out["phase"] = "silent"
    except Exception as e:            # noqa: BLE001
        out["phase"] = "runtime"
        out["error_type"] = type(e).__name__
        out["message"] = str(e).splitlines()[0]
    return out


HELLO = {"messages": [{"role": "user", "content": "hi"}]}
