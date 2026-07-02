"""Issue #8: schema-free map-reduce.

`a > worker.map("docs") > collector` builds and runs with NO `state=` arg. The
spread `field` is auto-registered as a plain input channel (the Send machinery
reads it), and the worker's scalar write key becomes an
``Annotated[list, accumulate]`` collection channel so N parallel workers each
contribute one entry into a list instead of clobbering one plain key.
"""

from typing import Annotated

import pytest

from nae.graph import AgenticGraph
from nae.nodes import AgentNode
from nae.state import AgenticState, accumulate as append
from nae.validation import schema_keys


def _agent(t, name, content=None, **kw):
    return AgentNode(name=name, llm=t.FakeLLM(responses=[t.ai(content or name)]), **kw)


# --- 1. no state=: spread field + accumulated worker writes are inferred -------

def test_schema_free_mapreduce_accumulates(t):
    dispatch = _agent(t, "dispatch", "d-out")
    summarize = AgentNode(name="summarize", llm=t.FakeLLM(echo=True),
                          writes=["summaries"])  # scalar write -> list channel
    reduce = _agent(t, "reduce", "r-out",
                    node_prompt="summaries: {summaries}", reads=["summaries"])
    dispatch > summarize.map("docs") > reduce
    g = AgenticGraph(start_node=dispatch, end_nodes={reduce})  # NO state=

    # both inferred channels are present
    keys = schema_keys(g.state_schema)
    assert "docs" in keys and "summaries" in keys

    out = g.invoke(message="go", docs=["a", "b", "c"])

    # the 3 worker outputs accumulated into a list of 3
    assert isinstance(out["summaries"], list)
    assert len(out["summaries"]) == 3
    assert sorted(out["summaries"]) == ["reply:a", "reply:b", "reply:c"]


# --- 2. docs is accepted at invoke (no unknown-state-key error) ----------------

def test_spread_field_accepted_at_invoke(t):
    dispatch = _agent(t, "dispatch", "d-out")
    summarize = AgentNode(name="summarize", llm=t.FakeLLM(echo=True), writes=["summaries"])
    reduce = _agent(t, "reduce", "r-out", node_prompt="{summaries}", reads=["summaries"])
    dispatch > summarize.map("docs") > reduce
    g = AgenticGraph(start_node=dispatch, end_nodes={reduce})

    # would raise "unknown state key 'docs'" if the field weren't registered
    out = g.invoke(message="go", docs=["x"])
    assert out is not None


# --- 3. single-item map still yields a 1-element list -------------------------

def test_single_item_map(t):
    dispatch = _agent(t, "dispatch", "d-out")
    summarize = AgentNode(name="summarize", llm=t.FakeLLM(echo=True), writes=["summaries"])
    reduce = _agent(t, "reduce", "r-out", node_prompt="{summaries}", reads=["summaries"])
    dispatch > summarize.map("docs") > reduce
    g = AgenticGraph(start_node=dispatch, end_nodes={reduce})

    out = g.invoke(message="go", docs=["only"])
    assert out["summaries"] == ["reply:only"]


# --- 4. typed mapped write: native int values accumulate ----------------------

def test_typed_mapped_write_accumulates_ints(t):
    dispatch = _agent(t, "dispatch", "d-out")
    # dict-form writes -> structured output; FakeLLM yields a native int score.
    score = AgentNode(name="score", llm=t.FakeLLM(structured_value={"score": 7}),
                      writes={"score": int})
    reduce = _agent(t, "reduce", "r-out", node_prompt="{score}", reads=["score"])
    dispatch > score.map("docs") > reduce
    g = AgenticGraph(start_node=dispatch, end_nodes={reduce})

    out = g.invoke(message="go", docs=["a", "b"])
    assert isinstance(out["score"], list) and len(out["score"]) == 2
    assert all(isinstance(v, int) for v in out["score"])


# --- 5. explicit state= still works and is NOT double-wrapped -----------------

def test_explicit_state_not_double_wrapped(t):
    class MapState(AgenticState):
        docs: list
        summaries: Annotated[list, append]

    dispatch = _agent(t, "dispatch", "d-out")
    summarize = AgentNode(name="summarize", llm=t.FakeLLM(echo=True), writes=["summaries"])
    reduce = _agent(t, "reduce", "r-out", node_prompt="{summaries}", reads=["summaries"])
    dispatch > summarize.map("docs") > reduce
    g = AgenticGraph(state=MapState, start_node=dispatch, end_nodes={reduce})

    out = g.invoke(message="go", docs=["a", "b", "c"])
    # flat list of 3 scalars, NOT a list-of-lists (no extra accumulate wrap)
    assert sorted(out["summaries"]) == ["reply:a", "reply:b", "reply:c"]


# --- 6. regression: a plain schema-free graph is unaffected --------------------

def test_plain_schema_free_chain_regression(t):
    a = _agent(t, "a", "a-out")
    b = _agent(t, "b", "b-out")
    a > b
    g = AgenticGraph(start_node=a, end_nodes={b})  # no map, no state=

    out = g.invoke(message="hi")
    contents = [m.content for m in out["messages"]]
    assert "a-out" in contents and "b-out" in contents
