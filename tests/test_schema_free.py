"""Issue #7: schema-free state.

A graph needs NO custom state schema. On the default ``AgenticState`` any
non-standard key a node READS (and no node writes) is auto-registered as a plain
INPUT channel and seeded at invoke; keys a node writes are auto-registered as
plain channels. The framework-managed RESERVED channels (messages/log/token)
cannot be user-declared on a node or seeded as an invoke extra kwarg, and
an unknown invoke key is a clear error rather than a silent drop.
"""

import pytest
from langchain_core.messages import HumanMessage

from pttai.graph import AgenticGraph
from pttai.nodes import AgentNode
from pttai.state import AgenticState
from pttai.validation import GraphValidationError, schema_keys


def _agent(t, name, content=None, **kw):
    return AgentNode(name=name, llm=t.FakeLLM(responses=[t.ai(content or name)]), **kw)


# --- 1. fully schema-free multi-node graph; a node reads an input key ---------

def test_schema_free_multinode_reads_input(t):
    # Two nodes, NO state= arg. `a` reads `topic` (no node writes it) -> it is
    # auto-registered as an input and seeded at invoke; the node sees its value.
    llm_a = t.FakeLLM(responses=[t.ai("ra")])
    a = AgentNode(name="a", llm=llm_a, node_prompt="topic is {topic}", reads=["topic"])
    b = _agent(t, "b", content="rb")
    a > b
    g = AgenticGraph(start_node=a, end_nodes=b)  # default AgenticState only

    assert "topic" in schema_keys(g.state_schema)  # auto-registered channel
    assert g.validate().ok is True                 # no dangling-read error
    out = g.invoke(message="x", topic="y")
    assert llm_a.last_messages[0].content == "topic is y"  # node saw the seeded value
    assert out["topic"] == "y"


# --- 2. a WRITTEN key still gets read-before-write ordering validation --------

def test_written_key_read_before_write_still_caught(t):
    # `result` IS written (by `late`), so it stays a COMPUTED key, NOT an input.
    # `early` reads it before `late` runs -> the ordering guardrail still fires
    # (proves never-written-read -> input did not weaken the real check).
    early = _agent(t, "early", node_prompt="use {result}", reads=["result"])
    late = _agent(t, "late", content="L", output_field="result")
    early > late
    with pytest.raises(GraphValidationError) as ei:
        AgenticGraph(start_node=early, end_nodes=late)  # default state
    msg = str(ei.value)
    assert "result" in msg
    assert "late" in msg            # names the (downstream) writer
    assert "upstream" in msg.lower()


def test_written_key_produced_upstream_passes(t):
    # The mirror of the above: producer first, consumer second -> valid.
    prod = _agent(t, "prod", content="P", output_field="result")
    cons = _agent(t, "cons", node_prompt="use {result}", reads=["result"])
    prod > cons
    g = AgenticGraph(start_node=prod, end_nodes=cons)  # default state
    assert g.validate().ok is True


# --- 3. reserved channels cannot be seeded at invoke --------------------------

def _single(t, **kw):
    a = _agent(t, "a", content="r", **kw)
    return AgenticGraph(start_node=a, end_nodes=a)


def test_invoke_reserved_log_raises(t):
    g = _single(t)
    with pytest.raises(ValueError, match="reserved"):
        g.invoke(message="x", log=["nope"])


def test_invoke_reserved_token_raises(t):
    g = _single(t)
    with pytest.raises(ValueError, match="reserved"):
        g.invoke(message="x", token={"m": {"total_tokens": 1}})


# --- 4. unknown invoke key -> clear error (not a silent drop) -----------------

def test_invoke_unknown_key_raises(t):
    g = _single(t)
    with pytest.raises(ValueError, match="bogus"):
        g.invoke(message="x", bogus=1)


def test_invoke_reserved_takes_precedence_over_unknown(t):
    # `token` is reserved AND in schema; the reserved error wins regardless.
    g = _single(t)
    with pytest.raises(ValueError, match="reserved"):
        g.invoke(message="x", token={})


# --- 5. a node declaring a reserved name as a user scalar -> build error ------

def test_node_writes_reserved_raises(t):
    a = _agent(t, "a", writes=["token"])  # user scalar collides with reserved
    with pytest.raises(ValueError, match="reserved"):
        AgenticGraph(start_node=a, end_nodes=a)


def test_node_reads_reserved_raises(t):
    a = _agent(t, "a", node_prompt="{log}", reads=["log"])
    with pytest.raises(ValueError, match="reserved"):
        AgenticGraph(start_node=a, end_nodes=a)


def test_default_messages_read_write_allowed(t):
    # `messages` is reserved but EXEMPT from the user-declaration guard: the
    # default node reads/writes `messages`, which must keep building.
    a = _agent(t, "a", content="hi")  # default reads/writes = messages
    g = AgenticGraph(start_node=a, end_nodes=a)
    out = g.invoke(message="hello")
    assert any(m.content == "hi" for m in out["messages"])


def test_framework_token_write_not_flagged(t):
    # The framework emits `token` from the agent delta (not via node.writes), so
    # a token-producing node must NOT trip the reserved-name guard.
    usage = {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
    a = AgentNode(name="a", llm=t.FakeLLM(
        responses=[t.ai("r", usage_metadata=usage, model_name="m")]))
    g = AgenticGraph(start_node=a, end_nodes=a)  # builds fine
    out = g.invoke(message="x")
    assert out["token"] == {"m": usage}
