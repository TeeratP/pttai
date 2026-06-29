"""Phase 4: compile-time state-availability validation + summary()."""

import io
import operator
from typing import Annotated, TypedDict

import pytest
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from pttai import fanout
from pttai.graph import AgenticGraph
from pttai.nodes import AgentNode, DecisionNode
from pttai.state import AgenticState
from pttai.validation import GraphValidationError


class SummaryState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    log: Annotated[list[str], operator.add]
    decision: str
    summary: str


class XYState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    log: Annotated[list[str], operator.add]
    decision: str
    x: Annotated[list, operator.add]
    y: Annotated[list, operator.add]


class TagState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    log: Annotated[list[str], operator.add]
    decision: str
    tag: str  # plain (no reducer) -> concurrent writes are illegal


class CfgState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    log: Annotated[list[str], operator.add]
    decision: str
    cfg: str  # plain entry key, may be supplied at invoke (inputs=)


def _agent(t, name, content=None, **kw):
    return AgentNode(name=name, llm=t.FakeLLM(responses=[t.ai(content or name)]), **kw)


def test_missing_key_raises(t):
    n = _agent(t, "n", input_field="context")  # 'context' never produced
    with pytest.raises(GraphValidationError) as ei:
        AgenticGraph(state=AgenticState, start_node=n, end_nodes=n)
    msg = str(ei.value)
    assert "n" in msg                       # names the node
    assert "context" in msg                 # names the missing key
    assert "available" in msg.lower()       # lists available keys
    assert "messages" in msg                # ... and they include the schema keys


def test_valid_multikey_passes(t):
    prod = _agent(t, "prod", output_field="summary")  # writes summary
    cons = _agent(t, "cons", reads=["summary"])        # reads summary
    prod > cons
    g = AgenticGraph(state=SummaryState, start_node=prod, end_nodes=cons)
    assert g.validate().ok is True


def test_diamond_join_no_false_positive(t):
    # a -> {b writes x, c writes y} -> d reads x AND y. Both branches run
    # (AND-union), so x and y are both available at d: no false positive.
    a = _agent(t, "a")
    b = _agent(t, "b", writes=["x"])
    c = _agent(t, "c", writes=["y"])
    d = _agent(t, "d", reads=["x", "y"])
    a > fanout(b, c) > d
    g = AgenticGraph(state=XYState, start_node=a, end_nodes=d)
    assert g.validate().ok is True


def test_write_unknown_key_caught(t):
    n = _agent(t, "n", output_field="sumary")  # typo, not in schema
    with pytest.raises(GraphValidationError, match="not declared"):
        AgenticGraph(state=SummaryState, start_node=n, end_nodes=n)


def test_parallel_unreduced_write_caught(t):
    # b and c run concurrently and both write the plain key `tag` -> illegal.
    a = _agent(t, "a")
    b = _agent(t, "b", output_field="tag")
    c = _agent(t, "c", output_field="tag")
    d = _agent(t, "d")
    a > fanout(b, c) > d
    with pytest.raises(GraphValidationError, match="reducer|concurrent"):
        AgenticGraph(state=TagState, start_node=a, end_nodes=d)


def test_summary_output(t):
    d = DecisionNode(name="clf", llm=t.FakeLLM(structured_value="pos"),
                     node_prompt="p", choices=["pos", "neg"])
    pos = _agent(t, "pos")
    neg = _agent(t, "neg")
    d["pos"] > pos
    d["neg"] > neg
    g = AgenticGraph(state=AgenticState, start_node=d, end_nodes={pos, neg})

    buf = io.StringIO()
    g.summary(file=buf)
    out = buf.getvalue()

    assert "initial:" in out and "decision" in out  # header lists initial keys
    for name in ("clf", "pos", "neg"):
        assert name in out                          # each node row present
    assert "DecisionNode" in out and "AgentNode" in out
    assert "messages" in out                        # reads column
    # clf writes the `decision` key and pos/neg are available downstream of it
    assert "available" in out


def test_validate_false_skips(t):
    # Knowingly invalid (reads a key never produced) but validation is disabled.
    n = _agent(t, "n", input_field="context")
    g = AgenticGraph(state=AgenticState, start_node=n, end_nodes=n, validate=False)
    assert g.compiled_graph is not None


def test_optional_read_warns_not_errors(t):
    n = _agent(t, "n", reads=["messages", "maybe?"])  # `maybe` never produced
    g = AgenticGraph(state=AgenticState, start_node=n, end_nodes=n)  # must not raise
    report = g.validate()
    assert report.ok is True                                   # no errors
    assert any("maybe" in w.message for w in report.warnings)  # but a warning


# --- dataflow ordering: read-before-written / read on a non-producing branch ---

def test_read_before_written_caught(t):
    # `early` reads the plain key `summary`, but its ONLY writer `late` runs
    # DOWNSTREAM -> at runtime `early` KeyErrors. This is THE proof the stricter
    # input-vs-computed seeding works (the old "seed=all schema keys" missed it).
    early = _agent(t, "early", reads=["summary"])    # reads computed key
    late = _agent(t, "late", output_field="summary")  # the sole producer, later
    early > late
    with pytest.raises(GraphValidationError) as ei:
        AgenticGraph(state=SummaryState, start_node=early, end_nodes=late)
    msg = str(ei.value)
    assert "summary" in msg              # names the offending key
    assert "late" in msg                 # names the actual (downstream) writer
    assert "upstream" in msg.lower()     # explains it's an ordering problem


def test_read_on_parallel_branch_caught(t):
    # b writes `summary`; its sibling c reads it. b and c are concurrent, so c is
    # NOT downstream of b and c's only ancestor (a) does not produce summary ->
    # hard error (the key is unavailable on c's path).
    a = _agent(t, "a")
    b = _agent(t, "b", output_field="summary")   # writes plain summary
    c = _agent(t, "c", reads=["summary"])         # reads it on the sibling branch
    d = _agent(t, "d")
    a > fanout(b, c) > d
    with pytest.raises(GraphValidationError) as ei:
        AgenticGraph(state=SummaryState, start_node=a, end_nodes=d)
    assert "summary" in str(ei.value)            # names the unavailable key


def test_input_override_passes(t):
    # (A) `cfg` is read but written by NO node -> already inferred as an input
    # (a no-node-writes schema key must be supplied at invoke) -> passes.
    n = _agent(t, "n", reads=["messages", "cfg"])
    g = AgenticGraph(state=CfgState, start_node=n, end_nodes=n)
    assert g.validate().ok is True

    # (B) now a node DOES write `cfg`, downstream of a reader. Without inputs it's
    # a computed read-before-written -> error...
    early = _agent(t, "early", reads=["messages", "cfg"])
    late = _agent(t, "late", output_field="cfg")
    early > late
    with pytest.raises(GraphValidationError, match="cfg"):
        AgenticGraph(state=CfgState, start_node=early, end_nodes=late)

    # ...but declaring inputs={"cfg"} marks it provided at invoke -> passes.
    early2 = _agent(t, "early2", reads=["messages", "cfg"])
    late2 = _agent(t, "late2", output_field="cfg")
    early2 > late2
    g2 = AgenticGraph(state=CfgState, start_node=early2, end_nodes=late2,
                      inputs={"cfg"})
    assert g2.validate().ok is True


def test_produced_upstream_passes(t):
    # producer `p` writes summary, consumer `c` reads it strictly downstream -> ok.
    p = _agent(t, "p", output_field="summary")
    c = _agent(t, "c", reads=["summary"])
    p > c
    g = AgenticGraph(state=SummaryState, start_node=p, end_nodes=c)
    assert g.validate().ok is True
