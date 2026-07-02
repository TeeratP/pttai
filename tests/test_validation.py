"""Phase 4: compile-time state-availability validation + summary()."""

import io
import operator
from typing import Annotated, TypedDict

import pytest
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages

from nae import fanout
from nae.graph import AgenticGraph
from nae.nodes import AgentNode, DecisionNode
from nae.state import AgenticState
from nae.validation import GraphValidationError, schema_keys


class SummaryState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    log: Annotated[list[str], operator.add]
    summary: str


class LoopState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    log: Annotated[list[str], operator.add]
    summary: str  # produced before the loop
    draft: str    # produced inside the loop body


class XYState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    log: Annotated[list[str], operator.add]
    x: Annotated[list, operator.add]
    y: Annotated[list, operator.add]


class TagState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    log: Annotated[list[str], operator.add]
    tag: str  # plain (no reducer) -> concurrent writes are illegal


class CfgState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    log: Annotated[list[str], operator.add]
    cfg: str  # plain entry key, may be supplied at invoke (inputs=)


def _agent(t, name, content=None, **kw):
    return AgentNode(name=name, llm=t.FakeLLM(responses=[t.ai(content or name)]), **kw)


def test_neverwritten_read_autoregisters_as_input(t):
    # A read of a key NO node produces is no longer a hard error: it is
    # auto-registered as a plain INPUT channel (schema-free state) and seeded at
    # invoke. (This replaces the old dangling-read hard error — see issue #7.)
    n = _agent(t, "n", node_prompt="{context}", reads=["context"])  # never written
    g = AgenticGraph(state=AgenticState, start_node=n, end_nodes=n)  # builds, no raise
    assert "context" in schema_keys(g.state_schema)  # registered as a channel
    assert g.validate().ok is True                   # and it's a legit input, not an error
    out = g.invoke(message="hi", context="ctx-value")
    assert out["context"] == "ctx-value"             # value seeded at invoke round-trips


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


def test_write_unknown_key_autoregistered(t):
    # A node writing a key the schema doesn't declare no longer errors; the key
    # is auto-registered as a plain channel, so the graph builds and runs and
    # the value round-trips through invoke.
    n = _agent(t, "n", content="hi", output_field="sumary")  # not in schema
    g = AgenticGraph(state=SummaryState, start_node=n, end_nodes=n)
    assert "sumary" in schema_keys(g.state_schema)
    out = g.invoke({"messages": [HumanMessage(content="x")], "log": []})
    assert out["sumary"] == "hi"


def test_default_state_no_state_arg(t):
    # The graph builds with NO state= arg and runs on the default AgenticState.
    a = _agent(t, "a", content="hello")
    g = AgenticGraph(start_node=a, end_nodes=a)
    out = g.invoke({"messages": [HumanMessage(content="hi")], "log": []})
    assert any(m.content == "hello" for m in out["messages"])


def test_undeclared_scalar_autoregisters_and_roundtrips(t):
    # Default state + a node writing an undeclared scalar key: the key is
    # auto-registered and its value round-trips through invoke.
    n = _agent(t, "n", content="val", output_field="result")
    g = AgenticGraph(start_node=n, end_nodes=n)
    assert "result" in schema_keys(g.state_schema)
    out = g.invoke({"messages": [HumanMessage(content="x")], "log": []})
    assert out["result"] == "val"


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

    assert "initial:" in out and "decision_clf" in out  # header lists channels incl. the router's per-node key
    assert "decision\n" not in out and "decision," not in out and "decision " not in out  # no bare shared `decision` channel
    for name in ("clf", "pos", "neg"):
        assert name in out                          # each node row present
    assert "DecisionNode" in out and "AgentNode" in out
    assert "messages" in out                        # reads column
    # clf writes the `decision_clf` key and pos/neg are available downstream of it
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


# --- cyclic dataflow: loop-carried read-before-write (issue #34) ---
#
# Back-edges carry loop-carried keys that do NOT exist on the first iteration.
# Before the fix, the availability fixpoint treated a loop-back edge like any
# forward predecessor, so a node reading a key produced ONLY by a downstream
# node that loops back was credited with it -> PASSED validation, then KeyError'd
# on iteration one. These tests pin that the analysis now excludes back-edges.


def test_cyclic_loop_carried_read_before_write_caught(t):
    # gen reads `summary`, but its ONLY writer `writer` is DOWNSTREAM and loops
    # back to gen via the gate. On first entry `summary` does not exist yet ->
    # runtime KeyError. This was a FALSE NEGATIVE before the back-edge fix
    # (the loop-back gate->gen edge wrongly made summary "available" at gen).
    from nae import ConditionNode

    gen = _agent(t, "gen", reads=["messages", "summary"])   # loop-carried read
    writer = _agent(t, "writer", output_field="summary")     # sole producer, downstream
    gate = ConditionNode(name="gate", condition=lambda s: "refine", choices=["refine", "accept"])
    fin = _agent(t, "fin")

    gen > writer > gate
    gate["refine"] > gen        # back-edge: closes the cycle gen->writer->gate->gen
    gate["accept"] > fin

    with pytest.raises(GraphValidationError) as ei:
        AgenticGraph(state=SummaryState, start_node=gen, end_nodes=fin)
    msg = str(ei.value)
    assert "summary" in msg          # names the loop-carried key
    assert "writer" in msg           # names the downstream (loop-back) producer
    assert "upstream" in msg.lower() # explains it's an ordering problem


def test_cyclic_legitimate_reads_pass(t):
    # A genuinely-valid loop must still validate clean (no false positive):
    #  - `gen` reads `summary`, produced by `pre` UPSTREAM-before-the-cycle;
    #  - `evaluate` reads `draft`, produced by `gen` EARLIER in the same loop
    #    body via a forward edge (gen->evaluate).
    # Neither read depends on a back-edge, so both are available on entry.
    from nae import ConditionNode

    pre = _agent(t, "pre", output_field="summary")               # before the loop
    gen = _agent(t, "gen", reads=["messages", "summary"],        # reads upstream key
                 writes={"draft": str})
    evaluate = _agent(t, "evaluate", reads=["messages", "draft"])  # reads same-loop-body key
    gate = ConditionNode(name="gate", condition=lambda s: "accept", choices=["refine", "accept"])
    fin = _agent(t, "fin")

    pre > gen > evaluate > gate
    gate["refine"] > gen        # back-edge; the loop body is gen->evaluate->gate
    gate["accept"] > fin

    g = AgenticGraph(state=LoopState, start_node=pre, end_nodes=fin)
    assert g.validate().ok is True
