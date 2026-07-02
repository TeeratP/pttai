"""ConditionNode: deterministic routing by a Python predicate (no LLM).

The code sibling of DecisionNode — writes its label to its `decision_{name}` field
and routes off it via the shared RouterNode machinery.
"""

import pytest

from nae.graph import AgenticGraph
from nae.nodes import AgentNode, DecisionNode, ConditionNode


def _agent(t, name, content):
    return AgentNode(name=name, llm=t.FakeLLM(responses=[t.ai(content)]))


# --- 1. routes by an input value, no LLM --------------------------------------

def test_routes_by_input_value(t):
    gate = ConditionNode(name="gate",
                         condition=lambda s: "left" if s["count"] < 5 else "right",
                         choices=["left", "right"], reads=["count"])
    a = _agent(t, "a", "AAA")
    b = _agent(t, "b", "BBB")
    gate["left"] > a
    gate["right"] > b
    g = AgenticGraph(start_node=gate, end_nodes=[a, b])  # schema-free: count auto-input

    out = g.invoke(message="x", count=3)
    assert out["decision_gate"] == "left"
    assert any(m.content == "AAA" for m in out["messages"])
    assert "gate:left" in out["log"]

    out = g.invoke(message="x", count=7)
    assert out["decision_gate"] == "right"
    assert any(m.content == "BBB" for m in out["messages"])


# --- 2. terminating loop with NO counter primitive ----------------------------

def test_terminating_loop_no_counter(t):
    # gen keeps appending a distinct reply; the condition routes back while the
    # message history is short, else to `done`. No counter channel involved.
    gen = AgentNode(name="gen", llm=t.FakeLLM(
        responses=[t.ai("g0"), t.ai("g1"), t.ai("g2"), t.ai("g3"), t.ai("g4")],
        repeat=True))
    loop = ConditionNode(name="loop",
                         condition=lambda s: "again" if len(s["messages"]) < 4 else "stop",
                         choices=["again", "stop"], reads=["messages"])
    done = _agent(t, "done", "DONE")
    gen > loop
    loop["again"] > gen
    loop["stop"] > done
    g = AgenticGraph(start_node=gen, end_nodes=done)

    out = g.invoke(message="start")   # terminates (no recursion-limit blow-up)
    assert out["decision_loop"] == "stop"
    assert out["messages"][-1].content == "DONE"


# --- 3. wiring / route resolution (mirrors the DecisionNode route tests) -------

def _cond(value="x"):
    return ConditionNode(name="c", condition=lambda s: value, choices=["x", "y"])


def test_route_picks_correct_branch(t):
    c = _cond()
    x = _agent(t, "x", "X")
    y = _agent(t, "y", "Y")
    c["x"] > x
    c["y"] > y
    assert c.route({"decision_c": "x"}) == "x"
    assert c.route({"decision_c": "y"}) == "y"


def test_route_childless_choice_raises(t):
    c = _cond()
    with pytest.raises(ValueError, match="does not have a child"):
        c.route({"decision_c": "x"})


def test_route_unknown_choice_raises(t):
    c = _cond()
    c["x"] > _agent(t, "x", "X")
    with pytest.raises(ValueError, match="not found"):
        c.route({"decision_c": "bogus"})


def test_condition_out_of_choices_raises(t):
    # the predicate must return a declared choice.
    c = ConditionNode(name="c", condition=lambda s: "nope", choices=["x", "y"])
    with pytest.raises(ValueError, match="not in choices"):
        c({"messages": []})


# --- 4. reads availability validation -----------------------------------------

def test_reads_before_write_caught(t):
    # `gate` reads `result`, which a DOWNSTREAM node writes -> read-before-write.
    gate = ConditionNode(name="gate", condition=lambda s: "go",
                         choices=["go"], reads=["result"])
    prod = AgentNode(name="prod", llm=t.FakeLLM(responses=[t.ai("P")]),
                     output_field="result")
    gate["go"] > prod
    with pytest.raises(Exception) as ei:   # GraphValidationError
        AgenticGraph(start_node=gate, end_nodes=prod)
    assert "result" in str(ei.value)


def test_reads_produced_upstream_passes(t):
    # producer first, condition reads its output second -> valid.
    prod = AgentNode(name="prod", llm=t.FakeLLM(responses=[t.ai("P")]),
                     output_field="result")
    gate = ConditionNode(name="gate", condition=lambda s: "go",
                         choices=["go"], reads=["result"])
    done = _agent(t, "done", "D")
    prod > gate
    gate["go"] > done
    g = AgenticGraph(start_node=prod, end_nodes=done)
    assert g.validate().ok is True


# --- 5. coexists with DecisionNode (the RouterNode refactor stays green) -------

def test_coexists_with_decision_node(t):
    dec = DecisionNode(name="dec", llm=t.FakeLLM(structured_value="a"),
                       node_prompt="p", choices=["a", "b"])
    cond = ConditionNode(name="cond",
                         condition=lambda s: "left" if s["count"] < 5 else "right",
                         choices=["left", "right"], reads=["count"])
    la = _agent(t, "la", "LA")
    lb = _agent(t, "lb", "LB")
    bb = _agent(t, "bb", "BB")
    dec["a"] > cond
    dec["b"] > bb
    cond["left"] > la
    cond["right"] > lb
    g = AgenticGraph(start_node=dec, end_nodes=[la, lb, bb])

    out = g.invoke(message="x", count=3)   # dec -> a -> cond -> left -> la
    assert out["decision_cond"] == "left"  # cond writes its own decision_cond channel
    assert any(m.content == "LA" for m in out["messages"])
