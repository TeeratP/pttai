"""HumanNode generalization: free-form resume value can land in any state key
(`into=`) for routing, and `show=` controls what's surfaced in the interrupt."""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from nae.graph import AgenticGraph
from nae.nodes import AgentNode, ConditionNode, HumanNode


def test_into_writes_raw_value_to_key(t):
    """into="approval" routes the resume value to the `approval` key, not messages."""
    a = AgentNode(name="a", llm=t.FakeLLM(responses=[t.ai("greeting")]), node_prompt="p")
    h = HumanNode(name="h", into="approval", n=0)
    a > h
    g = AgenticGraph(start_node=a, end_nodes=h, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t1"}}

    g.invoke({"messages": [t.ai("hi")], "log": []}, config=config)
    out = g.invoke(Command(resume="approve"), config=config)
    assert out["approval"] == "approve"


def test_approval_gate_flow(t):
    """Compose HumanNode(into=) with ConditionNode: human reply gates the route."""

    def build():
        agent = AgentNode(name="agent", node_prompt="p",
                          llm=t.FakeLLM(responses=[t.ai("draft")], repeat=True))
        gate = ConditionNode(name="gate",
                             condition=lambda s: "ship" if s["approval"] == "approve" else "redo",
                             choices=["ship", "redo"], reads=["approval"])
        ship = AgentNode(name="ship", node_prompt="p",
                         llm=t.FakeLLM(responses=[t.ai("shipped")]))
        agent > HumanNode(name="h", into="approval", n=0) > gate
        gate["ship"] > ship
        gate["redo"] > agent
        return AgenticGraph(start_node=agent, end_nodes=ship,
                            checkpointer=InMemorySaver())

    # approve -> routes to ship and ends there.
    g = build()
    cfg = {"configurable": {"thread_id": "ok"}}
    g.invoke({"messages": [t.ai("hi")], "log": []}, config=cfg)
    out = g.invoke(Command(resume="approve"), config=cfg)
    assert "gate:ship" in out["log"]
    assert "shipped" in [m.content for m in out["messages"]]

    # reject -> routes back through redo to agent (interrupts again, never ships).
    g = build()
    cfg = {"configurable": {"thread_id": "no"}}
    g.invoke({"messages": [t.ai("hi")], "log": []}, config=cfg)
    out = g.invoke(Command(resume="reject"), config=cfg)
    assert "gate:redo" in out["log"]
    assert "shipped" not in [m.content for m in out["messages"]]


def test_show_literal_surfaced_in_interrupt(t):
    """show="custom text" surfaces that literal in the interrupt payload."""
    a = AgentNode(name="a", llm=t.FakeLLM(responses=[t.ai("greeting")]), node_prompt="p")
    h = HumanNode(name="h", node_prompt="Review:", show="custom text")
    a > h
    g = AgenticGraph(start_node=a, end_nodes=h, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "s1"}}

    paused = g.invoke({"messages": [t.ai("hi")], "log": []}, config=config)
    assert paused["__interrupt__"][0].value == {"Review:": "custom text"}
