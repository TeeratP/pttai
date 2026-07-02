"""End-to-end: delta-returning nodes compose correctly under the state reducers."""

from langchain_core.messages import HumanMessage

from pttai.graph import AgenticGraph
from pttai.nodes import AgentNode, DecisionNode
from pttai.state import AgenticState


def test_linear_invoke_merges_messages_and_log(t):
    a1 = AgentNode(name="a1", llm=t.FakeLLM(responses=[t.ai("r1")]), node_prompt="p")
    a2 = AgentNode(name="a2", llm=t.FakeLLM(responses=[t.ai("r2")]), node_prompt="p")
    a1 > a2
    g = AgenticGraph(state=AgenticState, start_node=a1, end_nodes=a2)

    out = g.invoke({"messages": [HumanMessage(content="hi")], "log": []})

    assert [m.content for m in out["messages"]] == ["hi", "r1", "r2"]
    assert out["log"] == ["a1:r1", "a2:r2"]


def test_decision_routed_invoke(t):
    a = AgentNode(name="a", llm=t.FakeLLM(responses=[t.ai("intro")]), node_prompt="p")
    d = DecisionNode(name="d", llm=t.FakeLLM(structured_value="positive"),
                     node_prompt="p", choices=["positive", "negative"])
    pos = AgentNode(name="pos", llm=t.FakeLLM(responses=[t.ai("happy")]), node_prompt="p")
    neg = AgentNode(name="neg", llm=t.FakeLLM(responses=[t.ai("sad")]), node_prompt="p")
    a > d
    d["positive"] > pos
    d["negative"] > neg
    g = AgenticGraph(state=AgenticState, start_node=a, end_nodes={pos, neg})

    out = g.invoke({"messages": [HumanMessage(content="hi")], "log": []})

    # routed to the positive handler; decision label is NOT in messages
    assert [m.content for m in out["messages"]] == ["hi", "intro", "happy"]
    assert out["decision_d"] == "positive"
    assert "d:positive" in out["log"]
