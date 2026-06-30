"""HumanNode interrupts the run and resumes via Command(resume=...) with a checkpointer."""

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from pttai.graph import AgenticGraph
from pttai.nodes import AgentNode, HumanNode
from pttai.state import AgenticState


def test_interrupt_then_resume(t):
    a = AgentNode(name="a", llm=t.FakeLLM(responses=[t.ai("greeting")]), node_prompt="p")
    inp = HumanNode(name="inp", node_prompt="Your input:", n=0)
    b = AgentNode(name="b", llm=t.FakeLLM(responses=[t.ai("done")]), node_prompt="p")
    a > inp > b
    g = AgenticGraph(state=AgenticState, start_node=a, end_nodes=b,
                     checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "test-1"}}

    paused = g.invoke({"messages": [HumanMessage(content="hi")], "log": []}, config=config)
    # Halted at the HumanNode: node `b` ("done") has not run yet, and the
    # checkpoint still has pending work.
    assert [m.content for m in paused["messages"]] == ["hi", "greeting"]
    assert g.compiled_graph.get_state(config).next  # pending task to resume

    final = g.invoke(Command(resume="my answer"), config=config)
    contents = [m.content for m in final["messages"]]
    assert "my answer" in contents      # human reply was appended
    assert contents[-1] == "done"       # continued to node b
