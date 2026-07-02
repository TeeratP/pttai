"""Phase 4: node reasoning_effort, caching, retry, timeout, and durability passthrough."""

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from nae.graph import AgenticGraph
from nae.nodes import AgentNode
from nae.state import AgenticState


# --- Item 2: reasoning_effort -------------------------------------------------

def test_reasoning_effort_passed_as_invoke_kwarg(t):
    llm = t.FakeLLM(responses=[t.ai("ok")])
    node = AgentNode(name="a", llm=llm, node_prompt="p", reasoning_effort="low")
    node({"messages": [HumanMessage("hi")], "log": []})
    assert llm.last_kwargs.get("reasoning_effort") == "low"


def test_no_reasoning_effort_by_default(t):
    llm = t.FakeLLM(responses=[t.ai("ok")])
    node = AgentNode(name="a", llm=llm, node_prompt="p")
    node({"messages": [HumanMessage("hi")], "log": []})
    assert "reasoning_effort" not in llm.last_kwargs


# --- Item 3: cache / retry / timeout -----------------------------------------

def test_node_caching_skips_recompute(t):
    llm = t.FakeLLM(responses=[t.ai("r1")], repeat=True)
    node = AgentNode(name="a", llm=llm, node_prompt="p", cache_ttl=60)
    g = AgenticGraph(state=AgenticState, start_node=node, end_nodes=node)
    state = {"messages": [HumanMessage("same")], "log": []}
    g.invoke(state)
    g.invoke(state)
    assert llm.invoke_count == 1  # second run hit the node cache


def test_node_retry_recovers_from_transient_failure(t):
    llm = t.FakeLLM(responses=[t.ai("ok")], fail_times=1)
    node = AgentNode(name="a", llm=llm, node_prompt="p", retry=True)
    g = AgenticGraph(state=AgenticState, start_node=node, end_nodes=node)
    out = g.invoke({"messages": [HumanMessage("hi")], "log": []})
    assert out["messages"][-1].content == "ok"
    assert llm.invoke_count == 2  # failed once, retried, then succeeded


# --- Item 4: durability -------------------------------------------------------

def test_durability_passthrough(t):
    node = AgentNode(name="a", llm=t.FakeLLM(responses=[t.ai("done")]), node_prompt="p")
    g = AgenticGraph(state=AgenticState, start_node=node, end_nodes=node,
                     checkpointer=InMemorySaver())
    out = g.invoke({"messages": [HumanMessage("hi")], "log": []},
                   config={"configurable": {"thread_id": "d1"}}, durability="exit")
    assert out["messages"][-1].content == "done"
