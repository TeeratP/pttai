"""Graph-level streaming and async passthroughs (sync nodes via LangGraph threadpool)."""

import asyncio

from langchain_core.messages import HumanMessage

from nae.graph import AgenticGraph
from nae.nodes import AgentNode
from nae.state import AgenticState


def _two_node(t):
    a = AgentNode(name="a", llm=t.FakeLLM(responses=[t.ai("r1")]), node_prompt="p")
    b = AgentNode(name="b", llm=t.FakeLLM(responses=[t.ai("r2")]), node_prompt="p")
    a > b
    return AgenticGraph(state=AgenticState, start_node=a, end_nodes=b)


def test_stream_yields_node_updates(t):
    g = _two_node(t)
    chunks = list(g.stream({"messages": [HumanMessage(content="hi")], "log": []}))
    seen = set().union(*(c.keys() for c in chunks))
    assert {"a", "b"}.issubset(seen)


def test_ainvoke_runs_to_completion(t):
    g = _two_node(t)
    out = asyncio.run(g.ainvoke({"messages": [HumanMessage(content="hi")], "log": []}))
    assert [m.content for m in out["messages"]] == ["hi", "r1", "r2"]


def test_astream_yields_node_updates(t):
    g = _two_node(t)

    async def collect():
        return [c async for c in g.astream({"messages": [HumanMessage(content="hi")], "log": []})]

    chunks = asyncio.run(collect())
    seen = set().union(*(c.keys() for c in chunks))
    assert {"a", "b"}.issubset(seen)
