"""RAG retriever tool: formatting and binding to an AgentNode (no real vector store)."""

from langchain_core.messages import HumanMessage, ToolMessage

from pttai.nodes import AgentNode
from pttai.tools import make_retriever_tool


class _Doc:
    def __init__(self, page_content):
        self.page_content = page_content


class _Retriever:
    def __init__(self, docs):
        self._docs = docs

    def invoke(self, query):
        return self._docs


def test_retriever_tool_formats_docs():
    tool = make_retriever_tool(_Retriever([_Doc("alpha"), _Doc("beta")]),
                               name="kb", description="search kb")
    assert tool.name == "kb"
    out = tool.invoke({"query": "anything"})
    assert "alpha" in out and "beta" in out


def test_retriever_tool_binds_and_runs_in_agent(t):
    tool = make_retriever_tool(_Retriever([_Doc("ctx")]))
    node = AgentNode(name="agent", node_prompt="p", llm=t.FakeLLM(responses=[
        t.tool_call_msg("search_documents", {"query": "q"}),
        t.ai("answer"),
    ]), tools=[tool])
    delta = node({"messages": [HumanMessage(content="ask")], "log": []})

    tool_msgs = [m for m in delta["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs
    assert "ctx" in tool_msgs[0].content
