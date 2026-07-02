"""Sample (clean): RAG QA with a retriever tool. Should NOT be flagged."""
from pttai import AgentNode, AgenticGraph
from pttai.tools import make_retriever_tool
from langchain_core.documents import Document


class _Retriever:
    def invoke(self, query):
        return [Document(page_content="pttai compiles a >-wired DSL to a LangGraph StateGraph.")]


def build_graph(llm):
    search = make_retriever_tool(_Retriever(), name="search_docs",
                                 description="Search the knowledge base.")
    answer = AgentNode(name="answer", llm=llm, tools=[search],
                       node_prompt="Answer only from the search_docs results.")
    return AgenticGraph(start_node=answer, end_nodes={answer})
