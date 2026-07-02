"""RAG QA — retrieve over a small corpus, then answer grounded in the passages.

The flagship NLP pipeline: a retriever tool searches a tiny knowledge base and
the ``AgentNode`` runs the retrieve -> answer loop internally (call model, run
the retriever tool, feed the passages back, answer grounded in them).

We use nae's real ``make_retriever_tool`` (from ``nae.tools``) wrapping a
tiny in-memory keyword retriever. ``ChromaRAG`` is the batteries-included path,
but ``langchain_chroma`` is an optional extra that is not installed in the
offline test env, so the retriever here is a dependency-free stand-in exposing
the same ``.invoke(query) -> [Document]`` contract ``make_retriever_tool`` wants.
Set ``OPENAI_API_KEY`` (and ``pip install -e .[rag]``) to swap in real embeddings.

    python examples/nlp/rag_qa.py
"""

import os
import sys

_EX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # examples/
sys.path.insert(0, os.path.dirname(_EX))  # repo root -> `import nae` works from a bare clone
sys.path.insert(0, _EX)  # -> `from _llm import get_llm`
from _llm import get_llm

from langchain_core.documents import Document

# A tiny knowledge base. In a real pipeline these would be chunks in a vector DB.
CORPUS = [
    "The nae library compiles a `>`-wired DSL down to a native LangGraph StateGraph.",
    "An AgentNode runs the model<->tool loop internally until the model returns a final answer.",
    "A DecisionNode uses constrained structured output so the model must return one of its choices.",
    "The build-time validator statically checks that every state key a node reads is produced upstream.",
    "LangGraph supports streaming, async, checkpointers, and human-in-the-loop interrupts.",
]


class KeywordRetriever:
    """Dependency-free retriever: rank corpus docs by word overlap with the query.

    Exposes the ``.invoke(query) -> [Document]`` contract that
    ``make_retriever_tool`` (and any LangChain retriever) uses.
    """

    def __init__(self, corpus, k: int = 2):
        self.corpus = corpus
        self.k = k

    def invoke(self, query: str):
        q = set(query.lower().split())
        ranked = sorted(self.corpus, key=lambda d: len(q & set(d.lower().split())), reverse=True)
        return [Document(page_content=d) for d in ranked[: self.k]]


def nae_version() -> str:
    from nae import AgentNode, AgenticGraph
    from nae.tools import make_retriever_tool

    search = make_retriever_tool(
        KeywordRetriever(CORPUS),
        name="search_docs",
        description="Search the nae knowledge base for passages relevant to the question.",
    )
    answer = AgentNode(
        llm=get_llm(),
        tools=[search],
        node_prompt="Answer the question ONLY from the search_docs results. Ground every claim in a retrieved passage.",
    )
    graph = AgenticGraph(start_node=answer, end_nodes={answer})
    return graph.invoke("What does an AgentNode do with tools?")["messages"][-1].content


# --- equivalent in raw LangGraph ---
def langgraph_version() -> str:
    from langgraph.graph import StateGraph, MessagesState, START
    from langgraph.prebuilt import ToolNode, tools_condition

    # The retriever tool is the same StructuredTool; nae's make_retriever_tool
    # is a thin wrapper, so raw LangGraph binds it exactly the same way.
    from nae.tools import make_retriever_tool
    search = make_retriever_tool(KeywordRetriever(CORPUS), name="search_docs",
                                 description="Search the nae knowledge base.")

    from langchain_core.messages import SystemMessage
    llm_with_tools = get_llm().bind_tools([search])
    system = SystemMessage("Answer the question ONLY from the search_docs results.")

    def call_model(state: MessagesState):
        return {"messages": [llm_with_tools.invoke([system] + state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", ToolNode([search]))
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", tools_condition)  # tools? -> tools : END
    builder.add_edge("tools", "call_model")                       # loop back
    graph = builder.compile()

    result = graph.invoke({"messages": [{"role": "user", "content": "What does an AgentNode do with tools?"}]})
    return result["messages"][-1].content


if __name__ == "__main__":
    print("[nae]     ", nae_version())
    print("[langgraph] ", langgraph_version())
