"""
RAG (retrieval-augmented generation) tooling for the Agentic Framework.

`make_retriever_tool` wraps any LangChain retriever (anything exposing
`.invoke(query)` returning documents) as a StructuredTool that an AgentNode can
use via `AgentNode(tools=[tool])`. `ChromaRAG` is a thin convenience around a
Chroma vector store; its import of `langchain_chroma` is lazy so this module
imports fine without the optional `rag` dependency installed.
"""

from langchain_core.tools import StructuredTool


def _format_docs(docs) -> str:
    return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)


def make_retriever_tool(retriever: object,
                        name: str = "search_documents",
                        description: str = "Search the knowledge base for relevant context.") -> StructuredTool:
    """Wrap a LangChain retriever as a StructuredTool bindable to an AgentNode.

    Args:
        retriever: Any object with `.invoke(query)` returning a list of documents.
        name: Tool name exposed to the model.
        description: Tool description exposed to the model.
    """

    def search(query: str) -> str:
        return _format_docs(retriever.invoke(query))

    return StructuredTool.from_function(func=search, name=name, description=description)


class ChromaRAG:
    """Convenience wrapper around a Chroma vector store.

    Requires the optional `rag` extra: `pip install -e .[rag]`.
    """

    def __init__(self, embeddings, collection_name: str = "nae", persist_directory=None):
        from langchain_chroma import Chroma  # lazy import: optional dependency
        self.store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )

    def add_texts(self, texts, metadatas=None):
        return self.store.add_texts(texts, metadatas=metadatas)

    def add_documents(self, documents):
        return self.store.add_documents(documents)

    def as_retriever(self, **kwargs):
        return self.store.as_retriever(**kwargs)

    def as_tool(self,
                name: str = "search_documents",
                description: str = "Search the knowledge base for relevant context.",
                **retriever_kwargs) -> StructuredTool:
        return make_retriever_tool(
            self.as_retriever(**retriever_kwargs), name=name, description=description
        )
