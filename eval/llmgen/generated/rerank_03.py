"""Generated pipeline for task 'rerank' (sample 3).

TASK: Given a query, retrieve candidate passages, rerank them by relevance, then answer the query using the top reranked passage.
"""

from pttai import AgentNode, AgenticGraph, fanout
from pttai.tools import make_retriever_tool


def build_graph(llm):
    # --- Retriever tool ---
    # The retriever must be provided at runtime via invoke(config) or state;
    # make_retriever_tool wraps a LangChain retriever into a tool the AgentNode can call.
    # If you don't have a retriever available, replace this with your own retriever tool.
    retriever_tool = make_retriever_tool()

    # --- Steps ---
    retrieve = AgentNode(
        name="retrieve",
        llm=llm,
        tools=[retriever_tool],
        node_prompt=(
            "You are a retrieval assistant. Use the retriever tool to find 8-12 candidate passages "
            "relevant to the user's query. Return the passages so they can be reranked."
        ),
        reads=["messages"],
        writes=["messages"],
    )

    rerank = AgentNode(
        name="rerank",
        llm=llm,
        node_prompt=(
            "Rerank the candidate passages by relevance to the user's query. "
            "Select the single best passage. "
            "Output a short message containing:\n"
            "1) the best passage text\n"
            "2) the reason it is most relevant\n"
            "Keep it concise."
        ),
        reads=["messages"],
        writes=["messages"],
    )

    answer = AgentNode(
        name="answer",
        llm=llm,
        node_prompt=(
            "Answer the user's query using ONLY the top reranked passage. "
            "If the passage does not contain enough information to answer, say you don't know. "
            "Cite the passage by quoting a brief supporting phrase."
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # Sequential: retrieve -> rerank -> answer
    retrieve > rerank > answer

    return AgenticGraph(start_node=retrieve, end_nodes={answer})
