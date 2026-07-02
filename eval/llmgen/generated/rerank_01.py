"""Generated pipeline for task 'rerank' (sample 1).

TASK: Given a query, retrieve candidate passages, rerank them by relevance, then answer the query using the top reranked passage.
"""

from pttai import AgentNode, AgenticGraph
from pttai.tools import make_retriever_tool


def build_graph(llm):
    # Tooling / RAG
    retriever_tool = make_retriever_tool(
        name="retriever",
        description="Retrieve candidate passages relevant to the user's query. Input should be the query string.",
    )

    # 1) Retrieve candidate passages
    retrieve = AgentNode(
        name="retrieve",
        llm=llm,
        tools=[retriever_tool],
        node_prompt=(
            "You are a retrieval assistant.\n"
            "Use the retriever tool to gather multiple candidate passages for the user's query.\n"
            "Goal: maximize recall. Return the passages you retrieved for later reranking.\n\n"
            "User query:\n{query}"
        ),
        reads=["query"],
        writes=["messages"],
    )

    # 2) Rerank by relevance (still LLM-based, using retrieved passages as context)
    rerank = AgentNode(
        name="rerank",
        llm=llm,
        node_prompt=(
            "You are a precise reranker.\n"
            "You will be given the user's query and a list of retrieved candidate passages.\n"
            "Rerank the passages by relevance to the query.\n\n"
            "Requirements:\n"
            "- Choose the single best passage.\n"
            "- Provide:\n"
            "  1) the index (starting at 1) of the best passage\n"
            "  2) the text of that best passage (verbatim or near-verbatim)\n"
            "  3) a short justification (1-2 sentences)\n\n"
            "User query:\n{query}"
        ),
        reads=["query", "messages"],
        writes=["messages"],
    )

    # 3) Answer using the top reranked passage
    answer = AgentNode(
        name="answer",
        llm=llm,
        node_prompt=(
            "You are a helpful answer assistant.\n"
            "Use ONLY the top reranked passage provided to answer the user's query.\n\n"
            "Instructions:\n"
            "- If the passage is insufficient to answer confidently, say so briefly.\n"
            "- Do not invent facts not present in the passage.\n"
            "- Provide a concise final answer.\n\n"
            "User query:\n{query}"
        ),
        reads=["query", "messages"],
        writes=["messages"],
    )

    # Wiring: retrieve -> rerank -> answer
    # Graph is schema-free by default; validator will ensure reads exist on the start inputs.
    return AgenticGraph(
        start_node=retrieve,
        end_nodes={answer},
    )
