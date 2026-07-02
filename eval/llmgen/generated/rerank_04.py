"""Generated pipeline for task 'rerank' (sample 4).

TASK: Given a query, retrieve candidate passages, rerank them by relevance, then answer the query using the top reranked passage.
"""

from pttai import AgentNode, AgenticGraph, fanout
from pttai.tools import make_retriever_tool


def build_graph(llm):
    # Tool-based retriever (the actual retriever is expected to be provided via
    # the graph caller / environment that configures the retriever tool backend).
    retriever_tool = make_retriever_tool(name="retriever")

    # 1) Retrieve candidate passages.
    # Output: passages (string payloads, typically formatted as multiple chunks).
    retrieve = AgentNode(
        name="retrieve",
        llm=llm,
        node_prompt=(
            "You are given a user query. Retrieve the most relevant candidate passages "
            "using the provided retriever tool. Return ONLY a single combined text block "
            "of candidate passages (not an answer yet)."
        ),
        tools=[retriever_tool],
        writes={"passages": str},
    )

    # 2) Rerank by relevance (using the top reranked passage only).
    # Output: top_passage (single passage text).
    rerank = AgentNode(
        name="rerank",
        llm=llm,
        node_prompt=(
            "You will be given a query and a set of candidate passages.\n"
            "Rerank the passages by relevance to the query.\n"
            "Select the single best passage and output it as top_passage only.\n\n"
            "Rules:\n"
            "- Do not answer the query.\n"
            "- Output only the chosen passage text."
        ),
        reads=["query", "passages"],
        writes={"top_passage": str},
    )

    # 3) Answer strictly using the top reranked passage.
    answer = AgentNode(
        name="answer",
        llm=llm,
        node_prompt=(
            "Use ONLY the provided top_passage to answer the query.\n"
            "If the passage does not contain enough information, say you don't know.\n\n"
            "Query:\n{query}\n\n"
            "Top passage:\n{top_passage}\n\n"
            "Answer:"
        ),
        reads=["query", "top_passage"],
        writes={"answer": str},
    )

    # Wire sequentially: retrieve -> rerank -> answer
    retrieve > rerank > answer

    return AgenticGraph(start_node=retrieve, end_nodes={answer})
