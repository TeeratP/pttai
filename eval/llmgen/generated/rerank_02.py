"""Generated pipeline for task 'rerank' (sample 2).

TASK: Given a query, retrieve candidate passages, rerank them by relevance, then answer the query using the top reranked passage.
"""

from pttai import AgentNode, AgenticGraph, ConditionNode, fanout
from pttai.tools import make_retriever_tool


def build_graph(llm):
    # Note: Replace these with your actual retriever source.
    # The graph expects `retriever` to be available at runtime when invoking the graph.
    # You can either:
    #   1) Seed `inputs={"retriever": <your retriever>}` into graph.invoke(...)
    #   2) Wrap your retriever into a tool and pass it via external wiring.
    #
    # For a fully self-contained implementation, you'd need access to a concrete
    # retriever here—which isn't possible from the provided function signature alone.
    retriever_tool = make_retriever_tool(
        retriever="{retriever}",
        name="retrieve",
        description="Retrieve candidate passages relevant to the query."
    )

    query_frame = AgentNode(
        name="frame_query",
        llm=llm,
        node_prompt=(
            "You are a search query assistant.\n"
            "Rewrite the user's query into 1 concise search query that preserves the intent.\n"
            "Return only the rewritten query."
        ),
        reads=["query"],
        writes={"search_query": str},
        # No tools needed here.
    )

    retrieve = AgentNode(
        name="retrieve_candidates",
        llm=llm,
        node_prompt=(
            "Use the retrieval tool to fetch candidate passages relevant to:\n"
            "{search_query}\n\n"
            "Return a list of passages (each passage should include enough text to be usable)."
        ),
        tools=[retriever_tool],
        reads=["search_query", "query"],
        writes={"passages": list},
    )

    rerank_and_answer = AgentNode(
        name="rerank_and_answer",
        llm=llm,
        node_prompt=(
            "You are given a user query and a list of candidate passages.\n"
            "1) Rerank the passages by relevance to the query.\n"
            "2) Select the TOP 1 passage from the reranked results.\n"
            "3) Answer the query using ONLY the information from the top passage.\n"
            "If the top passage does not contain enough information, say you cannot answer from the provided passage.\n\n"
            "Return:\n"
            "- the selected top passage\n"
            "- a brief relevance score (0-1)\n"
            "- the final answer"
        ),
        reads=["query", "passages"],
        writes={"top_passage": str, "score": float, "answer": str},
    )

    # Optional guard: if retrieval produced no passages, produce a fallback answer.
    no_passages = AgentNode(
        name="no_passages_fallback",
        llm=llm,
        node_prompt=(
            "You are a helpful assistant.\n"
            "No relevant passages were retrieved.\n"
            "Answer the user's query as best as possible without retrieved evidence, "
            "and clearly state that the answer is not grounded in retrieved passages."
        ),
        reads=["query"],
        writes={"answer": str, "top_passage": str, "score": float},
    )

    has_passages = ConditionNode(
        name="has_passages",
        condition=lambda state: "yes" if (state.get("passages") and len(state["passages"]) > 0) else "no",
        choices=["yes", "no"],
        reads=["passages"],
    )

    query_frame > retrieve > has_passages
    has_passages["yes"] > rerank_and_answer
    has_passages["no"] > no_passages

    # End nodes are both possible terminal responders.
    return AgenticGraph(
        start_node=query_frame,
        end_nodes={rerank_and_answer, no_passages},
    )
