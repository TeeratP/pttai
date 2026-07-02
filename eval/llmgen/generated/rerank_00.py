"""Generated pipeline for task 'rerank' (sample 0).

TASK: Given a query, retrieve candidate passages, rerank them by relevance, then answer the query using the top reranked passage.
"""

from pttai import AgentNode, DecisionNode, AgenticGraph
from pttai.tools import make_retriever_tool


def build_graph(llm):
    # Tools: retriever (built from state-provided docs retriever or a default hook).
    # The returned tool expects the query and uses the retriever configured at build time.
    # Here we rely on the caller to seed/attach an actual retriever via state or outer wiring.
    # If you want a concrete retriever, replace `retriever=None` with your retriever instance
    # and set it in this module.
    retriever_tool = make_retriever_tool(
        retriever=None,  # caller can wrap/override by swapping this module or by providing retriever in state
        name="retrieve_passages",
        description="Retrieve candidate passages relevant to the user's query."
    )

    # 1) Retrieve candidates
    retrieve = AgentNode(
        name="retrieve",
        llm=llm,
        node_prompt=(
            "You are a retrieval assistant. Use the tool to retrieve candidate passages for the given query.\n"
            "Return the retrieved passages in the conversation so the next step can rerank them.\n"
            "If the tool returns passages, include them clearly with identifiers."
        ),
        tools=[retriever_tool],
        reads=["messages"],
        writes=["messages"],
    )

    # 2) Rerank by relevance and pick top passage
    rerank = AgentNode(
        name="rerank",
        llm=llm,
        node_prompt=(
            "Rerank the provided candidate passages by relevance to the user's query.\n"
            "Then output ONLY the top passage (verbatim) that best answers the query.\n"
            "Include enough context from the passage to answer, but do not include multiple passages."
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # 3) Answer using top passage
    answer = AgentNode(
        name="answer",
        llm=llm,
        node_prompt=(
            "You are a helpful QA assistant.\n"
            "Use ONLY the top passage (the most recent passage content in the conversation) to answer the query.\n"
            "If the passage does not contain enough information, say you don't know.\n"
            "Provide a concise answer."
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # Optional: ensure we always proceed to answer after rerank.
    # (DecisionNode used as an explicit marker; single-choice keeps flow simple.)
    route_to_answer = DecisionNode(
        name="route_to_answer",
        llm=llm,
        node_prompt="Choose whether to answer now.",
        choices=["answer"],
        input_field="messages",
    )
    route_to_answer["answer"] > answer

    retrieve > rerank > route_to_answer

    return AgenticGraph(start_node=retrieve, end_nodes={answer})
