"""Generated pipeline for task 'rag_qa' (sample 4).

TASK: Answer a user question grounded in a small document corpus: retrieve relevant passages with a search tool, then answer using only those passages.
"""

from pttai import AgentNode, AgenticGraph
from pttai.tools import make_retriever_tool


def build_graph(llm):
    # Tool: the graph expects the caller to provide `retriever` in state
    # (via graph.invoke({"retriever": ..., "message": ...}) or equivalent).
    # make_retriever_tool can bind to that retriever at runtime.
    retrieve = make_retriever_tool(retriever_key="retriever", name="retrieve")

    # 1) Retrieve relevant passages
    retrieve_node = AgentNode(
        name="retrieve",
        llm=llm,
        tools=[retrieve],
        node_prompt=(
            "You are a search assistant. Use the `retrieve` tool to fetch relevant "
            "passages from the provided document corpus for the user's question. "
            "Then respond by writing ONLY the retrieved passages (verbatim excerpts) "
            "to `messages`."
        ),
    )

    # 2) Answer grounded strictly in retrieved passages
    answer_node = AgentNode(
        name="answer",
        llm=llm,
        node_prompt=(
            "Answer the user's question using ONLY the retrieved passages found in the "
            "conversation. Do not use outside knowledge.\n\n"
            "If the passages do not contain enough information to answer, say "
            "\"I don't know based on the provided passages.\"\n\n"
            "Question: {messages[-1]}\n"
            "Retrieved passages: (use only what is present in the conversation)."
        ),
        writes={"answer": str},
    )

    # Sequential: retrieve first, then answer from retrieved context
    retrieve_node > answer_node

    return AgenticGraph(start_node=retrieve_node, end_nodes={answer_node})
