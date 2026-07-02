"""Generated pipeline for task 'rag_qa' (sample 0).

TASK: Answer a user question grounded in a small document corpus: retrieve relevant passages with a search tool, then answer using only those passages.
"""

from pttai import AgentNode, AgenticGraph
from pttai.tools import make_retriever_tool


def build_graph(llm):
    # Tool-only contract:
    # - We retrieve from the caller-provided retriever (seeded via state or app code).
    # - Graph is reusable; the retriever can be passed in at invoke-time via `retriever=...`.
    #
    # Expected invoke state keys (seeded by the caller):
    # - "messages": conversation history (pttai default if using message= shorthand)
    # - "retriever": a LangChain retriever instance compatible with make_retriever_tool
    #
    # Output:
    # - "messages": assistant's grounded answer appended

    retrieve_tool = make_retriever_tool(
        retriever_key="retriever",
        tool_name="search_corpus",
        tool_description=(
            "Searches a small document corpus for passages relevant to the user's question. "
            "Use it to ground answers strictly in retrieved passages."
        ),
    )

    # 1) Retrieve relevant passages (tool-using AgentNode). We rely on tool output
    #    to be included in the model's context for the next step.
    retriever = AgentNode(
        name="retrieve",
        llm=llm,
        tools=[retrieve_tool],
        node_prompt=(
            "You are a retrieval assistant. "
            "Use the provided search tool to find the most relevant passages to answer the user's question. "
            "When you call the tool, provide the user's question as the search query. "
            "Collect enough passages to support a grounded answer."
        ),
        # Default reads/writes are on messages; tool calls happen inside this AgentNode.
        max_tool_iterations=8,
    )

    # 2) Answer grounded only in retrieved passages. No external knowledge.
    answerer = AgentNode(
        name="answer",
        llm=llm,
        node_prompt=(
            "You are an expert assistant. Answer the user's question using ONLY the retrieved passages. "
            "Do not use any outside knowledge. "
            "If the passages do not contain enough information, say you cannot find the answer in the corpus. "
            "Cite by quoting brief snippets from the retrieved passages where possible."
        ),
        max_tool_iterations=1,
    )

    # Sequential: retrieve first, then answer strictly from retrieved content.
    start = retriever > answerer
    return AgenticGraph(start_node=start, end_nodes={answerer})
