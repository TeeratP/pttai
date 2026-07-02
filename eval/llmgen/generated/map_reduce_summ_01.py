"""Generated pipeline for task 'map_reduce_summ' (sample 1).

TASK: Summarize each of several documents independently in parallel, then reduce the per-document summaries into one combined summary.
"""

from pttai import AgentNode, AgenticGraph, fanout


def build_graph(llm):
    summarize_doc = AgentNode(
        name="summarize_doc",
        llm=llm,
        node_prompt=(
            "You are a careful summarizer. Summarize the following document.\n"
            "Return ONLY the summary (no headings, no bullet points unless essential)."
            "\n\nDocument:\n{doc}"
        ),
        reads=["doc"],
        writes=["messages"],
    )

    reduce_summaries = AgentNode(
        name="reduce_summaries",
        llm=llm,
        node_prompt=(
            "You are a synthesis editor. You will be given multiple document summaries.\n"
            "Combine them into one unified combined summary.\n"
            "Keep it concise, remove duplication, and preserve the most important points.\n\n"
            "Per-document summaries:\n{messages}"
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # Input contract:
    # - state must include: docs: list[str] (each item becomes {doc} for a parallel summarizer)
    # - graph starts from a node fed by map over docs
    # Wiring:
    # summarize_doc is mapped over state["docs"], then reduce_summaries combines them.
    start = summarize_doc.map("docs") > reduce_summaries

    return AgenticGraph(start_node=summarize_doc, end_nodes={reduce_summaries})
