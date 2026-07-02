"""Generated pipeline for task 'map_reduce_summ' (sample 4).

TASK: Summarize each of several documents independently in parallel, then reduce the per-document summaries into one combined summary.
"""

from typing import List
from pttai import AgentNode, AgenticGraph, fanout, AgenticState


def build_graph(llm):
    summarizer = AgentNode(
        name="summarizer",
        llm=llm,
        node_prompt=(
            "You summarize a single document.\n"
            "Task: Produce a concise summary (5-8 bullet points) capturing the key facts, "
            "arguments, and any important numbers.\n"
            "Document:\n{document}"
        ),
        reads=["document"],
        writes=["messages"],
    )

    reducer = AgentNode(
        name="reducer",
        llm=llm,
        node_prompt=(
            "You are reducing multiple per-document summaries into one combined summary.\n"
            "Given several summaries, write a single cohesive summary with:\n"
            "1) A 5-7 sentence overview\n"
            "2) A consolidated list of the top themes (5-10 bullets)\n"
            "3) Any cross-document agreements/disagreements (if present)\n"
            "Per-document summaries (in order):\n{messages}"
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # Pipeline:
    # - Run summarizer concurrently for each item in state["documents"] (map-reduce join at reducer)
    # - Reducer produces the final combined summary in state["messages"] (default AgenticState behavior)
    start = summarizer.map("documents") > reducer

    return AgenticGraph(start_node=start, end_nodes={reducer})
