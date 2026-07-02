"""Generated pipeline for task 'map_reduce_summ' (sample 3).

TASK: Summarize each of several documents independently in parallel, then reduce the per-document summaries into one combined summary.
"""

from pttai import AgentNode, AgenticGraph, fanout


def build_graph(llm):
    # Map inputs: provide N documents as a list of strings in `documents`.
    # Map-style worker summarizes each document into `messages` using the input `messages`
    # history; we keep it simple by having each worker receive the current doc as its
    # scalar read and write to `messages` (default).
    doc_summarizer = AgentNode(
        name="doc_summarizer",
        llm=llm,
        reads=["document"],
        node_prompt=(
            "Summarize the following document concisely.\n\n"
            "Requirements:\n"
            "- 3 to 6 bullet points\n"
            "- Capture key facts, decisions, and any numeric details\n"
            "- Avoid speculation\n\n"
            "DOCUMENT:\n{document}"
        ),
        # For map workers, the reducer-friendly default is writing to `messages`.
        # We'll use the message content as the per-document summary payload.
        writes=["messages"],
    )

    # Reduce inputs: take all per-document summaries from `messages` and combine.
    reducer = AgentNode(
        name="reducer",
        llm=llm,
        reads=["messages"],
        node_prompt=(
            "You are given multiple per-document summaries. Combine them into ONE coherent "
            "combined summary.\n\n"
            "Requirements:\n"
            "- 1 paragraph overall synthesis + 5-10 bullet key takeaways\n"
            "- Remove redundancy\n"
            "- If summaries disagree, note the discrepancy briefly\n\n"
            "PER-DOCUMENT SUMMARIES (as messages):\n{messages}"
        ),
        writes=["messages"],
    )

    # The wiring: a "fanout" over a fixed set isn't suitable because the number of
    # documents is dynamic. In pttai, use a map-reduce pattern: worker.map("field") > collect.
    # Here, workers should run over each item in state["documents"], binding it to `document`.
    #
    # We express this as: (seed a state key) -> doc_summarizer.map("documents") -> reducer.
    # Since pttai's map API is part of node objects, we rely on the DSL behavior:
    #   doc_summarizer.map("documents") applies the worker to every item in state["documents"].
    #
    # The reducer is the join point.
    #
    # Note: `doc_summarizer` expects a scalar read `document`, which will be populated
    # per-mapped item.
    graph_start = doc_summarizer  # map will be applied in the next line

    # Map-reduce: summarize each document independently in parallel, then reduce.
    # (In pttai DSL, worker.map("field") > collect is the correct pattern.)
    start_to_map = graph_start.map("documents") > reducer

    # start_to_map is itself a node wiring expression; use doc_summarizer as start_node.
    return AgenticGraph(start_node=doc_summarizer, end_nodes={reducer})
