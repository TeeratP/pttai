"""Generated pipeline for task 'map_reduce_summ' (sample 2).

TASK: Summarize each of several documents independently in parallel, then reduce the per-document summaries into one combined summary.
"""

from pttai import AgentNode, AgenticGraph, fanout


def build_graph(llm):
    # Parallel per-document summarization
    summarizer = AgentNode(
        name="summarize_doc",
        llm=llm,
        node_prompt=(
            "Summarize the following document clearly and concisely. "
            "Return ONLY the summary text.\n\n"
            "Document:\n{document}"
        ),
        reads=["document"],
        writes={"summary": str},
        # This node will be executed once per item when mapped.
    )

    # Reducer: combine many per-document summaries
    reducer = AgentNode(
        name="reduce_summaries",
        llm=llm,
        node_prompt=(
            "You will be given a list of per-document summaries.\n"
            "Combine them into ONE coherent combined summary:\n"
            "- Remove repetition\n"
            "- Preserve key facts and distinctions between documents\n"
            "- Keep it concise but complete\n\n"
            "Per-document summaries:\n{summaries}"
        ),
        reads=["summaries"],
        writes={"combined_summary": str},
    )

    # Note: validate uses declared reads/writes; the map join should provide
    # `summaries` as a list of the per-doc summary strings.
    graph = AgenticGraph(
        start_node=summarizer.map("document") > reducer,
        end_nodes={reducer},
    )
    return graph
