"""Generated pipeline for task 'map_reduce_summ' (sample 0).

TASK: Summarize each of several documents independently in parallel, then reduce the per-document summaries into one combined summary.
"""

from pttai import AgentNode, AgenticGraph, fanout, AgenticState


def build_graph(llm):
    # Summarize one document independently (runs in parallel via fanout)
    summarize_one = AgentNode(
        name="summarize_one",
        llm=llm,
        reads=["messages"],
        node_prompt=(
            "You will be given ONE document in the conversation.\n\n"
            "Task: Summarize it in 5-8 concise bullet points that capture the key facts, "
            "arguments, and any important numbers. Avoid filler.\n\n"
            "Document to summarize is the most recent user message content."
        ),
        # Default writes to `messages` (appends assistant summary to the conversation)
        # We also seed `log`/`token` via AgenticState.
    )

    # Reduce step: combine all per-document summaries into one unified summary.
    reduce = AgentNode(
        name="reduce",
        llm=llm,
        reads=["messages"],
        node_prompt=(
            "You will receive multiple per-document summaries (as assistant messages) "
            "in the conversation.\n\n"
            "Task: Write ONE combined summary of the entire document set. "
            "Requirements:\n"
            "- Start with a 2-3 sentence overview.\n"
            "- Then provide 6-10 bullets of the highest-signal shared themes and differences.\n"
            "- If the summaries contain conflicting claims, note the discrepancy.\n"
            "- Keep it concise and specific."
        ),
    )

    # One-line topology: fan out N summarizers, then join at reducer.
    # NOTE: fanout(b, c) expects explicit nodes; to summarize 'several' documents,
    # we wire 3 parallel summarizers as a practical default.
    #
    # The caller should place each document into the conversation and invoke with
    # a messages state such that each parallel branch sees the intended document
    # as its most recent input. (e.g., separate graphs per document count, or
    # map-reduce variant if you extend this module.)
    s1 = summarize_one
    s2 = AgentNode(
        name="summarize_two",
        llm=llm,
        reads=["messages"],
        node_prompt=summarize_one.node_prompt,
    )
    s3 = AgentNode(
        name="summarize_three",
        llm=llm,
        reads=["messages"],
        node_prompt=summarize_one.node_prompt,
    )

    start = s1
    start > fanout(s1, s2, s3) > reduce

    return AgenticGraph(start_node=start, end_nodes={reduce})
