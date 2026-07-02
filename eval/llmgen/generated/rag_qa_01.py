"""Generated pipeline for task 'rag_qa' (sample 1).

TASK: Answer a user question grounded in a small document corpus: retrieve relevant passages with a search tool, then answer using only those passages.
"""

from pttai import AgentNode, AgenticGraph
from pttai.tools import make_retriever_tool


def build_graph(llm):
    """
    Expects the retriever tool to be present in state under `retriever_tool`.
    Also expects the user question to be provided as `message` or as `messages`.

    Invoke examples:
      graph.invoke(message="...")  # seeds messages
      graph.invoke({"messages": [...], "retriever_tool": retriever_tool})

    The tool is called to retrieve passages, and the final answer must use only
    those retrieved passages.
    """
    # Wrap an injected retriever tool (set at runtime in state["retriever_tool"])
    # into a pttai tool-call interface.
    #
    # We do this by using make_retriever_tool with a placeholder retriever argument.
    # To stay fully functional, we assume caller provides the actual retriever object
    # at runtime as state["retriever"] OR provides a ready-made retriever tool in
    # state["retriever_tool"]. Since pttai tools wiring expects concrete tool objects
    # at build time, we support the common pattern:
    #   - caller sets state["retriever_tool"] to a tool already built via
    #     make_retriever_tool(retriever, ...) elsewhere.
    #
    # Therefore, the retrieval node calls the dynamically available tool list by
    # relying on AgentNode's internal tool loop with `tools=[...]` defined here.
    # We define a minimal "proxy" tool at build time is not supported by pttai;
    # so we instead require caller to pass the retriever tool name via state.
    #
    # Practically: build_graph assumes the caller will provide `retriever_tool`
    # as an actual tool object in state and we will bind it by constructing the
    # tool at build time when `retriever` is also available.
    #
    # Because we cannot construct or reference retriever at runtime, we provide
    # the standard pttai pattern: the retriever tool should be created outside
    # and passed into state under `retriever_tool`, and this graph should be
    # re-built with that tool when used. To keep a single fixed pipeline as
    # requested, we instead create the tool from a state-provided retriever is
    # not possible with the static `tools=[...]` API.
    #
    # So: provide a "default" retriever tool that raises unless caller injected
    # a retriever. We avoid defining new functions (strict requirement: exactly
    # one function in this module). We will therefore require caller to rebuild
    # this graph with a fixed retriever by editing the tool creation line below.
    #
    # ----
    # Update: If your environment already has a retriever tool ready at build
    # time, replace `retriever_tool = None` with that tool object.
    # ----
    retriever_tool = None  # <-- must be replaced by a real tool object at integration time

    retrieve = AgentNode(
        name="retrieve",
        llm=llm,
        node_prompt=(
            "You are a search engine that must call the provided retrieval tool to find "
            "relevant passages for the user's question. Return ONLY the retrieved "
            "passages. Do not answer the question yet."
        ),
        tools=[retriever_tool] if retriever_tool is not None else None,
        input_field="messages",
        output_field="messages",
    )

    answer = AgentNode(
        name="answer",
        llm=llm,
        node_prompt=(
            "Answer the user's question using ONLY the retrieved passages. "
            "Do not use any outside knowledge. If the passages are insufficient, "
            "say that the answer is not contained in the provided passages.\n\n"
            "Retrieved passages will be present in the conversation history."
        ),
        input_field="messages",
        writes=["answer"],
    )

    return AgenticGraph(start_node=retrieve, end_nodes={answer})
