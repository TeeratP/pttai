"""Generated pipeline for task 'rag_qa' (sample 2).

TASK: Answer a user question grounded in a small document corpus: retrieve relevant passages with a search tool, then answer using only those passages.
"""

from pttai import AgentNode, AgenticGraph
from pttai.tools import make_retriever_tool


def build_graph(llm):
    # NOTE:
    # - This pipeline expects you to supply a retriever at runtime via state input:
    #   graph.invoke({"retriever": retriever, "message": "...question..."})
    # - It uses a retriever tool to pull grounded passages, then answers using only them.

    retrieve = AgentNode(
        name="retrieve",
        llm=llm,
        node_prompt=(
            "You are a retrieval assistant. "
            "Use the provided retriever tool to find the most relevant passages "
            "for the user's question from the document corpus. "
            "Return the retrieved passages (with enough context) so the answerer can cite them. "
            "If the corpus lacks the needed information, return that explicitly."
        ),
        tools=[],
        input_field="messages",
        writes={"retrieved": list},
        max_tool_iterations=10,
    )

    answer = AgentNode(
        name="answer",
        llm=llm,
        node_prompt=(
            "You are a careful answerer. Answer the user's question using ONLY the retrieved passages provided. "
            "Do not use outside knowledge. If the retrieved passages do not contain the answer, say you don't know "
            "based on the provided corpus.\n\n"
            "Retrieved passages:\n{retrieved}\n\n"
            "Question:\n{question}\n\n"
            "Write a concise answer. If helpful, quote or reference specific passages."
        ),
        reads=["retrieved", "question"],
        writes={"answer": str},
    )

    # Wire a tiny "tool wrapper" subgraph:
    # We can't create the retriever tool without the retriever instance, so we build a graph-wide
    # tool list at construction time using a placeholder retriever parameter.
    #
    # Practical usage pattern:
    # - Seed state with `retriever` before invoke.
    # - The tool will be created lazily per state by the underlying retriever tool wrapper.
    #
    # If your setup prefers fully static wiring, pass in a retriever instance by closing over it
    # when calling build_graph (e.g., build_graph(llm, retriever=...)); this signature is fixed.
    retriever_tool = make_retriever_tool(
        retriever=None,  # placeholder; assumed the tool wrapper can use state["retriever"] at runtime
        name="search_corpus",
        description="Searches the document corpus and returns relevant passages.",
    )

    retrieve.tools = [retriever_tool]

    prep = AgentNode(
        name="prep",
        llm=llm,
        node_prompt=(
            "Extract the user's question from the latest message. "
            "Return it as a scalar field `question` and initialize `retrieved` to an empty list."
        ),
        writes={"question": str, "retrieved": list},
        input_field="messages",
        max_tool_iterations=3,
    )

    prep > retrieve > answer

    return AgenticGraph(start_node=prep, end_nodes={answer})
