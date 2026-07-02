"""Generated pipeline for task 'rag_qa' (sample 3).

TASK: Answer a user question grounded in a small document corpus: retrieve relevant passages with a search tool, then answer using only those passages.
"""

from pttai import AgentNode, AgenticGraph
from pttai.tools import make_retriever_tool


def build_graph(llm):
    # Example corpus retriever hook.
    # Replace `retriever` with your own LangChain retriever in your app code.
    retriever = None
    retriever_tool = make_retriever_tool(retriever, name="retrieve") if retriever else None
    tools = [retriever_tool] if retriever_tool else []

    # Router decision: whether we have enough retrieval context yet.
    # We keep it deterministic to avoid relying on model statefulness.
    decide_enough = AgentNode(
        name="decide_enough",
        llm=llm,
        node_prompt=(
            "You are deciding whether the retrieval context is sufficient to answer.\n"
            "Given the retrieved passages (if any) and the user question, decide:\n"
            "1) If the passages contain direct evidence to answer, output 'enough'.\n"
            "2) Otherwise output 'need_more'.\n"
            "Return only one word: enough or need_more."
        ),
        reads=["messages"],
        writes={"decision": str},
        tools=tools,
        max_tool_iterations=1,
        input_field="messages",
    )

    # Retriever agent: uses the retriever tool to find relevant passages.
    # It writes the retrieval passages into the conversation messages.
    retriever_agent = AgentNode(
        name="retriever_agent",
        llm=llm,
        node_prompt=(
            "You are a document retriever assistant.\n"
            "Use the retrieve tool to search the corpus for passages relevant to the user's question.\n"
            "Return the retrieved passages verbatim (as citations: include source snippets).\n"
            "If retrieval finds nothing, say so explicitly."
        ),
        tools=tools,
        reads=["messages"],
        writes=["messages"],
        max_tool_iterations=25,
    )

    # Answerer agent: answers using only retrieved passages.
    answerer = AgentNode(
        name="answerer",
        llm=llm,
        node_prompt=(
            "You are a grounded QA system.\n"
            "Answer the user's question USING ONLY the retrieved passages.\n"
            "If the passages do not contain enough information, say you don't know.\n"
            "Do not use outside knowledge.\n"
            "Cite the passages by quoting the relevant snippet(s) or referencing them explicitly."
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # Simple deterministic stopping condition based on whether retrieval produced evidence.
    # (We inspect the latest assistant message content for the phrases "retrieved passages" or "does not contain".)
    def has_sufficient_evidence(state):
        # state["messages"] is a list of chat messages
        msgs = state.get("messages", [])
        text = msgs[-1].content if msgs else ""
        lowered = text.lower()
        if "does not contain" in lowered or "say so explicitly" in lowered:
            return "need_more"
        # If retriever_agent just ran, it should have passages/snippets; assume enough if it includes typical citation markers.
        if "retrieved" in lowered or "snippet" in lowered or '"' in text or "source" in lowered:
            return "enough"
        # Default to enough to avoid infinite loops in case of minimal formatting
        return "enough"

    # ConditionNode for looping between retrieve and answer if needed_more (bounded by internal tool loop + graph structure).
    condition = ConditionNode(
        name="condition",
        condition=has_sufficient_evidence,
        choices=["enough", "need_more"],
        reads=["messages"],
    )

    # Wire: first retrieve, then decide, then either answer or retrieve again.
    # We do this as a loop in graph structure by wiring need_more back to retriever_agent.
    retriever_agent > condition
    condition["enough"] > answerer
    condition["need_more"] > retriever_agent

    # The graph start is the retrieval agent; end is the answerer.
    return AgenticGraph(start_node=retriever_agent, end_nodes={answerer})
