"""Generated pipeline for task 'multi_hop' (sample 0).

TASK: Answer a multi-hop question that needs two sequential lookups: first find an intermediate fact, then use it to look up and produce the final answer.
"""

from pttai import AgentNode, AgenticGraph
from pttai.tools import make_retriever_tool


def build_graph(llm):
    # Assumes the graph state is schema-free by default (AgenticState).
    # The caller should provide:
    #   - question: str
    # Optional for better retrieval:
    #   - retriever_1: a LangChain retriever object (for hop-1)
    #   - retriever_2: a LangChain retriever object (for hop-2)
    #
    # If retriever_1 / retriever_2 are not provided, you can still run by wiring
    # your own retriever tools inside this module. Here we use tool objects
    # when retrievers exist in state by delegating tool selection to the LLM agent.
    #
    # Note: make_retriever_tool creates a tool bound to a specific retriever instance.
    # We'll create two agent nodes that each expect the appropriate retriever tool to exist.
    #
    # For strict "two sequential lookups", we force:
    #   hop1_retrieve (tool) -> hop1_extract -> hop2_retrieve (tool) -> answer

    # Hop 1: retrieve context to identify an intermediate fact.
    hop1_retrieve = AgentNode(
        name="hop1_retrieve",
        llm=llm,
        node_prompt=(
            "You must perform LOOKUP #1 to find an intermediate fact needed to answer the question.\n"
            "Use the retriever tool available in state (if provided) to search.\n"
            "Question: {question}\n\n"
            "Goal: produce the most relevant evidence snippets that allow extraction of ONE intermediate fact."
        ),
        # The tool is optional at build time; if retriever_1 isn't supplied, the LLM can still reason
        # from provided messages, but for a real deployment you should pass retriever_1 in state.
        tools=[
            # Tool expects a retriever object at construction time; we provide a placeholder by
            # making retriever tools that operate on state-provided retrievers via the tool wrapper
            # created elsewhere. If your environment doesn't support that pattern, replace these
            # with concrete retrievers in your app code.
            make_retriever_tool(name="retriever_1", retriever_key="retriever_1")
        ],
        reads=["question"],
        writes=["messages"],
    )

    # Extract the intermediate fact from hop1 evidence.
    hop1_extract = AgentNode(
        name="hop1_extract",
        llm=llm,
        node_prompt=(
            "Extract the SINGLE intermediate fact needed for LOOKUP #2.\n"
            "Intermediate fact must be specific, atomic, and directly usable as a search query or key.\n"
            "If multiple candidates exist, pick the best one and explain internally (do not output analysis).\n\n"
            "Original question: {question}\n"
            "Use the evidence from the conversation to derive the intermediate fact."
        ),
        reads=["question"],
        writes=["messages"],
    )

    # Hop 2: retrieve final supporting evidence using the intermediate fact.
    hop2_retrieve = AgentNode(
        name="hop2_retrieve",
        llm=llm,
        node_prompt=(
            "You must perform LOOKUP #2 using the intermediate fact you extracted.\n"
            "Retrieve evidence required to answer the ORIGINAL question completely and accurately.\n\n"
            "Original question: {question}\n"
            "Intermediate fact (from prior step): use the latest extracted content in the conversation."
        ),
        tools=[
            make_retriever_tool(name="retriever_2", retriever_key="retriever_2")
        ],
        reads=["question"],
        writes=["messages"],
    )

    # Produce final answer grounded in hop2 evidence.
    answer = AgentNode(
        name="answer",
        llm=llm,
        node_prompt=(
            "Write the final answer to the user's question.\n"
            "Use ONLY evidence from LOOKUP #2 (latest retrieved context) and be specific.\n"
            "If the evidence is insufficient, say what is missing.\n\n"
            "Question: {question}\n"
            "Return a concise but complete answer."
        ),
        reads=["question"],
        writes=["messages"],
    )

    # Two sequential lookups enforced via wiring order:
    # hop1_retrieve -> hop1_extract -> hop2_retrieve -> answer
    hop1_retrieve > hop1_extract > hop2_retrieve > answer

    return AgenticGraph(start_node=hop1_retrieve, end_nodes={answer})
