"""Generated pipeline for task 'multi_hop' (sample 3).

TASK: Answer a multi-hop question that needs two sequential lookups: first find an intermediate fact, then use it to look up and produce the final answer.
"""

from pttai import AgentNode, AgenticGraph
from pttai.tools import make_retriever_tool


def build_graph(llm):
    """
    Multi-hop QA with two sequential retrieval lookups:
      1) find an intermediate fact (via retriever #1)
      2) use that fact to retrieve final supporting info (via retriever #2)
      3) produce the final answer (using both hops context)
    """

    # Retriever tools
    # (Assumes your retrievers are available in the environment where pttai.rag is configured.
    # If your setup uses different retriever instances, adjust accordingly.)
    hop1_tool = make_retriever_tool(
        retriever_name="hop1",  # expects a retriever with this name in your setup
        description="Find the intermediate fact required to answer the question (first hop).",
    )
    hop2_tool = make_retriever_tool(
        retriever_name="hop2",  # expects a retriever with this name in your setup
        description="Find final evidence needed once the intermediate fact is known (second hop).",
    )

    # 1) Extract a concrete intermediate fact, and retrieve evidence for it.
    find_intermediate = AgentNode(
        name="find_intermediate",
        llm=llm,
        node_prompt=(
            "You are doing multi-hop question answering.\n"
            "Task: Determine a specific intermediate fact needed to answer the user's question.\n"
            "Use the FIRST retriever tool to search for evidence supporting that intermediate fact.\n\n"
            "Return ONLY one scalar value in your final message: the intermediate fact "
            "(a short, specific statement)."
        ),
        tools=[hop1_tool],
        max_tool_iterations=25,
        writes={"intermediate_fact": str},
    )

    # 2) With the intermediate fact, retrieve final evidence for the final answer.
    retrieve_final = AgentNode(
        name="retrieve_final",
        llm=llm,
        node_prompt=(
            "You are doing the second hop of multi-hop QA.\n"
            "User question:\n{question}\n\n"
            "Intermediate fact (from hop 1):\n{intermediate_fact}\n\n"
            "Use the SECOND retriever tool to find final evidence that directly supports answering "
            "the user's question using the intermediate fact.\n\n"
            "Return ONLY supporting notes as a short message. Do not answer yet."
        ),
        tools=[hop2_tool],
        max_tool_iterations=25,
        reads=["question", "intermediate_fact"],
        writes={"final_notes": str},
    )

    # 3) Produce the final answer using both hops.
    answer = AgentNode(
        name="answer",
        llm=llm,
        node_prompt=(
            "Answer the user's multi-hop question.\n\n"
            "Question:\n{question}\n\n"
            "Intermediate fact used:\n{intermediate_fact}\n\n"
            "Final evidence / notes:\n{final_notes}\n\n"
            "Produce a concise final answer. If the evidence is insufficient, say so."
        ),
        reads=["question", "intermediate_fact", "final_notes"],
        writes=["messages"],  # append the final answer as the conversation message
    )

    # Wiring (two sequential lookups, then answer)
    start = find_intermediate
    start > retrieve_final > answer

    return AgenticGraph(start_node=start, end_nodes={answer})
