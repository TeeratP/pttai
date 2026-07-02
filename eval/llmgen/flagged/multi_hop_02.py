"""Generated pipeline for task 'multi_hop' (sample 2).

TASK: Answer a multi-hop question that needs two sequential lookups: first find an intermediate fact, then use it to look up and produce the final answer.
"""

from pttai import AgentNode, AgenticGraph
from pttai.tools import make_retriever_tool


def build_graph(llm):
    # Two sequential lookups:
    # 1) retrieve an intermediate fact
    # 2) retrieve the final answer using that intermediate fact
    #
    # This pipeline assumes `llm` can be passed tool-using nodes (AgentNode does the tool loop).
    # We use a generic retriever tool to keep the module self-contained.
    # If you have a specific retriever instance, replace `retriever` below accordingly.
    retriever = None  # replace with your retriever instance if available

    retrieve_intermediate = AgentNode(
        name="retrieve_intermediate",
        llm=llm,
        node_prompt=(
            "You are a research assistant. Use the retriever tool to find an intermediate fact "
            "needed to answer the user's question in the next step.\n\n"
            "User question:\n{question}\n\n"
            "Return the single best intermediate fact as plain text."
        ),
        tools=[make_retriever_tool(retriever)] if retriever is not None else None,
        reads=["question"],
        writes={"intermediate": str},
    )

    retrieve_final = AgentNode(
        name="retrieve_final",
        llm=llm,
        node_prompt=(
            "You are a research assistant. Use the retriever tool to answer the user's question "
            "using the previously found intermediate fact.\n\n"
            "User question:\n{question}\n\n"
            "Intermediate fact:\n{intermediate}\n\n"
            "Now produce the final answer as plain text."
        ),
        tools=[make_retriever_tool(retriever)] if retriever is not None else None,
        reads=["question", "intermediate"],
        writes={"answer": str},
    )

    return AgenticGraph(start_node=retrieve_intermediate, end_nodes={retrieve_final})
