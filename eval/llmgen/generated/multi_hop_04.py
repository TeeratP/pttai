"""Generated pipeline for task 'multi_hop' (sample 4).

TASK: Answer a multi-hop question that needs two sequential lookups: first find an intermediate fact, then use it to look up and produce the final answer.
"""

from pttai import AgentNode, AgenticGraph
from pttai.tools import make_retriever_tool


def build_graph(llm):
    # Tools are optional but enable the required "two sequential lookups".
    # We define one retriever tool (for hop1) and reuse it for hop2.
    #
    # Note: make_retriever_tool is a wrapper; the actual retriever configuration
    # (vector store, etc.) is application-specific. If you have a retriever,
    # replace retriever=None with your own retriever instance in your codebase.
    #
    # This pipeline still works as an LLM-only multi-hop approach if tools
    # are removed/disabled; however the intended design is two sequential
    # lookups via tools.
    retriever_tool = make_retriever_tool(retriever=None, name="retriever", description=None)

    # Hop 1: find an intermediate fact needed for the final question.
    hop1 = AgentNode(
        name="hop1_intermediate",
        llm=llm,
        tools=[retriever_tool],
        node_prompt=(
            "You are a research assistant. The user will ask a question that requires "
            "MULTI-HOP reasoning (two sequential lookups). \n\n"
            "Step 1 (HOP 1): Identify the most important intermediate fact needed to "
            "answer the user's question. Then use the retriever tool to find and cite "
            "that intermediate fact.\n\n"
            "Return ONLY the intermediate fact as a concise statement, plus any "
            "retrieval-backed citations you used."
        ),
        # Use the conversation history for context.
        input_field="messages",
        writes={"intermediate_fact": str},
        reads=["messages"],
    )

    # Hop 2: use intermediate fact to look up the final answer.
    hop2 = AgentNode(
        name="hop2_final",
        llm=llm,
        tools=[retriever_tool],
        node_prompt=(
            "You are a research assistant continuing from a completed HOP 1.\n\n"
            "Step 2 (HOP 2): Using ONLY the provided intermediate fact, search with "
            "the retriever tool to find the information necessary to answer the "
            "original user question. Then produce the final answer.\n\n"
            "Constraints:\n"
            "- Use the intermediate fact as the key for the second lookup.\n"
            "- Include citations from the second lookup in your answer.\n"
            "- Provide a clear, final response to the user."
        ),
        input_field="messages",
        reads=["messages", "intermediate_fact"],
        writes=["messages"],
    )

    # Starter agent: ensures we have the question in state["messages"] and
    # adds any necessary framing for the two-hop research workflow.
    starter = AgentNode(
        name="starter",
        llm=llm,
        node_prompt=(
            "You will receive a user question. Prepare it for a two-hop research workflow.\n"
            "Do NOT answer yet. Just restate the question clearly for downstream agents."
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # Wire sequentially: starter -> hop1 -> hop2
    starter > hop1 > hop2

    return AgenticGraph(start_node=starter, end_nodes={hop2})
