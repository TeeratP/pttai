"""Generated pipeline for task 'multi_hop' (sample 1).

TASK: Answer a multi-hop question that needs two sequential lookups: first find an intermediate fact, then use it to look up and produce the final answer.
"""

from pttai import AgentNode, AgenticGraph
from pttai.tools import make_retriever_tool


def build_graph(llm):
    retriever_tool_1 = make_retriever_tool(
        retriever=None, tool_name="retriever_1", tool_description="Retrieve facts to identify an intermediate fact for the question."
    )
    retriever_tool_2 = make_retriever_tool(
        retriever=None, tool_name="retriever_2", tool_description="Retrieve evidence using the intermediate fact to answer the original question."
    )

    # 1) Find an intermediate fact.
    find_intermediate = AgentNode(
        name="find_intermediate",
        llm=llm,
        tools=[retriever_tool_1],
        reads=["messages"],
        writes=["messages"],
        node_prompt=(
            "You are a research agent. "
            "Your job is to answer a multi-hop question by first finding ONE intermediate fact "
            "that will be needed to complete the next lookup. "
            "\n\nWorkflow:\n"
            "1) Restate the user's question briefly.\n"
            "2) Identify what intermediate fact would unlock the final answer (e.g., a key entity, date, definition, or parameter).\n"
            "3) Use the provided retriever tool to find that intermediate fact.\n"
            "4) Conclude with a single line in the format:\n"
            "INTERMEDIATE_FACT: <the fact you will use for the second lookup>\n"
            "5) Do not answer the final question yet; stop after producing INTERMEDIATE_FACT."
        ),
    )

    # 2) Use intermediate fact to do second lookup and produce final answer.
    answer_final = AgentNode(
        name="answer_final",
        llm=llm,
        tools=[retriever_tool_2],
        reads=["messages"],
        writes=["messages"],
        node_prompt=(
            "You are a research agent continuing a two-step lookup.\n"
            "Use the previously produced INTERMEDIATE_FACT to guide your second retrieval.\n"
            "\nWorkflow:\n"
            "1) Extract INTERMEDIATE_FACT from the conversation.\n"
            "2) Use the provided retriever tool to retrieve evidence needed to answer the original question.\n"
            "3) Provide the final answer to the user's original question.\n"
            "4) Include a short justification citing what you retrieved (no need for perfect citations, just reference the evidence)."
        ),
    )

    return AgenticGraph(start_node=find_intermediate, end_nodes={answer_final})
