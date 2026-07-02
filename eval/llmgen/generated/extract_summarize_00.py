"""Generated pipeline for task 'extract_summarize' (sample 0).

TASK: Extract the key entities and claims from a document, then write a short summary that uses the extracted information.
"""

from pttai import AgentNode, AgenticGraph


def build_graph(llm):
    extract = AgentNode(
        name="extract",
        llm=llm,
        node_prompt=(
            "You are an information extraction assistant.\n\n"
            "Task: Extract the key entities and their key claims from the user's document.\n\n"
            "Output format (STRICT):\n"
            "Entities:\n"
            "- <Entity 1>: <very short description if present>\n"
            "- <Entity 2>: ...\n"
            "\n"
            "Claims:\n"
            "- <Claim 1> (attributed to <entity if stated>)\n"
            "- <Claim 2> (attributed to <entity if stated>)\n"
            "\n"
            "Notes:\n"
            "- Only extract what is supported by the document.\n"
            "- If no clear entities/claims exist, say 'None found.'"
        ),
        reads=["messages"],
        writes={"extraction": str},
    )

    summarize = AgentNode(
        name="summarize",
        llm=llm,
        node_prompt=(
            "You are a summarization assistant.\n\n"
            "Using the extracted Entities and Claims below, write a short summary "
            "(3-6 sentences) that accurately reflects the document.\n\n"
            "Requirements:\n"
            "- Mention the most important entities.\n"
            "- Explicitly incorporate the key claims.\n"
            "- Do not add facts not present in the extracted information.\n\n"
            "Extracted information:\n"
            "{extraction}"
        ),
        reads=["extraction"],
        writes={"summary": str},
    )

    route_to_end = extract > summarize

    return AgenticGraph(start_node=extract, end_nodes={summarize})
