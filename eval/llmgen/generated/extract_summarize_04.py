"""Generated pipeline for task 'extract_summarize' (sample 4).

TASK: Extract the key entities and claims from a document, then write a short summary that uses the extracted information.
"""

from pttai import AgentNode, DecisionNode, AgenticGraph


def build_graph(llm):
    extract = AgentNode(
        name="extract",
        llm=llm,
        node_prompt=(
            "You are an information extraction engine.\n"
            "Given a document, extract:\n"
            "1) Key entities (people, organizations, products, places, concepts).\n"
            "2) Key claims (statements the document makes about those entities).\n"
            "3) Any relations between entities and claims (e.g., who claims what, about whom).\n"
            "Output MUST be short and structured as bullet lists with the exact headings:\n"
            "Entities:\n"
            "- ...\n"
            "Claims:\n"
            "- ...\n"
            "Relations:\n"
            "- ...\n"
            "\n"
            "Document:\n"
            "{messages}"
        ),
        writes=["messages"],
    )

    maybe_empty = DecisionNode(
        name="maybe_empty",
        llm=llm,
        node_prompt=(
            "Decide whether the extracted content contains at least one entity and one claim.\n"
            "If the extraction looks empty or missing, choose 'incomplete'. Otherwise choose 'complete'."
        ),
        choices=["complete", "incomplete"],
        input_field="messages",
        tools=None,
    )

    revise = AgentNode(
        name="revise",
        llm=llm,
        node_prompt=(
            "The previous extraction seems incomplete. Re-extract key entities, claims, and relations.\n"
            "Follow the same output format and headings EXACTLY:\n"
            "Entities:\n"
            "- ...\n"
            "Claims:\n"
            "- ...\n"
            "Relations:\n"
            "- ...\n\n"
            "Document (again):\n"
            "{messages}"
        ),
        writes=["messages"],
    )

    summarize = AgentNode(
        name="summarize",
        llm=llm,
        node_prompt=(
            "Write a short summary (3-6 sentences) that uses the extracted information.\n"
            "Requirements:\n"
            "- Mention the most important entities.\n"
            "- Include the most important claims.\n"
            "- Reflect any relations between entities and claims.\n"
            "- Do not introduce new facts not supported by the extracted content.\n\n"
            "Extracted content:\n"
            "{messages}"
        ),
        writes=["messages"],
    )

    extract > maybe_empty
    maybe_empty["complete"] > summarize
    maybe_empty["incomplete"] > revise > summarize

    return AgenticGraph(start_node=extract, end_nodes={summarize})
