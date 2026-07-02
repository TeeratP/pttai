"""Generated pipeline for task 'extract_summarize' (sample 1).

TASK: Extract the key entities and claims from a document, then write a short summary that uses the extracted information.
"""

from pttai import AgentNode, AgenticGraph, fanout


def build_graph(llm):
    extract_entities = AgentNode(
        name="extract_entities",
        llm=llm,
        node_prompt=(
            "Extract the key entities from the document.\n"
            "Return ONLY a concise bullet list of entities (names/organizations/people/places/concepts)."
        ),
        reads=["messages"],
        writes={"entities": str},
    )

    extract_claims = AgentNode(
        name="extract_claims",
        llm=llm,
        node_prompt=(
            "Extract the key claims from the document.\n"
            "Return ONLY concise numbered claims, preserving meaning and scope.\n"
            "If the document has uncertainties or opinions, reflect that in the claim text."
        ),
        reads=["messages"],
        writes={"claims": str},
    )

    write_summary = AgentNode(
        name="write_summary",
        llm=llm,
        node_prompt=(
            "Using ONLY the extracted entities and claims, write a short summary.\n"
            "Requirements:\n"
            "- 5-7 sentences\n"
            "- Mention the most important entities\n"
            "- Cover the core claims in a compact way\n"
            "- Do not introduce new facts not present in the claims\n\n"
            "Entities:\n{entities}\n\n"
            "Claims:\n{claims}"
        ),
        reads=["entities", "claims"],
        writes=["messages"],
    )

    # Run extraction in parallel, then summarize using both outputs.
    return AgenticGraph(
        start_node=fanout(extract_entities, extract_claims) > write_summary,
        end_nodes={write_summary},
    )
