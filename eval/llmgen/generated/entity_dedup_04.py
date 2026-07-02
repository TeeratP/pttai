"""Generated pipeline for task 'entity_dedup' (sample 4).

TASK: Extract the named entities from several documents in parallel, then reconcile and deduplicate them into one canonical list.
"""

from pttai import AgentNode, AgenticGraph, fanout


def build_graph(llm):
    extract_one = AgentNode(
        name="extract_one",
        llm=llm,
        reads=["messages"],
        node_prompt=(
            "You are an information extraction system.\n"
            "Extract named entities from the provided document(s).\n\n"
            "Requirements:\n"
            "1) Focus on named entities (people, organizations, locations, products, "
            "works, events, and other proper nouns).\n"
            "2) Output a deduplicated list of entities as JSON.\n"
            "3) For each entity, include: {\"text\": <surface form>, \"type\": <one of "
            "PERSON|ORG|LOC|PRODUCT|WORK|EVENT|OTHER> }.\n"
            "4) Keep entities faithful to the text; do not add guesses.\n"
            "5) If the input is multiple documents, extract from all of them.\n"
            "6) Return ONLY JSON (no markdown, no commentary).\n\n"
            "Document content is provided in the conversation history."
        ),
        writes={"entities": list},
    )

    # In pttai AgentNode doesn't guarantee exact structured types unless the tool/structured output
    # is configured; we therefore reconcile using an LLM in a deterministic-ish single pass.
    # The output will be a canonical JSON list of entities.
    reconcile = AgentNode(
        name="reconcile",
        llm=llm,
        reads=["messages"],
        node_prompt=(
            "You are an entity reconciliation and deduplication system.\n"
            "You will receive multiple entity lists extracted from multiple documents.\n\n"
            "Task:\n"
            "1) Merge all entities.\n"
            "2) Deduplicate entities that refer to the same real-world entity.\n"
            "   - Use string normalization, common aliases, and context.\n"
            "   - Prefer canonical surface form when possible.\n"
            "3) Preserve entity type when reliable; otherwise use OTHER.\n"
            "4) Output ONLY JSON with a single key: "
            "{\"canonical_entities\": [ {\"text\": ..., \"type\": ...}, ... ] }\n"
            "5) canonical_entities must be deduplicated and stable (sort by text).\n"
            "6) Do not include commentary or extra keys.\n\n"
            "All extracted lists are included in the conversation history."
        ),
        writes={"canonical_entities": list},
    )

    # Run multiple extraction branches in parallel over the same input (messages),
    # then reconcile at the end.
    extractor_panel = fanout(extract_one, extract_one, extract_one)
    panel = extractor_panel > reconcile

    return AgenticGraph(start_node=extract_one, end_nodes={reconcile})
