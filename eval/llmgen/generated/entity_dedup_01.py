"""Generated pipeline for task 'entity_dedup' (sample 1).

TASK: Extract the named entities from several documents in parallel, then reconcile and deduplicate them into one canonical list.
"""

from typing import List, Dict, Any
from pttai import AgentNode, AgenticGraph, fanout


def build_graph(llm):
    extractor_prompt = (
        "You extract named entities from text.\n"
        "Task: Extract named entities mentioned in the given documents.\n\n"
        "Output requirements:\n"
        "- Return a JSON array of entity objects.\n"
        "- Each object MUST have keys: \"name\" (string), \"type\" (string), and \"evidence\" (string).\n"
        "- \"type\" must be one of: PERSON, ORG, GPE, LOCATION, PRODUCT, EVENT, DATE, WORK_OF_ART, LAW, OTHER.\n"
        "- Use the smallest faithful surface form for \"name\".\n"
        "- \"evidence\" must be a short exact quote (or near-quote) from the input text that supports the entity.\n\n"
        "Documents:\n{documents}"
    )

    reconciler_prompt = (
        "You reconcile multiple entity lists into one canonical deduplicated list.\n\n"
        "Input: Several parallel extractions. Each extraction is a JSON array of entity objects with keys:\n"
        "\"name\", \"type\", \"evidence\".\n\n"
        "Reconciliation rules:\n"
        "1) Deduplicate by canonical entity identity.\n"
        "   - Merge obvious spelling variants, casing differences, and common abbreviations.\n"
        "   - If 'name' differs but evidence indicates same entity, merge.\n"
        "2) Keep consistent 'type'. If uncertain, choose the most supported type by evidence.\n"
        "3) For each canonical entity, output an object with:\n"
        "   - \"canonical_name\" (string)\n"
        "   - \"type\" (string)\n"
        "   - \"aliases\" (array of strings; include merged surface forms)\n"
        "   - \"evidences\" (array of short quotes; include at least 1)\n"
        "4) The final output MUST be a JSON array of canonical entity objects.\n\n"
        "Extractions:\n{entity_lists}"
    )

    dedupe_safety_prompt = (
        "Final pass to ensure formatting and strict compliance.\n"
        "Convert the input canonical entity JSON into the exact final JSON array format required:\n"
        "[{\"canonical_name\": str, \"type\": str, \"aliases\": [str, ...], \"evidences\": [str, ...]}, ...]\n"
        "Remove any invalid entries. Ensure all fields exist.\n\n"
        "Input canonical entities (JSON):\n{canonical_json}"
    )

    # One extractor per shard of the input documents.
    shard_names = ["shard0", "shard1", "shard2"]

    extract_nodes = []
    for shard in shard_names:
        extract_nodes.append(
            AgentNode(
                name=f"extract_{shard}",
                llm=llm,
                reads=["documents_by_shard"],
                node_prompt=extractor_prompt.replace(
                    "{documents}", f"{{documents_by_shard[{shard}]}}"
                ),
                writes={"entities": list},
            )
        )

    # Reconcile into canonical list.
    reconcile = AgentNode(
        name="reconcile",
        llm=llm,
        reads=["entities_by_shard"],
        node_prompt=reconciler_prompt.replace(
            "{entity_lists}", "{entities_by_shard}"
        ),
        writes={"canonical_entities": list},
    )

    # Final validation / formatting.
    finalize = AgentNode(
        name="finalize",
        llm=llm,
        reads=["canonical_entities"],
        node_prompt=dedupe_safety_prompt.replace("{canonical_json}", "{canonical_entities}"),
        writes={"final_entities": list},
    )

    # Wire: parallel extraction -> reconcile -> finalize
    graph = fanout(*extract_nodes) > reconcile > finalize

    return AgenticGraph(start_node=extract_nodes[0], end_nodes={finalize})
