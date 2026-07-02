"""Generated pipeline for task 'entity_dedup' (sample 0).

TASK: Extract the named entities from several documents in parallel, then reconcile and deduplicate them into one canonical list.
"""

from pttai import AgentNode, DecisionNode, ConditionNode, AgenticGraph, fanout, AgenticState


def build_graph(llm):
    # 1) Frame: decide extraction strategy / output schema
    frame = DecisionNode(
        name="frame",
        llm=llm,
        node_prompt=(
            "You are the orchestration planner.\n"
            "Given the incoming documents in state['messages'], decide which extraction mode to use.\n"
            "Choose the closest option.\n\n"
            "Options:\n"
            "- 'standard': extract named entities (people, orgs, locations, dates, products, etc.) from the text.\n"
            "- 'sparse_docs': documents may be short/fragmentary; still extract what exists.\n"
            "- 'needs_rewrite': the text is ambiguous or poorly formatted; rewrite lightly before extracting.\n"
            "Output only one choice."
        ),
        choices=["standard", "sparse_docs", "needs_rewrite"],
        input_field="messages",
    )

    # 2) Extractors: run in parallel (multiple passes to improve recall)
    extractor_1 = AgentNode(
        name="extractor_1",
        llm=llm,
        reads=["messages"],
        writes={"entities": list},
        node_prompt=(
            "Extract named entities from the provided documents.\n"
            "Return a LIST named 'entities'. Each entity item MUST be a JSON object with:\n"
            "- 'text': the surface form as it appears\n"
            "- 'type': one of [PERSON, ORG, GPE, LOC, DATE, PRODUCT, EVENT, WORK, LAW, MONEY, PERCENT, OTHER]\n"
            "- 'context': a short snippet (max ~30 words) showing how it was used\n"
            "- 'confidence': 0.0 to 1.0 (your estimate)\n"
            "Rules:\n"
            "- Include entities even if repeated.\n"
            "- Do NOT attempt to canonicalize; only extract.\n"
            "- Prefer recall over precision.\n"
            "Documents:\n{messages}"
        ),
    )

    extractor_2 = AgentNode(
        name="extractor_2",
        llm=llm,
        reads=["messages"],
        writes={"entities": list},
        node_prompt=(
            "Second-pass named entity extraction for higher recall.\n"
            "Return a LIST named 'entities' using the exact schema described in extractor_1.\n\n"
            "Focus additionally on:\n"
            "- abbreviations/aliases of organizations\n"
            "- implied entities referenced by pronouns where unambiguous\n"
            "- product names and project/repo names if present\n\n"
            "Documents:\n{messages}"
        ),
    )

    extractor_3 = AgentNode(
        name="extractor_3",
        llm=llm,
        reads=["messages"],
        writes={"entities": list},
        node_prompt=(
            "Third-pass named entity extraction with a different lens: dates/metrics/legislation and events.\n"
            "Return a LIST named 'entities' using the exact schema described in extractor_1.\n\n"
            "Documents:\n{messages}"
        ),
    )

    # 3) Combine extracted lists into one pool
    aggregator = AgentNode(
        name="aggregator",
        llm=llm,
        reads=["entities"],
        writes={"pool": list},
        node_prompt=(
            "You will receive a combined view of entities extracted in parallel (state['entities']).\n"
            "Flatten into one list called 'pool' and keep items exactly as provided.\n\n"
            "If entities is nested, flatten to a single list of entity objects.\n"
            "Output: update state['pool'] with the flattened list."
        ),
    )

    # 4) Reconcile: canonicalize and deduplicate
    reconcile = AgentNode(
        name="reconcile",
        llm=llm,
        reads=["pool"],
        writes={"canonical_entities": list},
        node_prompt=(
            "Reconcile and deduplicate entities into one canonical list.\n"
            "Input: 'pool' is a list of entity objects with fields: text, type, context, confidence.\n\n"
            "Task:\n"
            "1) Group entities that refer to the same real-world entity (aliases, spelling variants, abbreviations).\n"
            "2) For each canonical entity, output a JSON object with:\n"
            "   - 'canonical_text': the chosen canonical label\n"
            "   - 'type': the final entity type\n"
            "   - 'aliases': list of other surface forms observed (including canonical_text if you wish)\n"
            "   - 'evidence': up to 3 context snippets from the pool\n"
            "   - 'confidence': a combined confidence (heuristic is fine)\n\n"
            "Dedup rules:\n"
            "- If two entities have different types (e.g., ORG vs PERSON) but share text, keep separate unless evidence strongly indicates otherwise.\n"
            "- Prefer the most common/most complete surface form as canonical_text.\n"
            "- Preserve date/metric entities as they appear unless clearly the same.\n\n"
            "Output ONLY the canonical list as state['canonical_entities'] update."
        ),
    )

    # 5) Optional: final cleanup (sort/stabilize)
    finalize = AgentNode(
        name="finalize",
        llm=llm,
        reads=["canonical_entities"],
        writes={"canonical_entities": list},
        node_prompt=(
            "Finalize the canonical entity list.\n"
            "- Remove near-duplicates within the canonical_entities list.\n"
            "- Sort entities deterministically by (type, canonical_text, then confidence desc).\n"
            "- Keep output format the same.\n\n"
            "Return updated state['canonical_entities']."
        ),
    )

    # Routing by frame choice
    # (All modes share the same extraction/reconciliation backbone; different extract prompts handle mode implicitly.)
    # To keep wiring valid and simple, we route into a shared parallel extraction stage regardless of choice.
    to_extract = DecisionNode(
        name="to_extract",
        llm=llm,
        node_prompt="All extraction modes run the same pipeline; route to extraction stage now. Choose 'go'.",
        choices=["go"],
        input_field="messages",
    )

    # Ensure each frame choice wires to to_extract so DecisionNode structural validation passes.
    frame["standard"] > to_extract
    frame["sparse_docs"] > to_extract
    frame["needs_rewrite"] > to_extract

    # Parallel extraction then join
    to_extract > fanout(extractor_1, extractor_2, extractor_3) > aggregator > reconcile > finalize

    # End node is finalize; graph state is schema-free by default.
    return AgenticGraph(start_node=frame, end_nodes={finalize})
