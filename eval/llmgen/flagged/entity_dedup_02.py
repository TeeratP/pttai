"""Generated pipeline for task 'entity_dedup' (sample 2).

TASK: Extract the named entities from several documents in parallel, then reconcile and deduplicate them into one canonical list.
"""

from pttai import AgentNode, AgenticGraph, fanout


def build_graph(llm):
    # 1) Extract named entities per document (runs in parallel via fanout)
    extractor_prompt = (
        "You are an information extraction engine.\n"
        "Extract named entities from the provided document.\n\n"
        "Return the entities as JSON in the reply content ONLY with this schema:\n"
        "{\n"
        '  "entities": [\n'
        '    {"name": string, "type": string, "evidence": string}\n'
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Use type labels like Person, Organization, Location, Date, Event, Product, Work, Other.\n"
        "- evidence must be a short exact span copied from the document.\n"
        "- Deduplicate within this document by (name,type,evidence).\n"
        "- If nothing is found, return {\"entities\": []}.\n"
    )

    extract_1 = AgentNode(
        name="extract_doc_1",
        llm=llm,
        node_prompt=extractor_prompt + "DOCUMENT:\n{document_1}",
        reads=["document_1"],
        writes="messages",
    )
    extract_2 = AgentNode(
        name="extract_doc_2",
        llm=llm,
        node_prompt=extractor_prompt + "DOCUMENT:\n{document_2}",
        reads=["document_2"],
        writes="messages",
    )
    extract_3 = AgentNode(
        name="extract_doc_3",
        llm=llm,
        node_prompt=extractor_prompt + "DOCUMENT:\n{document_3}",
        reads=["document_3"],
        writes="messages",
    )

    # 2) Reconcile + deduplicate into canonical entities
    #    (joined after fanout; one final canonical list)
    canonical_prompt = (
        "You are an entity resolution and canonicalization engine.\n"
        "You will be given multiple extraction outputs (JSON text) from different documents.\n\n"
        "Task:\n"
        "- Reconcile entities that refer to the same real-world entity.\n"
        "- Deduplicate into ONE canonical list.\n\n"
        "Return JSON ONLY in the reply content with this schema:\n"
        "{\n"
        '  "canonical_entities": [\n'
        "    {\n"
        '      "canonical_name": string,\n'
        '      "type": string,\n'
        '      "aliases": [string, ...],\n'
        '      "sources": [string, ...]\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- sources should identify which document(s) contributed (e.g., [\"document_1\",\"document_2\"]).\n"
        "- aliases should include the different observed surface forms across inputs.\n"
        "- Keep canonical_name stable: prefer the most complete/standard form you see.\n"
        "- If types conflict, choose the most consistent one; reflect uncertainty by picking the broader/most general appropriate type.\n"
    )

    reconcile = AgentNode(
        name="reconcile_and_deduplicate",
        llm=llm,
        node_prompt=canonical_prompt
        + "EXTRACTION OUTPUTS (may include JSON text):\n{messages}",
        reads=["messages"],
        writes="messages",
    )

    # Parallel extraction, then single reconcile step.
    extract_1 > fanout(extract_2, extract_3)  # join happens at the next node
    (extract_1 > fanout(extract_2, extract_3) > reconcile)

    # Start at extract_doc_1 so it provides the initial branch; fanout adds extract_doc_2/3.
    return AgenticGraph(start_node=extract_1, end_nodes={reconcile})
