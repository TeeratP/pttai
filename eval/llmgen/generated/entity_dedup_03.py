"""Generated pipeline for task 'entity_dedup' (sample 3).

TASK: Extract the named entities from several documents in parallel, then reconcile and deduplicate them into one canonical list.
"""

from pttai import AgentNode, AgenticGraph, fanout, AgenticState


def build_graph(llm):
    # 1) Frame: produce a single extraction prompt and required output schema.
    frame = AgentNode(
        name="frame",
        llm=llm,
        node_prompt=(
            "You will be given multiple documents. Extract named entities from each document.\n"
            "First, infer the user's entity extraction intent from the documents' topic.\n"
            "Then output a SPECIFICATION for extraction that the extractors must follow.\n\n"
            "Output requirements (follow exactly):\n"
            "- Output a JSON object with keys: \"entity_types\" (list of strings) and "
            "\"instructions\" (string).\n"
            "- entity_types should include the major NE categories likely present "
            "(e.g., PERSON, ORG, GPE, LOC, PRODUCT, EVENT, DATE, WORK_OF_ART, NORP, "
            "LAW, LANGUAGE, FAC, TITLES).\n"
            "- instructions must specify normalization rules (e.g., keep original form + "
            "a canonical form suggestion, merge obvious variants, avoid hallucinating, "
            "and never invent entities not present in text).\n"
            "- Be conservative: only extract entities explicitly supported by the text.\n\n"
            "Return ONLY the JSON."
        ),
        writes={"entity_spec": dict},
        reads=["documents"],
    )

    # 2) Extract per document in parallel (map-style parallel branches via fanout).
    #    We assume the caller seeds `documents` as a list of strings.
    #    The graph uses fanout with fixed number of branches; to support several docs,
    #    we request the caller to pass exactly N docs (common pattern in pttai examples).
    #
    #    If you need dynamic-length documents, you'd use .map(), but the `.map()` API
    #    is not in the authoritative cheatsheet for pttai public names. So we use
    #    a practical fixed-branch approach.
    #
    #    Choose a safe default for "several documents": 3.
    doc0_extract = AgentNode(
        name="doc0_extract",
        llm=llm,
        node_prompt=(
            "Extract named entities from DOCUMENT 0.\n\n"
            "Entity SPEC (JSON): {entity_spec}\n\n"
            "Document:\n{document}\n\n"
            "Return ONLY a JSON object with key \"entities\" as a list of objects, each:\n"
            "- \"mention\": the exact surface form from the text\n"
            "- \"type\": one of entity_types from SPEC\n"
            "- \"canonical\": your best canonical form suggestion\n"
            "- \"evidence\": a short quote/snippet from the text that supports the entity\n\n"
            "Do not include anything else besides the JSON."
        ),
        reads=["entity_spec", "document"],
        writes={"extracted": dict},
    )

    doc1_extract = AgentNode(
        name="doc1_extract",
        llm=llm,
        node_prompt=doc0_extract.node_prompt,
        reads=["entity_spec", "document"],
        writes={"extracted": dict},
    )

    doc2_extract = AgentNode(
        name="doc2_extract",
        llm=llm,
        node_prompt=doc0_extract.node_prompt,
        reads=["entity_spec", "document"],
        writes={"extracted": dict},
    )

    # 3) Reconcile: merge all extracted entities, deduplicate, and output canonical list.
    reconcile = AgentNode(
        name="reconcile",
        llm=llm,
        node_prompt=(
            "Reconcile and deduplicate named entities from multiple extraction results.\n\n"
            "Entity SPEC (JSON): {entity_spec}\n\n"
            "Extraction results are provided as JSON objects with key \"extracted\".\n"
            "Each \"extracted\" contains: {\"entities\": [ {mention,type,canonical,evidence}, ... ] }.\n\n"
            "Task:\n"
            "1) Merge duplicates across documents using canonical forms and entity_type.\n"
            "2) Combine evidence snippets for the merged canonical entity.\n"
            "3) If there is ambiguity (same mention maps to multiple entities), keep separate.\n"
            "4) Output a single JSON object with key \"canonical_entities\" as a list.\n"
            "Each canonical entity object must have:\n"
            "- \"canonical\": canonical form\n"
            "- \"type\": type\n"
            "- \"mentions\": list of all mentions merged\n"
            "- \"evidence\": list of short evidence snippets (no more than 5 per entity)\n"
            "5) Deterministically sort canonical_entities by (type, canonical).\n\n"
            "Return ONLY the JSON."
        ),
        reads=["entity_spec", "doc_extractions"],
        writes={"final": dict},
    )

    # Wire the extraction subgraph:
    # - We will have the caller provide documents as a list: documents=[doc0, doc1, doc2]
    # - Each extractor node expects `document` and will use the correct slice.
    #   Since we can't use `.map()` with only the API listed, we hardwire three
    #   branches and route each with a distinct `document` key via an AgentNode
    #   that selects per index.
    #
    # To stay within the "declare reads/writes" constraints, we add three small
    # selector nodes that write the correct `document` scalar.
    select0 = AgentNode(
        name="select0",
        llm=llm,
        node_prompt=(
            "Select DOCUMENT 0 from the provided documents list and return ONLY that document text.\n"
            "documents list (JSON list of strings): {documents}\n"
            "Return a plain string."
        ),
        reads=["documents"],
        writes={"document": str},
    )
    select1 = AgentNode(
        name="select1",
        llm=llm,
        node_prompt=(
            "Select DOCUMENT 1 from the provided documents list and return ONLY that document text.\n"
            "documents list (JSON list of strings): {documents}\n"
            "Return a plain string."
        ),
        reads=["documents"],
        writes={"document": str},
    )
    select2 = AgentNode(
        name="select2",
        llm=llm,
        node_prompt=(
            "Select DOCUMENT 2 from the provided documents list and return ONLY that document text.\n"
            "documents list (JSON list of strings): {documents}\n"
            "Return a plain string."
        ),
        reads=["documents"],
        writes={"document": str},
    )

    # Join of extraction branches:
    # Each extractor writes `extracted`; we gather them into a single `doc_extractions`
    # dict at reconcile. We'll use another agent node to assemble.
    assemble = AgentNode(
        name="assemble",
        llm=llm,
        node_prompt=(
            "Assemble extraction outputs into a single JSON object.\n\n"
            "Given:\n"
            " - doc0: {doc0}\n"
            " - doc1: {doc1}\n"
            " - doc2: {doc2}\n\n"
            "Return ONLY JSON: {\"doc_extractions\": {\"doc0\": doc0, \"doc1\": doc1, \"doc2\": doc2}}"
        ),
        reads=["doc0_extracted", "doc1_extracted", "doc2_extracted"],
        writes={"doc_extractions": dict},
    )

    # Rewire: connect frame -> selectors -> extractors -> assemble -> reconcile.
    # Note: we need to map each extractor output into distinct state keys so assemble can read them.
    rename0 = AgentNode(
        name="rename0",
        llm=llm,
        node_prompt="Take extracted JSON and return it unchanged.",
        reads=["extracted"],
        writes={"doc0_extracted": dict},
    )
    rename1 = AgentNode(
        name="rename1",
        llm=llm,
        node_prompt="Take extracted JSON and return it unchanged.",
        reads=["extracted"],
        writes={"doc1_extracted": dict},
    )
    rename2 = AgentNode(
        name="rename2",
        llm=llm,
        node_prompt="Take extracted JSON and return it unchanged.",
        reads=["extracted"],
        writes={"doc2_extracted": dict},
    )

    # Connectivity:
    # frame must run before any extraction so `entity_spec` is available.
    # Then run selectors+extractors in parallel, then assemble+reconcile.
    frame > fanout(
        # Document 0 pipeline
        select0 > doc0_extract > rename0,
        # Document 1 pipeline
        select1 > doc1_extract > rename1,
        # Document 2 pipeline
        select2 > doc2_extract > rename2,
    ) > assemble > reconcile

    return AgenticGraph(start_node=frame, end_nodes={reconcile})
