"""Generated pipeline for task 'extract_summarize' (sample 3).

TASK: Extract the key entities and claims from a document, then write a short summary that uses the extracted information.
"""

from pttai import AgentNode, AgenticGraph, fanout


def build_graph(llm):
    # 1) Extract entities + claims into a structured JSON-like text payload.
    extract = AgentNode(
        name="extract",
        llm=llm,
        node_prompt=(
            "You are an information extraction engine.\n"
            "Given a document (in the user's message), extract:\n"
            "1) Key entities: list 5-15 entities as objects with fields: "
            "name, type (e.g., person/org/product/law/concept/metric), and brief_role.\n"
            "2) Key claims: list 5-20 claims as objects with fields: "
            "claim, claim_type (factual/causal/evaluative/predictive/normative), "
            "supporting_evidence (quote or short paraphrase), and confidence (low/med/high).\n"
            "3) Claim-to-entity links: for each claim, list the entity names that it refers to.\n\n"
            "Output format (STRICT):\n"
            "{\n"
            '  "entities": [ ... ],\n'
            '  "claims": [ ... ]\n'
            "}\n"
            "No extra text outside the JSON object."
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # 2) Create two parallel views: a compact structured list and a rationale outline.
    # Both read the same extracted content (messages history already contains it).
    compact_entities_claims = AgentNode(
        name="compact_entities_claims",
        llm=llm,
        node_prompt=(
            "Transform the extracted JSON (found in the conversation) into a compact outline.\n"
            "Return ONLY this text (no JSON, no preamble):\n\n"
            "ENTITIES:\n"
            "- <EntityName> (<EntityType>): <brief_role>\n"
            "...\n\n"
            "CLAIMS:\n"
            "- (<claim_type>, conf=<confidence>) <claim>\n"
            "  Evidence: <supporting_evidence>\n"
            "  Mentions: <entity names comma-separated>\n"
            "...\n"
        ),
        reads=["messages"],
        writes=["messages"],
    )

    summary_outline = AgentNode(
        name="summary_outline",
        llm=llm,
        node_prompt=(
            "Using the extracted entities and claims (JSON in the conversation), draft:\n"
            "1) 3-6 bullet-point summary capturing the document's main points,\n"
            "2) the strongest 1-2 claims and what evidence supports them.\n"
            "Return ONLY the bullets in this form:\n"
            "- ...\n"
            "..."
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # 3) Final summary that uses the extracted information (entities + claims).
    write_summary = AgentNode(
        name="write_summary",
        llm=llm,
        node_prompt=(
            "Write a short summary (max 120 words) of the document using only the extracted information.\n"
            "Requirements:\n"
            "- Mention 2-5 key entities by name.\n"
            "- Reference 2-4 key claims, including their type when helpful.\n"
            "- Include one brief supporting evidence snippet or paraphrase.\n"
            "- Do not introduce facts not present in the document/extraction.\n\n"
            "Return ONLY the summary text."
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # Fan-out after extraction so both branches use the same extracted material.
    # They rejoin at the writer.
    pipeline = extract > fanout(compact_entities_claims, summary_outline) > write_summary

    return AgenticGraph(start_node=extract, end_nodes={write_summary})
