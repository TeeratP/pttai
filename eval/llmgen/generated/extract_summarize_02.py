"""Generated pipeline for task 'extract_summarize' (sample 2).

TASK: Extract the key entities and claims from a document, then write a short summary that uses the extracted information.
"""

from pttai import AgentNode, DecisionNode, ConditionNode, AgenticGraph, fanout


def build_graph(llm):
    # 1) Extract entities and claims
    extractor = AgentNode(
        name="extractor",
        llm=llm,
        node_prompt=(
            "You are an information extraction system.\n\n"
            "Given the document in the conversation, extract:\n"
            "1) Key entities (people, organizations, products, places, concepts) as a bullet list.\n"
            "2) Key claims as a bullet list.\n\n"
            "For EACH claim, format it as:\n"
            "- Claim: <claim text>\n"
            "  Evidence: <short quoted / paraphrased evidence span from the document>\n"
            "  Entities mentioned: <comma-separated entity names if present; else 'none'>\n\n"
            "Be faithful to the text. If the document lacks evidence for a claim, do not invent one.\n"
            "Output only the extracted entities and claims."
        ),
    )

    # 2) Decide whether we have enough extracted content to summarize
    has_content = ConditionNode(
        name="has_content",
        condition=lambda state: (
            "yes"
            if state.get("messages")
            and any(
                ("Claim:" in (m.content or "")) or ("Evidence:" in (m.content or ""))
                for m in state["messages"]
            )
            else "no"
        ),
        choices=["yes", "no"],
        reads=["messages"],
    )

    # 3) Fallback summarizer if extraction is empty/insufficient
    fallback_summarizer = AgentNode(
        name="fallback_summarizer",
        llm=llm,
        node_prompt=(
            "The extraction did not yield clear claims/entities.\n\n"
            "Write a short summary of the document anyway based on the document text.\n"
            "Summary requirements:\n"
            "- 3 to 5 sentences.\n"
            "- Include the main topic and any explicit, directly-supported facts.\n"
            "- Do not invent details not present in the document.\n"
            "Output ONLY the summary."
        ),
    )

    # 4) Summarize using extracted information
    summarizer = AgentNode(
        name="summarizer",
        llm=llm,
        node_prompt=(
            "You are a summarizer that must use the provided extracted information.\n\n"
            "The conversation contains extraction output with:\n"
            "- Key entities\n"
            "- Key claims with evidence\n\n"
            "Write a short summary (3 to 5 sentences) that:\n"
            "- Mentions the most important entities.\n"
            "- States the most important claims, reflecting the evidence.\n"
            "- Avoids adding new information beyond the extracted claims.\n"
            "Output ONLY the summary."
        ),
    )

    # Wiring:
    extractor > has_content
    has_content["no"] > fallback_summarizer
    has_content["yes"] > summarizer

    end_nodes = {fallback_summarizer, summarizer}
    return AgenticGraph(start_node=extractor, end_nodes=end_nodes)
