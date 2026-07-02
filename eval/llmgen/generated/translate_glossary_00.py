"""Generated pipeline for task 'translate_glossary' (sample 0).

TASK: Detect the language of a passage, translate it into English, then build a short glossary of the key terms that appear in the translation.
"""

from pttai import AgentNode, DecisionNode, ConditionNode, AgenticGraph, fanout


def build_graph(llm):
    # 1) Detect language
    detect_lang = AgentNode(
        name="detect_lang",
        llm=llm,
        reads=["messages"],
        writes={"language": str},
        node_prompt=(
            "Detect the language of the user's passage.\n"
            "Return ONLY the language name (e.g., 'Spanish', 'French', 'Chinese', 'English').\n"
            "Passage is the conversation so far."
        ),
        # We keep the reply in structured output (language) and still allow normal message history usage.
        # The default messages write behavior would append, so prevent it by writing only `language`.
        # (pttai: if writes is set, AgentNode returns only those keys.)
        writes=["language"],
    )

    # 2) Translate into English (use detected language)
    translate = AgentNode(
        name="translate",
        llm=llm,
        reads=["messages", "language"],
        writes={"translation": str},
        node_prompt=(
            "Translate the passage (from the conversation) into English.\n"
            "The detected source language is: {language}\n\n"
            "Return ONLY the English translation, without commentary."
        ),
        writes=["translation"],
    )

    # 3) Build a short glossary from the English translation
    glossary = AgentNode(
        name="glossary",
        llm=llm,
        reads=["translation"],
        writes={"glossary": str},
        node_prompt=(
            "Read the following English translation and build a short glossary.\n"
            "Guidelines:\n"
            "- Extract 6 to 12 key terms/phrases that matter for understanding the passage.\n"
            "- For each term, provide a brief definition (1 sentence max) in simple English.\n"
            "- Output format: one line per entry: 'Term — Definition'\n"
            "- Return ONLY the glossary lines.\n\n"
            "ENGLISH TRANSLATION:\n{translation}"
        ),
        writes=["glossary"],
    )

    # Optional: if translation is already English, still "translate" (should be identity).
    # We keep it simple and always translate; real routing could be added but is unnecessary.

    graph = detect_lang > translate > glossary
    return AgenticGraph(start_node=detect_lang, end_nodes={glossary})
