"""Generated pipeline for task 'translate_glossary' (sample 2).

TASK: Detect the language of a passage, translate it into English, then build a short glossary of the key terms that appear in the translation.
"""

from pttai import AgentNode, DecisionNode, AgenticGraph, fanout


def build_graph(llm):
    detect_lang = DecisionNode(
        name="detect_lang",
        llm=llm,
        node_prompt=(
            "Detect the primary language of the user's passage. "
            "Return exactly one choice. If the language is mixed, choose the dominant one."
        ),
        choices=["English", "Spanish", "French", "German", "Italian", "Portuguese", "Dutch", "Swedish", "Norwegian", "Danish", "Finnish", "Polish", "Czech", "Slovak", "Hungarian", "Romanian", "Russian", "Ukrainian", "Bulgarian", "Greek", "Turkish", "Arabic", "Hebrew", "Persian", "Hindi", "Urdu", "Bengali", "Punjabi", "Tamil", "Telugu", "Marathi", "Gujarati", "Kannada", "Malayalam", "Thai", "Vietnamese", "Indonesian", "Malay", "Filipino", "Chinese", "Japanese", "Korean", "Latin", "Other"],
        input_field="messages",
        reads=["messages"],
    )

    translate = AgentNode(
        name="translate",
        llm=llm,
        node_prompt=(
            "You are a translation engine. Translate the following passage into English. "
            "Preserve meaning, proper nouns, and any important formatting. "
            "Return ONLY the translated English text."
        ),
        input_field="messages",
        reads=["messages"],
        writes={"translation": str},
    )

    glossary = AgentNode(
        name="glossary",
        llm=llm,
        node_prompt=(
            "Build a short glossary from the translated passage.\n\n"
            "Rules:\n"
            "- Extract 8-15 key terms or phrases (content words, technical terms, and important concepts).\n"
            "- For each term, give a one-sentence definition in simple English.\n"
            "- Only include terms that actually appear in the passage.\n"
            "- Output as a compact bullet list: '- Term: definition'.\n"
            "Return ONLY the glossary."
        ),
        input_field="messages",
        writes={"glossary": str},
        reads=["translation"],
    )

    # We need a node to set up the translation input from the state produced by translate.
    # The detection choice exists mostly to make the graph explicit; translation itself uses the original messages.
    # Route all languages to the same translation step.
    translate_branch = fanout  # alias for readability

    # Wire: detect_lang -> translate -> glossary. Other choices all map to translate.
    # Because DecisionNode routing uses explicit handlers per choice, we attach translate as the child for each.
    # (DecisionNode returns a choice in state["decision"] and we index it.)
    # Note: We'll build routes for all choices by listing them explicitly in an if-free way:
    for _choice in detect_lang.choices:
        detect_lang[_choice] > translate

    # Continue sequentially
    translate > glossary

    return AgenticGraph(start_node=detect_lang, end_nodes={glossary})
