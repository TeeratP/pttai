"""Generated pipeline for task 'translate_glossary' (sample 4).

TASK: Detect the language of a passage, translate it into English, then build a short glossary of the key terms that appear in the translation.
"""

from pttai import AgentNode, DecisionNode, ConditionNode, AgenticGraph, fanout
from pttai.tools import make_retriever_tool  # optional (not used)


def build_graph(llm):
    # 1) Detect language (LLM, constrained)
    # We avoid too many choices; the model can still return one of these.
    detect_lang = DecisionNode(
        name="detect_lang",
        llm=llm,
        node_prompt=(
            "Detect the language of the user's passage.\n"
            "Return exactly one of the choices."
        ),
        choices=[
            "English",
            "Spanish",
            "French",
            "German",
            "Italian",
            "Portuguese",
            "Dutch",
            "Swedish",
            "Norwegian",
            "Danish",
            "Polish",
            "Russian",
            "Ukrainian",
            "Chinese",
            "Japanese",
            "Korean",
            "Arabic",
            "Hebrew",
            "Hindi",
            "Bengali",
            "Thai",
            "Vietnamese",
            "Turkish",
            "Persian",
            "Other",
        ],
        input_field="messages",
    )

    # 2) Translate to English based on detected language
    translate = AgentNode(
        name="translate",
        llm=llm,
        node_prompt=(
            "You are a translation engine.\n"
            "Translate the passage into English.\n\n"
            "Original language (detected): {language}\n"
            "Passage:\n{passage}\n\n"
            "Output only the English translation."
        ),
        reads=["language", "passage"],
        writes={"translation": str},
        # We don't want extra tool-loop behavior here.
        tools=None,
    )

    # 3) Build a short glossary of key terms that appear in the translation
    glossary = AgentNode(
        name="glossary",
        llm=llm,
        node_prompt=(
            "You are building a glossary.\n"
            "Given the English translation below, extract 6-12 key terms.\n"
            "For each term, provide a short definition/synopsis in 1 sentence.\n\n"
            "English translation:\n{translation}\n\n"
            "Return the glossary as plain text, one term per line like:\n"
            "- Term: definition"
        ),
        reads=["translation"],
        writes={"glossary": str},
    )

    # 4) Normalize decision output into "language" for downstream nodes.
    #    Since DecisionNode routes via `decision` field, we still need the chosen
    #    label in state. We'll route all choices to a single normalizer node.
    set_language = AgentNode(
        name="set_language",
        llm=llm,
        node_prompt=(
            "Capture the chosen language label from state for downstream steps.\n"
            "Chosen label: {decision}\n\n"
            "Write only the label."
        ),
        reads=["decision"],
        writes={"language": str},
    )

    # Connect everything:
    # detect_lang chooses a branch; all branches go to set_language,
    # then translation and glossary run sequentially.
    # (Note: DecisionNode branches must be individually wired.)
    # Wire each choice to set_language.
    for c in detect_lang.choices:
        detect_lang[c] > set_language

    # Seed passage into the graph via `message=` invoke shorthand would place
    # the text into `messages`. We'll create a deterministic step that pulls the
    # user passage into a scalar `passage` for prompt interpolation.
    # This step is a plain AgentNode so it can read from `messages`.
    extract_passage = AgentNode(
        name="extract_passage",
        llm=llm,
        node_prompt=(
            "Extract the user's passage text to a single string.\n"
            "Return only the passage content (no quotes, no extra commentary).\n"
            "If multiple messages exist, use the last user message."
        ),
        reads=["messages"],
        writes={"passage": str},
    )

    # Only do translation/glossary after extract_passage.
    extract_passage > detect_lang
    set_language > translate > glossary

    return AgenticGraph(start_node=extract_passage, end_nodes={glossary})
