"""Generated pipeline for task 'translate_glossary' (sample 1).

TASK: Detect the language of a passage, translate it into English, then build a short glossary of the key terms that appear in the translation.
"""

from pttai import AgentNode, DecisionNode, AgenticGraph, ConditionNode
from pttai.tools import make_retriever_tool


def build_graph(llm):
    # 1) Detect language
    detect_lang = AgentNode(
        name="detect_lang",
        llm=llm,
        node_prompt=(
            "You are a language identifier.\n"
            "Detect the language of the user's passage.\n"
            "Return ONLY the language name (e.g., English, Spanish, French, Chinese, Arabic)."
        ),
        reads=["messages"],
        writes=["answer"],
    )

    # 2) Decide translation direction (LLM constrained branching)
    #    Note: We translate into English regardless of language; the branching is used
    #    to keep the flow explicit and validate wiring.
    translate_choice = DecisionNode(
        name="translate_choice",
        llm=llm,
        node_prompt="Will the language of the passage be English already, or not?",
        choices=["already_english", "translate_to_english"],
        input_field="messages",
    )

    translate_to_english = AgentNode(
        name="translate_to_english",
        llm=llm,
        node_prompt=(
            "Translate the provided passage into English.\n"
            "Preserve meaning and key details.\n"
            "Return ONLY the English translation (no explanations)."
        ),
        reads=["messages"],
        writes=["messages"],
    )

    already_english_passthrough = AgentNode(
        name="already_english_passthrough",
        llm=llm,
        node_prompt=(
            "The passage is already in English.\n"
            "Return ONLY the passage as-is (no explanations)."
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # 3) Build a short glossary from the English translation
    #    We ask for a compact list of key terms, each with a short definition.
    make_glossary = AgentNode(
        name="make_glossary",
        llm=llm,
        node_prompt=(
            "You are an assistant that builds a short glossary from a passage.\n"
            "From the English translation, extract the key terms (typically 6-10 terms).\n"
            "Return ONLY a glossary in the format:\n"
            "- Term: definition\n"
            "Keep definitions brief and clear."
        ),
        reads=["messages"],
        writes=["answer"],
    )

    # Wiring
    # Inputs:
    # - messages: seeded by graph.invoke(message=...) or caller.
    # Flow:
    # - detect_lang produces answer (language name) but glossary/translation primarily uses messages.
    # - translate_choice decides whether to translate.
    # - both branches produce the English text back into messages.
    # - make_glossary reads messages and writes glossary to answer.
    #
    # Note: DecisionNode uses its own input_field="messages"; it will see the current messages
    # content at that point (the same original passage). Branching is safe because we wire both.
    detect_lang > translate_choice
    translate_choice["translate_to_english"] > translate_to_english > make_glossary
    translate_choice["already_english"] > already_english_passthrough > make_glossary

    return AgenticGraph(start_node=detect_lang, end_nodes={make_glossary})
