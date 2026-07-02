"""Generated pipeline for task 'aspect_review' (sample 2).

TASK: Review a short piece of writing along several aspects at once -- clarity, grammar, and tone -- then merge the notes into one consolidated critique.
"""

from pttai import AgentNode, AgenticGraph, fanout
from pttai import AgenticState


def build_graph(llm):
    # Accepts input via:
    # - message=<string> shorthand, or
    # - invoke({"messages": [...], "log": [], "token": ...})
    #
    # The pipeline assumes the user's last message contains the writing to review.
    #
    # Output: consolidated critique in the final node's appended messages.

    extract = AgentNode(
        name="extract",
        llm=llm,
        reads=["messages"],
        node_prompt=(
            "Extract the writing-to-review from the user's messages.\n"
            "If multiple drafts are present, choose the most recent one.\n"
            "Return ONLY two paragraphs:\n"
            "1) 'WRITING:' followed by the writing verbatim.\n"
            "2) 'CONSTRAINTS:' listing any stated constraints (audience, length, style) or 'none'."
        ),
        writes=["messages"],
    )

    clarity_notes = AgentNode(
        name="clarity_notes",
        llm=llm,
        node_prompt=(
            "You are a clarity reviewer.\n"
            "Given the extracted writing below, provide concise bullet notes on clarity:\n"
            "- what's confusing or ambiguous\n"
            "- where ideas lack logical flow\n"
            "- any missing context the reader needs\n"
            "End with 2-3 concrete rewrite suggestions.\n\n"
            "{messages}"
        ),
        reads=["messages"],
        writes=["messages"],
    )

    grammar_notes = AgentNode(
        name="grammar_notes",
        llm=llm,
        node_prompt=(
            "You are a grammar reviewer.\n"
            "Given the extracted writing below, provide concise bullet notes on grammar and mechanics:\n"
            "- sentence structure issues\n"
            "- grammar, punctuation, and spelling problems\n"
            "- agreement/tense consistency\n"
            "If you can, include 1-2 micro-edits as before/after.\n\n"
            "{messages}"
        ),
        reads=["messages"],
        writes=["messages"],
    )

    tone_notes = AgentNode(
        name="tone_notes",
        llm=llm,
        node_prompt=(
            "You are a tone reviewer.\n"
            "Given the extracted writing below, provide concise bullet notes on tone:\n"
            "- whether the tone matches the apparent intent and constraints\n"
            "- places that feel too harsh/too casual/uncertain/etc.\n"
            "- word choices that may read differently than intended\n"
            "End with 2-3 specific tone adjustments (phrasing suggestions).\n\n"
            "{messages}"
        ),
        reads=["messages"],
        writes=["messages"],
    )

    merge = AgentNode(
        name="merge",
        llm=llm,
        node_prompt=(
            "You are the editor-chair.\n"
            "Merge the clarity, grammar, and tone notes into ONE consolidated critique.\n"
            "Requirements:\n"
            "- Start with a 2-3 sentence high-level assessment.\n"
            "- Then provide a prioritized list (most important first).\n"
            "  Each bullet must include: (a) the issue, (b) why it matters, (c) how to fix.\n"
            "- End with a short 'Suggested rewrite focus' section listing 3 key rewrite targets.\n\n"
            "INPUTS (merged from parallel reviewers):\n"
            "{messages}"
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # Parallel review, then consolidated merge.
    extract > fanout(clarity_notes, grammar_notes, tone_notes) > merge

    return AgenticGraph(start_node=extract, end_nodes={merge})
