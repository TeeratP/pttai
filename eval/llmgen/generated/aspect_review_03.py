"""Generated pipeline for task 'aspect_review' (sample 3).

TASK: Review a short piece of writing along several aspects at once -- clarity, grammar, and tone -- then merge the notes into one consolidated critique.
"""

from pttai import AgentNode, AgenticGraph, fanout


def build_graph(llm):
    # --- Parsers / reviewers (fan-out in parallel) ---
    clarity = AgentNode(
        name="clarity",
        llm=llm,
        node_prompt=(
            "You are a writing editor focused on CLARITY.\n"
            "Review the provided writing and any reviewer notes.\n\n"
            "Return ONLY bullet points with:\n"
            "1) What is unclear (quote up to 12 words each where helpful)\n"
            "2) The specific clarification needed\n"
            "3) A suggested rewrite for the most important sentence (one sentence rewrite)\n"
        ),
    )

    grammar = AgentNode(
        name="grammar",
        llm=llm,
        node_prompt=(
            "You are a writing editor focused on GRAMMAR and usage.\n"
            "Review the provided writing and any reviewer notes.\n\n"
            "Return ONLY bullet points with:\n"
            "1) The exact problem (spelling, punctuation, agreement, tense, word choice)\n"
            "2) A corrected version of the phrase/sentence\n"
            "3) Any consistency issue to fix (style, tense, capitalization)\n"
        ),
    )

    tone = AgentNode(
        name="tone",
        llm=llm,
        node_prompt=(
            "You are a writing editor focused on TONE.\n"
            "Review the provided writing and any reviewer notes.\n\n"
            "Return ONLY bullet points with:\n"
            "1) Tone mismatches (e.g., too harsh/too vague/inconsistent)\n"
            "2) Why it might read that way (cite a short phrase)\n"
            "3) Suggested alternative phrasing to achieve the intended tone\n"
            "If tone is not specified, infer a reasonable professional academic neutral tone and explain briefly.\n"
        ),
    )

    # --- Merger / consolidated critique ---
    merge = AgentNode(
        name="merge",
        llm=llm,
        node_prompt=(
            "You are an editor producing one CONSOLIDATED CRITIQUE.\n\n"
            "You will receive three reviewer outputs: clarity, grammar, and tone.\n"
            "Merge them into ONE cohesive critique with:\n"
            "A) Top issues (3-6 bullets), each bullet combining aspect(s)\n"
            "B) What to change first (ordered list)\n"
            "C) A revised version of the writing (full rewrite) that incorporates fixes while preserving intent\n"
            "D) One short note on style consistency to maintain in the next draft\n\n"
            "Constraints:\n"
            "- Do not repeat the raw reviewer bullets verbatim; synthesize.\n"
            "- Keep the rewrite concise but complete.\n"
            "- Preserve the original meaning; only adjust wording for clarity/grammar/tone.\n"
        ),
    )

    # --- Frame the task into the shared prompt for reviewers ---
    frame = AgentNode(
        name="frame",
        llm=llm,
        node_prompt=(
            "Restate the task as a single instruction package for reviewers.\n"
            "Given user input that contains:\n"
            "- Writing to review\n"
            "- Any existing reviewer notes\n"
            "Optional fields may include intended audience and intended tone.\n\n"
            "Output a concise prompt bundle that includes:\n"
            "1) The writing (verbatim)\n"
            "2) The existing notes (verbatim, if present)\n"
            "3) Intended audience (if present)\n"
            "4) Intended tone (if present)\n"
            "5) Explicitly ask reviewers to focus on clarity, grammar, and tone respectively.\n"
        ),
    )

    # Wiring: frame -> parallel reviewers -> merge
    frame > fanout(clarity, grammar, tone) > merge

    return AgenticGraph(start_node=frame, end_nodes={merge})
