"""Generated pipeline for task 'aspect_review' (sample 1).

TASK: Review a short piece of writing along several aspects at once -- clarity, grammar, and tone -- then merge the notes into one consolidated critique.
"""

from pttai import AgentNode, AgenticGraph, fanout


def build_graph(llm):
    # 1) Clarify the task into a single, shared target output format
    frame = AgentNode(
        name="frame",
        llm=llm,
        node_prompt=(
            "You are a writing reviewer orchestrator.\n"
            "Given the input writing plus the user's notes (if any), produce ONE consolidated plan:\n"
            "- Preserve the author's intent.\n"
            "- Identify the top issues in: clarity, grammar, and tone.\n"
            "- Specify how to fix them (concrete edits or principles).\n"
            "Output requirements for this node:\n"
            "Return a single concise critique spec with the following headings:\n"
            "Clarify:\n"
            "Grammar:\n"
            "Tone:\n"
            "Prioritized fixes (top 5):\n"
            "Do not write the final critique yet—only the spec."
        ),
    )

    # 2) Parallel reviewers that each focus on one aspect
    clarity = AgentNode(
        name="clarity",
        llm=llm,
        node_prompt=(
            "You are the clarity reviewer.\n"
            "Review the provided writing and notes.\n"
            "Focus ONLY on clarity: structure, missing context, vague claims, logical flow, and reader comprehension.\n"
            "Return:\n"
            "- 3-7 specific issues\n"
            "- For each: what to change (rewrite a sentence fragment if helpful)\n"
            "- Optional: a suggested reordered outline (max 5 bullets)\n"
            "Keep tone professional and actionable."
        ),
    )

    grammar = AgentNode(
        name="grammar",
        llm=llm,
        node_prompt=(
            "You are the grammar reviewer.\n"
            "Review the provided writing and notes.\n"
            "Focus ONLY on grammar and mechanical issues: agreement, tense consistency, punctuation, word choice errors, and malformed sentences.\n"
            "Return:\n"
            "- 3-7 specific issues\n"
            "- For each: corrected version of the affected phrase/sentence\n"
            "Keep tone professional and actionable."
        ),
    )

    tone = AgentNode(
        name="tone",
        llm=llm,
        node_prompt=(
            "You are the tone reviewer.\n"
            "Review the provided writing and notes.\n"
            "Focus ONLY on tone: voice consistency, politeness, formality level, emotional temperature, and whether phrasing matches the intended audience.\n"
            "Return:\n"
            "- 3-7 specific tone issues\n"
            "- For each: why it's a problem + an example replacement phrasing\n"
            "Keep tone professional and actionable."
        ),
    )

    # 3) Consolidate all notes into one unified critique
    verdict = AgentNode(
        name="verdict",
        llm=llm,
        node_prompt=(
            "You are the chair/editor.\n"
            "Inputs: a frame spec plus three reviewer notes (clarity, grammar, tone).\n"
            "Merge them into ONE consolidated critique.\n\n"
            "Output format (exact headings):\n"
            "Consolidated critique:\n"
            "- Brief overall assessment (2-4 sentences)\n"
            "- Priority issues (top 5):\n"
            "  1) Issue...\n"
            "     Fix...\n"
            "     Example edit (if possible)...\n"
            "  2) ...\n"
            "Grammar fixes (in order):\n"
            "- Bullet list of corrected phrases/sentences (as needed)\n"
            "Clarity fixes (in order):\n"
            "- Bullet list of rewrites/reorganizations (as needed)\n"
            "Tone fixes (in order):\n"
            "- Bullet list of replacement phrasing (as needed)\n"
            "Suggested rewrite (optional):\n"
            "- If you can, provide a short revised version or a paragraph that demonstrates the fixes.\n\n"
            "Constraints:\n"
            "- Do not contradict the frame spec.\n"
            "- Keep the critique concise but concrete.\n"
            "- Preserve the author's intent and do not rewrite everything unless necessary."
        ),
    )

    # Wiring: one shared input messages history -> frame and parallel reviewers -> verdict
    frame > fanout(clarity, grammar, tone) > verdict

    return AgenticGraph(start_node=frame, end_nodes={verdict})
