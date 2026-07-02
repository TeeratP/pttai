"""Generated pipeline for task 'aspect_review' (sample 0).

TASK: Review a short piece of writing along several aspects at once -- clarity, grammar, and tone -- then merge the notes into one consolidated critique.
"""

from pttai import AgentNode, AgenticGraph, fanout, ConditionNode


def build_graph(llm):
    # 1) Frame: ensure we have a clean, concrete writing target and what to produce.
    frame = AgentNode(
        name="frame",
        llm=llm,
        node_prompt=(
            "You are a careful writing reviewer.\n"
            "Given the user's input text (and any notes the user provided), do two things:\n"
            "1) Identify the writing to review as the 'Target Text'.\n"
            "2) Identify what type of consolidated critique to produce.\n\n"
            "Output requirements for your next step: produce a compact plan listing the aspects to cover:\n"
            "- clarity\n"
            "- grammar\n"
            "- tone\n"
            "and specify any additional constraints (e.g., preserve meaning, keep it professional).\n"
            "Write your plan as short bullet points."
        ),
    )

    # 2) Parallel reviewers: clarity, grammar, tone.
    clarity = AgentNode(
        name="clarity",
        llm=llm,
        node_prompt=(
            "You are a clarity reviewer.\n"
            "Using the Target Text and the plan from the previous step, provide a brief set of notes:\n"
            "- What is unclear and why (specific quotes helpful if present)\n"
            "- Proposed edits to improve structure and readability\n"
            "Keep it actionable and limited to the most important issues."
        ),
    )

    grammar = AgentNode(
        name="grammar",
        llm=llm,
        node_prompt=(
            "You are a grammar reviewer.\n"
            "Using the Target Text and the plan from the previous step, provide a brief set of notes:\n"
            "- Correct grammar, punctuation, and sentence-level issues\n"
            "- Point out any agreement/tense/spelling problems\n"
            "Provide suggested replacements where helpful. Keep it actionable and limited."
        ),
    )

    tone = AgentNode(
        name="tone",
        llm=llm,
        node_prompt=(
            "You are a tone reviewer.\n"
            "Using the Target Text and the plan from the previous step, provide a brief set of notes:\n"
            "- Does the tone match the intended audience/purpose? Explain mismatches\n"
            "- Identify phrases that feel too harsh, too vague, too casual, etc.\n"
            "- Suggest toned edits that preserve meaning.\n"
            "Keep it actionable and limited to the most important issues."
        ),
    )

    # 3) Merge into consolidated critique.
    merge = AgentNode(
        name="merge",
        llm=llm,
        node_prompt=(
            "You are the editor-in-chief consolidating multiple reviewer notes into one unified critique.\n"
            "You will receive the clarity, grammar, and tone notes (possibly overlapping).\n"
            "Produce exactly ONE consolidated critique that:\n"
            "1) Starts with a short overall assessment (1-2 sentences).\n"
            "2) Has three clearly labeled sections: Clarity, Grammar, Tone.\n"
            "3) In each section, list the top 3-7 changes/observations as bullets.\n"
            "4) Avoid duplication: if the same fix is mentioned by multiple reviewers, include it once.\n"
            "5) Be specific and actionable.\n\n"
            "Do not rewrite the full text; focus on critique and suggestions."
        ),
    )

    # 4) If the input is extremely short, ask for more context before finalizing.
    #    This uses deterministic branching, but still keeps the pipeline a single graph.
    #    The LLM-based nodes generate content into messages; this condition decides whether to proceed.
    def needs_more_context(state):
        # Determine based on conversation messages length/content (no LLM call).
        msgs = state.get("messages", [])
        text = "\n".join(
            getattr(m, "content", "") for m in msgs if hasattr(m, "content") and m.content is not None
        ).strip()
        # Heuristic: if the target text seems missing or too short, request more.
        if len(text) < 120:
            return "request_more"
        return "finalize"

    decide_context = ConditionNode(
        name="decide_context",
        condition=needs_more_context,
        choices=["request_more", "finalize"],
        reads=["messages"],
    )

    request_more = AgentNode(
        name="request_more",
        llm=llm,
        node_prompt=(
            "The provided input is too short or missing the Target Text.\n"
            "Ask the user for what you need to produce a consolidated critique.\n"
            "Request:\n"
            "- The exact text to review\n"
            "- Intended audience and purpose (if known)\n"
            "- Any target tone (e.g., formal, friendly, persuasive)\n"
            "Keep the questions brief (3-5 bullet points)."
        ),
    )

    finalize_passthrough = AgentNode(
        name="finalize",
        llm=llm,
        node_prompt=(
            "Finalize the consolidated critique.\n"
            "If you already have a complete critique, polish formatting only (headings, bullet structure) without adding new substantive points.\n"
            "If not, produce the best consolidated critique you can from available notes.\n"
            "Output the final critique as plain text."
        ),
    )

    # Wiring:
    # frame runs first, then critics in parallel, then merge.
    # After merge, branch deterministically to either ask for more context or finalize.
    frame > fanout(clarity, grammar, tone) > merge
    merge > decide_context
    decide_context["request_more"] > request_more
    decide_context["finalize"] > finalize_passthrough

    return AgenticGraph(start_node=frame, end_nodes={request_more, finalize_passthrough})
