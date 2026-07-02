"""Generated pipeline for task 'aspect_review' (sample 4).

TASK: Review a short piece of writing along several aspects at once -- clarity, grammar, and tone -- then merge the notes into one consolidated critique.
"""

from pttai import AgentNode, DecisionNode, AgenticGraph, fanout, AgenticState


def build_graph(llm):
    # --- Frame the job as a single decision to enable clean branching ---
    frame = AgentNode(
        name="frame",
        llm=llm,
        node_prompt=(
            "Given the user's writing (in state['messages'] history), choose what the output should focus on.\n"
            "Restate the task as ONE concise review goal, and keep it grounded in the provided text.\n"
            "Return a single-sentence decision target."
        ),
        reads=["messages"],
        writes=["messages"],  # default append; used as context for downstream nodes
    )

    # --- Classify the writing type to help adjust tone of critiques ---
    classify = DecisionNode(
        name="classify",
        llm=llm,
        node_prompt=(
            "Classify the writing into one category.\n"
            "Categories:\n"
            "- 'formal': academic/business/memo style\n"
            "- 'casual': personal/relaxed style\n"
            "- 'creative': fiction/poetry/scene setting\n"
            "- 'marketing': sales/landing/ads\n"
            "Choose the single best match based only on the text."
        ),
        choices=["formal", "casual", "creative", "marketing"],
        input_field="messages",
    )

    # --- Shared context for reviewers ---
    # Note: reviewers read from messages; they each produce one critique to be merged.
    clarify_agent = AgentNode(
        name="clarity_review",
        llm=llm,
        node_prompt=(
            "You are a writing coach specializing in CLARITY.\n"
            "Review the provided writing for:\n"
            "- main claim/thesis clarity\n"
            "- structure and logical flow\n"
            "- ambiguous references\n"
            "- missing context\n"
            "Produce:\n"
            "1) Bullet list of specific issues\n"
            "2) Bullet list of actionable fixes\n"
            "Keep the critique concise but concrete."
        ),
        reads=["messages"],
        writes=["messages"],
    )

    grammar_agent = AgentNode(
        name="grammar_review",
        llm=llm,
        node_prompt=(
            "You are a copy editor specializing in GRAMMAR & MECHANICS.\n"
            "Review the provided writing for:\n"
            "- grammar, tense consistency, subject-verb agreement\n"
            "- punctuation, capitalization\n"
            "- awkward phrasing caused by grammar\n"
            "Produce:\n"
            "1) Bullet list of specific problems (quote minimal fragments if helpful)\n"
            "2) Corrected suggestions for each problem\n"
            "Keep it crisp; don't rewrite the entire piece."
        ),
        reads=["messages"],
        writes=["messages"],
    )

    tone_agent = AgentNode(
        name="tone_review",
        llm=llm,
        node_prompt=(
            "You are a style consultant specializing in TONE.\n"
            "Review the provided writing for alignment of tone to intent and audience.\n"
            "Identify mismatches such as:\n"
            "- overly harsh/overly casual vs intended\n"
            "- hedging or confidence level issues\n"
            "- formality mismatch\n"
            "Produce:\n"
            "1) Bullet list of tone issues\n"
            "2) Bullet list of tone adjustments (with example phrases when useful)\n"
            "Keep the critique practical."
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # --- Consolidate reviews into one consolidated critique ---
    merge = AgentNode(
        name="merge_critique",
        llm=llm,
        node_prompt=(
            "You are the editor-in-chief. Merge the reviewers' notes into ONE consolidated critique.\n"
            "Requirements:\n"
            "- Cover clarity, grammar, and tone (all three aspects), even if a reviewer said 'no issues'.\n"
            "- Remove redundancies; keep non-overlapping points.\n"
            "- Prioritize the most important issues first.\n"
            "- For each major issue, provide a concrete actionable fix.\n"
            "- End with a short 'Top 3 next edits' list.\n"
            "Output format:\n"
            "Consolidated Critique:\n"
            "- Clarity:\n"
            "- Grammar:\n"
            "- Tone:\n"
            "Top 3 next edits:\n"
            "1) ...\n"
            "2) ...\n"
            "3) ..."
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # --- Route classified type to a prompt-tuning stage for tone expectations (still LLM) ---
    # This DecisionNode only affects tone; reviewers still run in parallel.
    tone_tuner = AgentNode(
        name="tone_tuner",
        llm=llm,
        node_prompt=(
            "You are calibrating tone expectations for the review based on the writing type.\n"
            "Writing type: {decision}\n"
            "Rewrite the following guidance as a short checklist (3-6 bullets) that the tone reviewer should use:\n"
            "- formal tone should stay disciplined and precise\n"
            "- casual tone should be friendly but still clear\n"
            "- creative tone should maintain imagery and voice consistency\n"
            "- marketing tone should emphasize benefits, clarity, and persuasive structure without sounding spammy\n"
            "Return ONLY the checklist."
        ),
        reads=["decision"],
        writes=["messages"],
    )

    # We must route each decision choice to the same tone_tuner node; each branch triggers the same next node.
    classify["formal"] > tone_tuner
    classify["casual"] > tone_tuner
    classify["creative"] > tone_tuner
    classify["marketing"] > tone_tuner

    # After tone_tuner, run reviewers concurrently, then merge.
    # The fanout is expressed as a single wiring line to keep it declarative.
    tone_tuner > fanout(clarify_agent, grammar_agent, tone_agent) > merge

    # Provide entry: frame the decision target into messages context.
    frame > classify

    return AgenticGraph(start_node=frame, end_nodes={merge}, state_schema=AgenticState)
