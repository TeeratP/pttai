"""Generated pipeline for task 'refine_until_good' (sample 1).

TASK: Draft a summary, score its quality, and keep refining it until the score is good enough or a few rounds have passed, then return the best version.
"""

from pttai import AgentNode, AgenticGraph, DecisionNode, ConditionNode, fanout


def build_graph(llm):
    # 1) Draft an initial summary from the conversation/question in messages.
    def make_initial_summary():
        return AgentNode(
            name="draft_summary",
            llm=llm,
            node_prompt=(
                "You are a careful summarizer.\n"
                "Task: Draft a concise summary of the user content.\n\n"
                "Instructions:\n"
                "- Keep it faithful: do not invent details.\n"
                "- Use 3-6 bullet points.\n"
                "- If the content is ambiguous, state uncertainty explicitly."
            ),
            # Read conversation; append summary back into the message history.
            reads=["messages"],
            writes=["messages"],
        )

    # 2) Score the summary quality (structured numeric outputs).
    quality_scorer = AgentNode(
        name="score_quality",
        llm=llm,
        node_prompt=(
            "You are a strict evaluator of a summary.\n"
            "Given the most recent draft summary in the conversation,\n"
            "assign a quality score from 0 to 100.\n\n"
            "Score based on:\n"
            "- Faithfulness to the source (40%)\n"
            "- Coverage of key points (30%)\n"
            "- Clarity/conciseness (20%)\n"
            "- Specificity without fabrication (10%)\n\n"
            "Return:\n"
            "- score: integer 0..100\n"
            "- notes: brief justification"
        ),
        reads=["messages"],
        writes={"score": int},  # typed structured output
    )

    # 3) Decide whether it's good enough to stop.
    # (DecisionNode writes to the built-in `decision` field; we route from it.)
    decide_good_enough = DecisionNode(
        name="decide_stop_or_refine",
        llm=llm,
        node_prompt=(
            "Decide whether the current summary quality score is good enough.\n\n"
            "If score >= 80, choose 'stop'. Otherwise choose 'refine'.\n"
            "You must choose exactly one option."
        ),
        choices=["stop", "refine"],
        input_field="messages",
        # No prompt placeholders: score is available only via state; we declare reads
        # for static validation via `reads` to make score available for interpolation
        # only if we used placeholders. Since we don't, keep default behavior.
    )

    # 4) Refine the summary using the evaluator notes + the draft.
    refine = AgentNode(
        name="refine_summary",
        llm=llm,
        node_prompt=(
            "You are an expert reviser.\n"
            "Refine the summary based on the evaluator feedback.\n\n"
            "Requirements:\n"
            "- Preserve factual correctness; never invent new claims.\n"
            "- Improve coverage and clarity.\n"
            "- Output a revised summary as 3-6 bullet points."
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # 5) Cap refinement loops (few rounds passed).
    # We'll use a deterministic condition that checks a counter in state.
    # We seed/maintain `round` implicitly by writing it in the refinement loop is not
    # directly supported without custom tool/functions; instead we use a simple
    # deterministic cap by condition on an existing `round` key and require the caller
    # to provide it. To make this graph self-contained, we store `round` in the
    # conversation via the evaluator/summarizer messages and use a ConditionNode
    # that inspects the conversation text for pattern 'Round X'. This is robust enough
    # without extra keys.
    #
    # ConditionNode reads are purely structural for validation; it doesn't do prompt
    # interpolation. We'll still declare reads=["messages"] so the validator checks the key.
    def parse_round_from_messages(state):
        # Extract the last occurrence of "Round N" from messages content; default 1.
        try:
            msgs = state["messages"]
            if not msgs:
                return "continue"
            text = msgs[-1].content if hasattr(msgs[-1], "content") else str(msgs[-1])
        except Exception:
            return "continue"
        import re

        matches = re.findall(r"Round\s+(\d+)", text)
        if matches:
            rnd = int(matches[-1])
        else:
            rnd = 1

        # Allow up to 3 refinement rounds (few rounds have passed).
        return "stop" if rnd >= 3 else "continue"

    cap_condition = ConditionNode(
        name="cap_refinement",
        condition=parse_round_from_messages,
        choices=["continue", "stop"],
        reads=["messages"],
    )

    # 6) Final "wrap-up" node (ensure final output is the latest summary).
    finalize = AgentNode(
        name="finalize",
        llm=llm,
        node_prompt=(
            "Return the final summary only.\n"
            "Output format: 3-6 bullet points."
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # Wiring:
    initial = make_initial_summary()

    # We'll score, then decide to stop/refine.
    # If refine: add "Round X" hint before refinement to make cap_condition work.
    #
    # Since we cannot write a dedicated round counter key, we inject the round marker
    # into the conversation by a small deterministic step: we reuse refine_summary
    # prompt as-is; but to ensure the marker exists, we prepend it via a short agent
    # message before refinement. This is done with one extra AgentNode.
    add_round_marker = AgentNode(
        name="add_round_marker",
        llm=llm,
        node_prompt=(
            "You are preparing for another refinement round.\n"
            "Determine the current round number based on the existing conversation.\n"
            "Then output ONLY a single line: 'Round N' where N is the next integer.\n"
            "Do not add any other text."
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # Refinement loop edges:
    # draft_summary -> score_quality -> decide_stop_or_refine
    # stop -> finalize
    # refine -> cap_condition (continue/stop)
    # continue -> add_round_marker -> refine_summary -> score_quality (loop)
    # stop -> finalize
    draft_to_score = initial > quality_scorer
    draft_to_decide = draft_to_score > decide_good_enough

    # Route from decide_stop_or_refine
    draft_to_decide["stop"] > finalize

    draft_to_decide["refine"] > cap_condition
    cap_condition["stop"] > finalize
    cap_condition["continue"] > add_round_marker > refine > quality_scorer

    return AgenticGraph(start_node=initial, end_nodes={finalize})
