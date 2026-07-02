"""Generated pipeline for task 'refine_until_good' (sample 2).

TASK: Draft a summary, score its quality, and keep refining it until the score is good enough or a few rounds have passed, then return the best version.
"""

from pttai import AgentNode, DecisionNode, AgenticGraph, ConditionNode, fanout


def build_graph(llm):
    # 1) Frame: restate the request as an explicit summarization task
    frame = AgentNode(
        name="frame",
        llm=llm,
        node_prompt=(
            "You are an expert summarizer.\n"
            "Given the user's input, create a concise summarization plan.\n"
            "Output ONLY:\n"
            "SUMMARY_GOAL: <one sentence about what the summary should achieve>\n"
            "AUDIENCE: <who it's for>\n"
            "LENGTH_GUIDE: <e.g., 5-8 sentences or ~120-180 words>\n"
            "KEY_POINTS: <3-6 bullet points of what must be included>\n"
            "OMISSIONS: <2 bullets of what to avoid>\n"
        ),
        writes={"plan": str},
    )

    # 2) Draft multiple candidate summaries in parallel (diversity)
    def candidate_prompt(variant: str) -> str:
        return (
            f"Variant: {variant}\n"
            "Use the PLAN from the conversation to write the summary.\n"
            "Requirements:\n"
            "- Use the key points.\n"
            "- Match the audience.\n"
            "- Follow length guide.\n"
            "- Keep it coherent and specific.\n"
            "Output ONLY the summary text (no headings)."
        )

    # We rely on the conversation to include the plan; since frame writes to plan,
    # we also include the plan by reading it and interpolating into node_prompt.
    draft_a = AgentNode(
        name="draft_a",
        llm=llm,
        reads=["plan", "messages"],
        node_prompt="{plan}\n\n" + candidate_prompt("concise-and-structural"),
        writes={"candidate": str},
    )

    draft_b = AgentNode(
        name="draft_b",
        llm=llm,
        reads=["plan", "messages"],
        node_prompt="{plan}\n\n" + candidate_prompt("balanced-and-readable"),
        writes={"candidate": str},
    )

    draft_c = AgentNode(
        name="draft_c",
        llm=llm,
        reads=["plan", "messages"],
        node_prompt="{plan}\n\n" + candidate_prompt("slightly-more-detailed"),
        writes={"candidate": str},
    )

    # 3) Collect candidate summaries into one text blob for scoring
    collect_candidates = AgentNode(
        name="collect_candidates",
        llm=llm,
        node_prompt=(
            "You will receive multiple candidate summaries (candidate from each branch)."
            "\nCreate a single scoring package:\n"
            "CANDIDATE_A: <text>\n"
            "CANDIDATE_B: <text>\n"
            "CANDIDATE_C: <text>\n"
            "Output ONLY this package."
        ),
    )

    # 4) Score quality of each candidate + pick a winner
    decide_score_and_pick = DecisionNode(
        name="decide_score_and_pick",
        llm=llm,
        node_prompt=(
            "You are an impartial quality evaluator.\n"
            "Given the scoring package with three candidates, choose the best candidate.\n\n"
            "Quality rubric:\n"
            "1) Coverage: includes key points\n"
            "2) Faithfulness: no unsupported claims\n"
            "3) Clarity: readable, well-formed\n"
            "4) Brevity: matches length guide\n"
            "5) Balance: doesn't over-emphasize minor points\n\n"
            "Pick exactly one.\n\n"
            "Return only the decision choice.\n"
        ),
        choices=["A", "B", "C"],
        input_field="messages",
    )

    # 5) Produce a refined final summary from the chosen candidate
    pick_to_refine = AgentNode(
        name="refine",
        llm=llm,
        node_prompt=(
            "You are refining the final summary.\n"
            "Decision (A/B/C) is provided as decision.\n"
            "Use the chosen candidate and improve it slightly for clarity and faithfulness.\n"
            "Keep the same length guide.\n"
            "Output ONLY the final improved summary text."
        ),
    )

    # 6) Loop control: optionally run one more refinement pass if quality is still low
    # (Use a deterministic score heuristic via an LLM-free ConditionNode? We'll use
    # an LLM-based AgentNode to rate, then ConditionNode to decide whether to loop.)
    rate_quality = AgentNode(
        name="rate_quality",
        llm=llm,
        node_prompt=(
            "Rate the final summary quality from 0 to 10 using the rubric:\n"
            "coverage, faithfulness, clarity, brevity, balance.\n"
            "Output ONLY a single integer score."
        ),
        writes={"score": int},
    )

    should_refine_more = ConditionNode(
        name="should_refine_more",
        condition=lambda state: "refine" if state.get("score", 0) < 8 else "stop",
        choices=["refine", "stop"],
        reads=["score", "messages"],
    )

    refine_again = AgentNode(
        name="refine_again",
        llm=llm,
        node_prompt=(
            "One more refinement pass.\n"
            "Improve faithfulness, clarity, and coverage without adding new unsupported claims.\n"
            "Keep it within the original length guide.\n"
            "Output ONLY the updated final summary text."
        ),
    )

    # Wiring:
    # Start with frame. Then draft candidates in parallel. Then collect, score/pick, refine.
    # Then rate and decide whether to loop one more refinement.
    frame > fanout(draft_a, draft_b, draft_c) > collect_candidates > decide_score_and_pick

    decide_score_and_pick["A"] > pick_to_refine
    decide_score_and_pick["B"] > pick_to_refine
    decide_score_and_pick["C"] > pick_to_refine

    # After refine, rate quality, then either loop or stop.
    pick_to_refine > rate_quality > should_refine_more
    should_refine_more["stop"] > pick_to_refine
    should_refine_more["refine"] > refine_again > pick_to_refine

    return AgenticGraph(start_node=frame, end_nodes={pick_to_refine})
