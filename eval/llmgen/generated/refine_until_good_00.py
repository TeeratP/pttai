"""Generated pipeline for task 'refine_until_good' (sample 0).

TASK: Draft a summary, score its quality, and keep refining it until the score is good enough or a few rounds have passed, then return the best version.
"""

from pttai import AgentNode, DecisionNode, AgenticGraph, ConditionNode


def build_graph(llm):
    # 1) Draft summary from the input text
    draft = AgentNode(
        name="draft",
        llm=llm,
        node_prompt=(
            "You are a careful summarizer.\n\n"
            "Task: Write a concise, accurate summary of the user's content.\n"
            "Requirements:\n"
            "- Capture the main points and key details.\n"
            "- Avoid speculation.\n"
            "- Use 5-8 bullet points.\n"
            "- If the content is long, prefer the most important facts.\n\n"
            "Content:\n{content}"
        ),
        reads=["content"],
        writes=["messages"],
    )

    # 2) Score the summary quality against explicit rubrics.
    #    Returns an integer score in state["score"].
    score_summary = AgentNode(
        name="score_summary",
        llm=llm,
        node_prompt=(
            "You are an expert evaluator.\n\n"
            "Evaluate the summary in terms of:\n"
            "1) Fidelity to the content (0-5)\n"
            "2) Coverage of main points (0-5)\n"
            "3) Specificity / usefulness (0-5)\n"
            "4) Clarity / concision (0-5)\n"
            "Total score = sum of the four sub-scores, so 0..20.\n\n"
            "Return ONLY the integer total score.\n\n"
            "Content:\n{content}\n\n"
            "Summary:\n{summary}"
        ),
        reads=["content", "summary"],
        writes={"score": int},
    )

    # 3) Decide whether to refine or accept.
    decide_refine = DecisionNode(
        name="decide_refine",
        llm=llm,
        node_prompt=(
            "You are deciding whether a summary needs refinement.\n"
            "Given a numeric score from 0..20, choose one option:\n"
            "- 'refine' if the summary is not good enough\n"
            "- 'accept' if it is good enough\n\n"
            "Heuristic: Accept when score >= 14, otherwise refine.\n"
            "Choose exactly one.\n"
            "Score: {score}"
        ),
        choices=["refine", "accept"],
        reads=["score"],
    )

    # 4) Refine summary if needed
    refine = AgentNode(
        name="refine",
        llm=llm,
        node_prompt=(
            "Refine the summary to improve quality while staying faithful to the content.\n\n"
            "Use this rubric to fix issues:\n"
            "- Increase fidelity and remove any weak/uncertain claims.\n"
            "- Ensure coverage of the main points.\n"
            "- Improve clarity and usefulness; keep it concise.\n"
            "- Keep the output as 5-8 bullet points.\n\n"
            "Content:\n{content}\n\n"
            "Current summary:\n{summary}\n\n"
            "Produce the improved summary."
        ),
        reads=["content", "summary"],
        writes=["messages"],
    )

    # 5) Convert the (latest) draft/refined messages into a scalar "final_summary"
    #    for the scoring to use in the loop and for the eventual output.
    #    (If refine isn't taken, draft messages still contain the summary.)
    extract_summary = AgentNode(
        name="extract_summary",
        llm=llm,
        node_prompt=(
            "Extract the summary text from the conversation.\n"
            "Return ONLY the summary as plain text (no extra commentary).\n"
            "If there are multiple messages, use the most recent one.\n"
        ),
        reads=["messages"],
        writes=["final_summary"],
    )

    # 6) Stop condition: accept if score is good enough, else continue for a few rounds.
    #    We keep iteration bounded with a simple counter stored in state["round"].
    #    Start round if not present; we rely on a deterministic ConditionNode for routing.
    #    Note: if state["round"] is missing, this ConditionNode should still work by treating it as 0.
    #    We'll implement the predicate in Python via closure.
    def should_accept_or_stop(state):
        # Accept if score >= 14
        score = state.get("score", 0)
        if score >= 14:
            return "done"
        # Allow a few rounds total; if round >= 3, force done.
        if state.get("round", 0) >= 3:
            return "done"
        return "refine_again"

    continue_or_done = ConditionNode(
        name="continue_or_done",
        condition=should_accept_or_stop,
        choices=["refine_again", "done"],
        reads=["score", "round"],
    )

    # 7) Increment round deterministically.
    #    Implement as an AgentNode writing a small scalar because we only can define pttai nodes,
    #    not raw python functions as nodes in this DSL.
    inc_round = AgentNode(
        name="inc_round",
        llm=llm,
        node_prompt=(
            "You maintain an integer iteration counter.\n"
            "Input round may be missing; treat missing as 0.\n"
            "Return the incremented value: round = round + 1.\n"
            "Return ONLY the integer.\n"
            "Current round: {round}"
        ),
        reads=["round"],
        writes={"round": int},
    )

    # Wiring / Loop:
    # draft -> extract_summary -> score_summary -> decide_refine -> (refine -> extract_summary -> score_summary)
    # Then final gating with continue_or_done decides done vs refine_again with a bounded counter.
    #
    # Since DecisionNode produces a single branch, and we also need a bounded loop,
    # we incorporate the counter + deterministic condition after scoring.
    #
    # We'll set initial round=0 by requiring it as an input at invoke-time.
    # (Validator ensures round is declared upstream; user should seed it when invoking.)
    #
    # To keep schema-free defaults, reads are declared, so users should provide round.
    # But to be robust, we also make DecisionNode only depend on score; round gating uses condition reads.
    #
    # We'll create a small "round_seed" node to set round if missing, using llm output.
    round_seed = AgentNode(
        name="round_seed",
        llm=llm,
        node_prompt=(
            "Initialize the iteration counter.\n"
            "If state already has 'round', keep it; otherwise set round = 0.\n"
            "Return ONLY the integer value for round.\n"
            "Current round in state (may be missing): {round}"
        ),
        reads=["round"],
        writes={"round": int},
    )

    # Assemble: linear start
    start = draft
    start > extract_summary

    # After extracting, score; scoring needs summary scalar plus content
    extract_summary > score_summary

    # Branch 1: initial quality decision
    score_summary > decide_refine
    decide_refine["accept"] > continue_or_done
    decide_refine["refine"] > refine

    # After refine, extract, re-score, then gate again
    refine > extract_summary > score_summary > continue_or_done

    # Gate routing:
    continue_or_done["done"] > extract_summary  # final extract_summary already has final_summary from latest message
    continue_or_done["refine_again"] > inc_round > refine

    # Add round_seed before scoring loop gates by wiring it early:
    # round_seed -> extract_summary -> score_summary
    round_seed > extract_summary

    return AgenticGraph(
        start_node=round_seed,
        end_nodes={extract_summary},
    )
