"""Generated pipeline for task 'refine_until_good' (sample 3).

TASK: Draft a summary, score its quality, and keep refining it until the score is good enough or a few rounds have passed, then return the best version.
"""

from typing import List, Dict, Any
from pttai import AgentNode, ConditionNode, DecisionNode, AgenticGraph, fanout, AgenticState


def build_graph(llm):
    # --- Step 1: Frame the task (extract what to summarize) ---
    frame = AgentNode(
        name="frame",
        llm=llm,
        node_prompt=(
            "You are an expert task framer.\n"
            "Given the user's request and any provided draft/context in the conversation,\n"
            "produce:\n"
            "1) what content/topic should be summarized (1-3 keywords or a short description)\n"
            "2) target audience (who it is for)\n"
            "3) required length and format constraints (e.g., 5 bullets, ~120 words, etc.)\n"
            "Return the result as a concise, structured draft in plain text.\n"
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # --- Step 2: Draft multiple candidate summaries in parallel ---
    # These should be plausibly different styles to give something to score.
    draft_plain = AgentNode(
        name="draft_plain",
        llm=llm,
        node_prompt=(
            "Write a high-quality summary following the framed constraints.\n"
            "Style: clear, direct, neutral tone.\n"
            "Use only information from the provided conversation context.\n"
            "Keep the required length/format.\n"
        ),
        reads=["messages"],
        writes=["messages"],
    )

    draft_bullets = AgentNode(
        name="draft_bullets",
        llm=llm,
        node_prompt=(
            "Write a high-quality summary following the framed constraints.\n"
            "Style: bullet points (concise bullets), neutral tone.\n"
            "Use only information from the provided conversation context.\n"
            "Keep the required length/format.\n"
        ),
        reads=["messages"],
        writes=["messages"],
    )

    draft_executive = AgentNode(
        name="draft_executive",
        llm=llm,
        node_prompt=(
            "Write a high-quality summary following the framed constraints.\n"
            "Style: executive brief (1 short paragraph, then 3 key takeaways).\n"
            "Use only information from the provided conversation context.\n"
            "Keep the required length/format.\n"
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # --- Step 3: Score quality of the candidate summaries ---
    # We'll ask for a structured score and a short diagnosis. We'll store score as int.
    # Note: structured writes with tools are not used; this node doesn't use tools.
    scorer = AgentNode(
        name="scorer",
        llm=llm,
        node_prompt=(
            "You are a strict evaluator.\n"
            "Based on the conversation (including the candidate drafts),\n"
            "evaluate the best candidate summary quality.\n"
            "Use the following rubric:\n"
            "- Fidelity: avoids adding unsupported claims\n"
            "- Relevance: captures the key points\n"
            "- Clarity: easy to understand, well organized\n"
            "- Format compliance: matches the requested format/length\n"
            "- Conciseness: no unnecessary filler\n"
            "Return:\n"
            "1) score as an integer from 0 to 100\n"
            "2) brief reasons (2-4 bullets)\n"
            "3) specific improvement instructions for revision (2-3 bullets)\n"
            "Be honest and conservative.\n"
        ),
        # Extract score into typed structured output and keep the rest in messages history.
        # We rely on the model to return a format compatible with the structured schema write.
        reads=["messages"],
        writes={"score": int},
    )

    # --- Step 4: Decide whether to refine based on score ---
    # ConditionNode is deterministic, so we gate on the typed integer score.
    # Provide default safety threshold.
    def should_refine(state: AgenticState) -> str:
        score = state.get("score", 0)
        return "refine" if score < 80 else "accept"

    gate = ConditionNode(
        name="gate",
        condition=should_refine,
        choices=["refine", "accept"],
        reads=["score"],
    )

    # --- Step 5: Refine summaries using the scorer diagnosis ---
    # We can't read the scorer's diagnosis directly (it was written into messages history),
    # so we instruct the model to leverage the latest messages as both candidates and evaluator notes.
    refine = AgentNode(
        name="refine",
        llm=llm,
        node_prompt=(
            "Revise the summary to improve quality based on the evaluator feedback in the conversation.\n"
            "Keep the requested format/length constraints.\n"
            "Do not introduce unsupported facts.\n"
            "Output ONLY the improved final summary.\n"
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # If we refine, re-score. This yields a few iterative rounds.
    # We'll cap iterations by using AgentNode's tool-iteration cap? Not applicable.
    # Instead, we rely on the graph structure repeating a finite loop by rewiring with end_nodes.
    # We'll implement a bounded two-pass refinement by adding a second gate variable via an iteration counter.
    # Since we must use only pttai primitives and no custom state mutation, we do a bounded design:
    # one refinement round only (good enough for most cases).
    #
    # Accept will end. Refine will then score and end.
    # --- Step 6: Final score + accept ---
    final_scorer = AgentNode(
        name="final_scorer",
        llm=llm,
        node_prompt=(
            "Final strict evaluation of the latest summary in the conversation.\n"
            "Return a single integer score 0-100 as 'score'.\n"
            "No extra text."
        ),
        reads=["messages"],
        writes={"score": int},
    )

    accept = AgentNode(
        name="accept",
        llm=llm,
        node_prompt=(
            "Produce the final answer for the user:\n"
            "1) the best final summary (as requested)\n"
            "If the conversation contains an improved summary, output that.\n"
            "Output only the final summary text."
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # --- Wire graph ---
    # Parallel drafts join -> scorer -> gate -> either accept or refine -> final_scorer -> accept
    # Ensure a proper terminal: only `accept` is the end node.
    #
    # Note: gate choice routing uses indexing: gate["accept"] > accept, gate["refine"] > refine
    graph = (
        frame
        > fanout(draft_plain, draft_bullets, draft_executive)
        > scorer
    )

    # Branch wiring
    graph = graph > gate
    graph = graph["accept"] > accept
    graph = graph["refine"] > refine > final_scorer > accept

    return AgenticGraph(start_node=frame, end_nodes={accept})
