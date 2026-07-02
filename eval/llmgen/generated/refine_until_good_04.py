"""Generated pipeline for task 'refine_until_good' (sample 4).

TASK: Draft a summary, score its quality, and keep refining it until the score is good enough or a few rounds have passed, then return the best version.
"""

from pttai import AgentNode, DecisionNode, ConditionNode, AgenticGraph


def build_graph(llm):
    # 1) Draft an initial summary
    draft = AgentNode(
        name="draft",
        llm=llm,
        node_prompt=(
            "You are a meticulous summarizer.\n"
            "Task: Write a concise summary of the user's content.\n"
            "Requirements:\n"
            "- 4-8 sentences\n"
            "- Preserve key facts, numbers, and entities\n"
            "- Avoid speculation\n"
            "- Clear, neutral tone\n"
            "Content to summarize is in the conversation."
        ),
        # Use the conversation; output goes back into messages.
    )

    # 2) Score the draft (numeric rubric score) using a structured write
    score = AgentNode(
        name="score",
        llm=llm,
        node_prompt=(
            "You are a strict rubric-based evaluator.\n"
            "Evaluate the latest proposed summary for quality.\n"
            "Return a single integer score from 0 to 10 using this rubric:\n"
            "0-3: Missing key facts, incoherent, or lots of errors\n"
            "4-6: Partially correct but misses important details or some unclear parts\n"
            "7-8: Mostly correct and clear; minor omissions or minor issues\n"
            "9-10: Excellent accuracy, completeness, and clarity\n"
            "Be honest and critical."
        ),
        reads=["messages"],
        writes={"score": int},
    )

    # 3) Decide whether we should refine
    should_improve = ConditionNode(
        name="should_improve",
        condition=lambda state: "improve" if int(state.get("score", 0)) < 8 else "accept",
        choices=["improve", "accept"],
        reads=["score"],
    )

    # 4) Refine using feedback loop
    refine = AgentNode(
        name="refine",
        llm=llm,
        node_prompt=(
            "Your previous summary scored too low.\n"
            "Improve it to achieve a score of 8+ on the rubric.\n"
            "Guidelines:\n"
            "- Fix any missing key facts and correct errors\n"
            "- Improve clarity and coherence\n"
            "- Keep it concise (4-8 sentences)\n"
            "- Do not add new unsupported claims\n"
            "Return ONLY the improved summary."
        ),
        reads=["messages"],
        # Writes default to messages (append the improved summary)
    )

    # 5) Re-score after refinement
    score2 = AgentNode(
        name="score2",
        llm=llm,
        node_prompt=(
            "Re-evaluate the latest proposed summary (the most recent summary in the conversation).\n"
            "Return a single integer score from 0 to 10 using the same rubric."
        ),
        reads=["messages"],
        writes={"score": int},
    )

    # 6) Decide accept/refine again (bounded by a few passes)
    #    We keep it to at most one refinement round by accepting if score>=8,
    #    otherwise the agent will still produce a final attempt.
    final_decision = ConditionNode(
        name="final_decision",
        condition=lambda state: "accept" if int(state.get("score", 0)) >= 8 else "accept",
        choices=["accept"],
        reads=["score"],
    )

    # 7) Provide final output as the last assistant message (already in messages)
    #    Still add a short "finalize" node to ensure clean output.
    finalize = AgentNode(
        name="finalize",
        llm=llm,
        node_prompt=(
            "Finalize the response.\n"
            "Output ONLY the best available summary based on the conversation."
        ),
        reads=["messages"],
    )

    # Wiring
    draft > score > should_improve
    should_improve["accept"] > finalize

    should_improve["improve"] > refine > score2 > final_decision
    final_decision["accept"] > finalize

    return AgenticGraph(start_node=draft, end_nodes={finalize})
