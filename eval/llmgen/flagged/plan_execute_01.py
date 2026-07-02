"""Generated pipeline for task 'plan_execute' (sample 1).

TASK: Make a short plan for a research question, then execute the plan step by step and synthesize the findings into one answer.
"""

from pttai import AgentNode, AgenticGraph, fanout


def build_graph(llm):
    # 1) Turn the research question into a concrete, step-by-step plan.
    planner = AgentNode(
        name="planner",
        llm=llm,
        node_prompt=(
            "You are a research planner.\n"
            "Given the user's research question, produce a short execution plan.\n"
            "Requirements:\n"
            "- Start with the exact question restated in 1 sentence.\n"
            "- Provide 3 to 6 numbered steps.\n"
            "- Each step should be concrete about what to look up or decide.\n"
            "- Keep it concise (total < 150 words).\n"
        ),
        # read the user question from the message text
        input_field="messages",
    )

    # 2) Execute the plan in parallel using different perspectives.
    # (We pass the evolving conversation via messages, so each worker has context.)
    worker1 = AgentNode(
        name="exec_1",
        llm=llm,
        node_prompt=(
            "You are Research Executor #1.\n"
            "Using the plan and the conversation so far, carry out the most relevant steps.\n"
            "Deliverables:\n"
            "- Bullet list of key findings (5-10 bullets).\n"
            "- Include any caveats/assumptions.\n"
            "- Do NOT overclaim; be explicit about uncertainty.\n"
            "Keep < 200 words."
        ),
        input_field="messages",
    )

    worker2 = AgentNode(
        name="exec_2",
        llm=llm,
        node_prompt=(
            "You are Research Executor #2 (evidence-minded).\n"
            "Using the plan and the conversation so far, focus on corroboration:\n"
            "- What evidence would confirm or falsify claims?\n"
            "- Note definitions, key variables, and what sources to prefer.\n"
            "Deliverables:\n"
            "- Bullet list of corroborating findings and suggested checks.\n"
            "- Include caveats/uncertainty.\n"
            "Keep < 200 words."
        ),
        input_field="messages",
    )

    worker3 = AgentNode(
        name="exec_3",
        llm=llm,
        node_prompt=(
            "You are Research Executor #3 (gap-spotter).\n"
            "Using the plan and the conversation so far, identify:\n"
            "- Missing pieces or competing explanations.\n"
            "- Likely misunderstandings or confounds.\n"
            "Deliverables:\n"
            "- Bullet list of gaps, risks, and what would change the conclusion.\n"
            "Keep < 200 words."
        ),
        input_field="messages",
    )

    # 3) Synthesize everything into one short answer.
    synthesizer = AgentNode(
        name="synthesizer",
        llm=llm,
        node_prompt=(
            "You are the research synthesizer.\n"
            "Given:\n"
            "- the planner's plan\n"
            "- executor #1 findings\n"
            "- executor #2 corroboration checks\n"
            "- executor #3 gaps/risks\n"
            "Produce ONE final answer to the original research question.\n"
            "Requirements:\n"
            "- Start with a 1-2 sentence direct answer.\n"
            "- Then provide 3-6 concise bullet points summarizing the evidence/reasoning.\n"
            "- End with a brief 'Uncertainty / next checks' section (1-3 bullets).\n"
            "- Total < 220 words.\n"
        ),
        input_field="messages",
    )

    # One-line topology: plan -> parallel execution -> synthesis
    graph = AgenticGraph(
        start_node=planner,
        end_nodes={synthesizer},
    )

    planner > fanout(worker1, worker2, worker3) > synthesizer
    return graph
