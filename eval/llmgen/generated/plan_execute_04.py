"""Generated pipeline for task 'plan_execute' (sample 4).

TASK: Make a short plan for a research question, then execute the plan step by step and synthesize the findings into one answer.
"""

from pttai import AgentNode, DecisionNode, ConditionNode, AgenticGraph, fanout


def build_graph(llm):
    # 1) Create a short execution plan (no tools; just structure the approach)
    plan = AgentNode(
        name="plan",
        llm=llm,
        node_prompt=(
            "You are a research planner. Given the user's research question, "
            "produce a short plan to answer it.\n\n"
            "Return 3-6 bullet steps, each starting with an action verb, e.g.:\n"
            "- Identify key terms and scope\n"
            "- Collect authoritative sources\n"
            "- Compare findings\n"
            "- Synthesize a final answer\n\n"
            "Question:\n{messages}"
        ),
    )

    # 2) Decide whether we should do deep reasoning or a lightweight pass
    #    (This is deterministic at build-time in the graph; the LLM will choose among two.)
    route = DecisionNode(
        name="route",
        llm=llm,
        node_prompt=(
            "Decide research depth.\n\n"
            "If the question is narrow and answerable quickly from general knowledge, choose 'light'. "
            "If it is broad, ambiguous, or likely needs careful synthesis, choose 'deep'."
        ),
        choices=["light", "deep"],
    )

    # 3a) Light execution: do a quick research-style response based on the question and plan
    light_exec = AgentNode(
        name="light_exec",
        llm=llm,
        node_prompt=(
            "Execute the following short research plan with a lightweight pass.\n\n"
            "PLAN:\n{messages}\n\n"
            "Now produce:\n"
            "1) A concise set of findings (2-5 bullets)\n"
            "2) A short answer (1-3 paragraphs) that directly addresses the question.\n\n"
            "Research question (same content as input messages):\n{messages}"
        ),
    )

    # 3b) Deep execution: more careful step-by-step analysis + synthesis
    deep_exec = AgentNode(
        name="deep_exec",
        llm=llm,
        node_prompt=(
            "Execute the following research plan with a deep, careful synthesis.\n\n"
            "PLAN:\n{messages}\n\n"
            "Now produce:\n"
            "1) Key sub-questions (2-4)\n"
            "2) Findings for each sub-question (bullets)\n"
            "3) Cross-cutting themes / disagreements (bullets)\n"
            "4) A synthesized final answer (1-3 paragraphs) with clear takeaways.\n\n"
            "Research question (same content as input messages):\n{messages}"
        ),
    )

    # 4) Final synthesis node (one answer regardless of route)
    synth = AgentNode(
        name="synth",
        llm=llm,
        node_prompt=(
            "You are the research analyst. Using the executed work below, write ONE final answer.\n\n"
            "Constraints:\n"
            "- Be accurate and specific.\n"
            "- Keep it short (about 8-14 sentences).\n"
            "- Include a brief 'So what?' takeaway at the end.\n\n"
            "Executed work:\n{messages}"
        ),
    )

    # Wiring:
    # - plan runs first
    # - route decides light vs deep
    # - both branches converge to synth
    #
    # Note: DecisionNode writes its choice to built-in `decision` and routes immediately.
    plan > route
    route["light"] > light_exec > synth
    route["deep"] > deep_exec > synth

    return AgenticGraph(start_node=plan, end_nodes={synth})
