"""Generated pipeline for task 'plan_execute' (sample 2).

TASK: Make a short plan for a research question, then execute the plan step by step and synthesize the findings into one answer.
"""

from typing import List

from pttai import AgentNode, AgenticGraph, fanout
from pttai import DecisionNode


def build_graph(llm):
    # 1) Plan the research
    make_plan = AgentNode(
        name="make_plan",
        llm=llm,
        reads=["messages"],
        node_prompt=(
            "You are a careful research planner.\n"
            "Given the user's research question, produce a short step-by-step plan "
            "to answer it.\n\n"
            "Requirements:\n"
            "- Start by restating the question in 1 sentence.\n"
            "- Then list 3 to 6 concrete steps.\n"
            "- Prefer primary sources or reputable references.\n"
            "- Keep it concise.\n"
            "Return the plan as a single message."
        ),
        writes={"plan": str},
        # Do not overwrite messages; keep plan in its own typed field.
        input_field="messages",
        output_field="messages",
    )

    # 2) Decide whether to branch into "quick" vs "deep" research
    #    (LLM chooses; this demonstrates DecisionNode usage).
    route_depth = DecisionNode(
        name="route_depth",
        llm=llm,
        node_prompt=(
            "Decide how much research is needed.\n"
            "Choose 'quick' if the question can be answered with a small set of general sources "
            "or common knowledge; choose 'deep' if it requires more thorough sourcing, definitions, "
            "or evidence.\n\n"
            "Base your decision on the plan."
        ),
        choices=["quick", "deep"],
        input_field="messages",
        reads=["plan"],
        # DecisionNode writes its choice into built-in `decision`.
    )

    # 3) Two research strategies running in parallel for robustness.
    #    We'll route depth by wiring both branches; only the chosen branch runs.
    #    Each branch produces findings, then a common synthesizer writes final answer.
    quick_research = AgentNode(
        name="quick_research",
        llm=llm,
        reads=["plan", "messages"],
        node_prompt=(
            "You are executing a QUICK research plan.\n"
            "Use the provided plan to gather key points and a few high-signal references "
            "from your own knowledge.\n\n"
            "Output format:\n"
            "- Key findings (bullet list)\n"
            "- 2-5 references or citations to look up (can be described as sources/authors/years)\n"
            "Be concise but specific. Do NOT invent citations that you can't justify; if unsure, say so."
        ),
        writes={"findings": str},
        input_field="messages",
        output_field="messages",
    )

    deep_research = AgentNode(
        name="deep_research",
        llm=llm,
        reads=["plan", "messages"],
        node_prompt=(
            "You are executing a DEEP research plan.\n"
            "Follow the provided plan to gather more comprehensive evidence, definitions, "
            "and contrasts.\n\n"
            "Output format:\n"
            "- Key findings (bullet list)\n"
            "- What evidence supports each finding (short)\n"
            "- 5-10 references or citations to look up (sources/authors/years). "
            "Do NOT fabricate specific citations; if you are uncertain, note it.\n"
            "- Any open questions or uncertainties."
        ),
        writes={"findings": str},
        input_field="messages",
        output_field="messages",
    )

    # 4) Synthesize into one final answer
    synthesize = AgentNode(
        name="synthesize",
        llm=llm,
        reads=["plan", "findings", "messages"],
        node_prompt=(
            "You are the research analyst.\n"
            "Synthesize the research findings into ONE final answer to the user's question.\n\n"
            "Requirements:\n"
            "- Start with a direct answer (2-4 sentences).\n"
            "- Then provide a brief justification grounded in the findings.\n"
            "- If there are uncertainties, mention them and what would resolve them.\n"
            "- End with a short 'Next steps' (1-3 bullets) for further research.\n"
        ),
        writes={"answer": str},
        input_field="messages",
        output_field="messages",
    )

    # Wire:
    # - First make a plan (writes plan)
    # - Then LLM routes quick vs deep (writes decision)
    # - Only the chosen branch runs and writes findings
    # - Then synthesis writes answer
    plan_node = make_plan
    route_depth > quick_research
    route_depth > deep_research
    # Note: route_depth is already connected by indexing choices below.
    # We'll do correct explicit choice wiring:
    route_depth["quick"] > quick_research
    route_depth["deep"] > deep_research

    # sequential: plan -> decision -> chosen research -> synthesize
    plan_node > route_depth > synthesize

    # Graph entry: start at plan; terminal: synthesize
    return AgenticGraph(start_node=plan_node, end_nodes={synthesize})
