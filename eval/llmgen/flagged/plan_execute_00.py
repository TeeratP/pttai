"""Generated pipeline for task 'plan_execute' (sample 0).

TASK: Make a short plan for a research question, then execute the plan step by step and synthesize the findings into one answer.
"""

from pttai import AgentNode, AgenticGraph, fanout, DecisionNode, ConditionNode

def build_graph(llm):
    # 1) Frame the research into a concrete plan (and optional scope)
    planner = AgentNode(
        name="planner",
        llm=llm,
        node_prompt=(
            "You are an expert research planner.\n"
            "Given the user's research question, create a concise research plan.\n"
            "Requirements:\n"
            "- Output 3-6 numbered steps.\n"
            "- Each step must say what to look for and why it matters.\n"
            "- End by stating what evidence would be sufficient to answer.\n"
            "Keep it short.\n"
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # 2) Decide whether we should do deeper research steps
    #    (This is LLM-driven so it can adapt to the user's question.)
    route = DecisionNode(
        name="route",
        llm=llm,
        node_prompt=(
            "Decide how to proceed with the research based on the latest user question "
            "and the proposed plan.\n"
            "If the question is likely answerable quickly with general knowledge, choose 'quick'.\n"
            "If it needs careful evidence gathering or multi-step reasoning, choose 'deep'.\n"
            "Return only the choice."
        ),
        choices=["quick", "deep"],
        input_field="messages",
    )

    # 3a) Quick path: light synthesis
    quick_research = AgentNode(
        name="quick_research",
        llm=llm,
        node_prompt=(
            "You are conducting a quick literature-free research pass.\n"
            "Using the provided research plan and the research question, "
            "draft the best available answer using general knowledge and reasoning.\n"
            "If you are uncertain, explicitly state assumptions.\n"
            "Output:\n"
            "- Key points (bullets)\n"
            "- One paragraph synthesis"
        ),
        writes=["messages"],
        reads=["messages"],
    )

    # 3b) Deep path: step-by-step execution and then synthesis
    #     (No external retriever is used; we still execute the plan step-by-step
    #      by iteratively reasoning and organizing evidence/arguments.)
    step_1 = AgentNode(
        name="step_1",
        llm=llm,
        node_prompt=(
            "Execute Research Step 1 from the plan.\n"
            "Return only:\n"
            "- What you looked for / what you decided to consider\n"
            "- 3-6 bullet findings relevant to the question\n"
        ),
        reads=["messages"],
        writes=["messages"],
    )

    step_2 = AgentNode(
        name="step_2",
        llm=llm,
        node_prompt=(
            "Execute Research Step 2 from the plan.\n"
            "Return only:\n"
            "- What you looked for / what you decided to consider\n"
            "- 3-6 bullet findings relevant to the question\n"
        ),
        reads=["messages"],
        writes=["messages"],
    )

    step_3 = AgentNode(
        name="step_3",
        llm=llm,
        node_prompt=(
            "Execute Research Step 3 from the plan.\n"
            "Return only:\n"
            "- What you looked for / what you decided to consider\n"
            "- 3-6 bullet findings relevant to the question\n"
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # Join deep findings into a single synthesis
    synth = AgentNode(
        name="synth",
        llm=llm,
        node_prompt=(
            "Synthesize the research findings into one final answer.\n"
            "Output format:\n"
            "1) Final Answer (2-5 sentences)\n"
            "2) Evidence/Reasoning bullets (3-6 bullets)\n"
            "3) Uncertainties or assumptions (optional, if any)\n"
            "Be direct and concise."
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # 4) Quality check: optionally request a revision
    # ConditionNode is deterministic: if the answer looks too short or missing structure, revise.
    # It reads the current conversation from messages and returns a choice.
    def quality_pred(state):
        # Simple heuristic; deterministic and cheap.
        text = "\n".join([m.content for m in state.get("messages", []) if hasattr(m, "content")]).strip()
        # If we don't see the expected sections, ask for revision.
        if "Final Answer" not in text or "Evidence/Reasoning" not in text:
            return "revise"
        # If it's extremely short, also revise.
        if len(text) < 300:
            return "revise"
        return "ok"

    quality = ConditionNode(
        name="quality",
        condition=quality_pred,
        choices=["ok", "revise"],
        reads=["messages"],
    )

    revise = AgentNode(
        name="revise",
        llm=llm,
        node_prompt=(
            "Revise the previous output to match the required format strictly:\n"
            "1) Final Answer (2-5 sentences)\n"
            "2) Evidence/Reasoning bullets (3-6 bullets)\n"
            "3) Uncertainties or assumptions (optional)\n"
            "Keep it concise."
        ),
        reads=["messages"],
        writes=["messages"],
    )

    # Wiring:
    # planner -> route
    # route["quick"] -> quick_research -> synth -> quality -> (ok OR revise -> synth)
    # route["deep"]  -> parallel steps -> synth -> quality -> (ok OR revise -> synth)
    graph = AgenticGraph(
        start_node=planner,
        end_nodes={synth},
    )

    planner > route
    route["quick"] > quick_research > synth
    route["deep"] > fanout(step_1, step_2, step_3) > synth
    synth > quality
    quality["ok"] > synth
    quality["revise"] > revise > synth

    return graph
