"""Generated pipeline for task 'plan_execute' (sample 3).

TASK: Make a short plan for a research question, then execute the plan step by step and synthesize the findings into one answer.
"""

from pttai import AgentNode, AgenticGraph, DecisionNode, fanout


def build_graph(llm):
    # 1) Frame the task into a concrete research plan
    plan_maker = AgentNode(
        name="plan_maker",
        llm=llm,
        node_prompt=(
            "You are a research planner.\n"
            "Given the user's research question, produce a short plan with 3-6 steps.\n"
            "Output format:\n"
            "Plan:\n"
            "1) ...\n"
            "2) ...\n"
            "3) ...\n"
            "Each step should be actionable and should specify what to look for."
        ),
    )

    # 2) Decide how to proceed (direct answer vs. deeper investigation)
    decide_depth = DecisionNode(
        name="decide_depth",
        llm=llm,
        node_prompt=(
            "Decide whether the research question can be answered with a brief synthesis "
            "or needs deeper investigation.\n\n"
            "Choose:\n"
            "- 'brief': you can answer directly with general knowledge and light reasoning\n"
            "- 'deep': you should gather more evidence by following the plan steps\n\n"
            "Be conservative: if uncertain, choose 'deep'."
        ),
        choices=["brief", "deep"],
    )

    # 3A) Brief route: answer immediately with a structured response
    brief_answer = AgentNode(
        name="brief_answer",
        llm=llm,
        node_prompt=(
            "You are a research synthesizer.\n"
            "Using the user's question, write a concise answer.\n\n"
            "Constraints:\n"
            "- 1 short paragraph + 3-5 bullet points\n"
            "- If you make assumptions, list them as bullets\n"
            "- No citations required"
        ),
    )

    # 3B) Deep route: execute the plan step-by-step, then synthesize
    # Executor: expands each plan step into an intermediate finding.
    step_executor = AgentNode(
        name="step_executor",
        llm=llm,
        node_prompt=(
            "You are executing a research plan step by step.\n\n"
            "Input includes the user's question and the plan.\n"
            "Produce TWO things:\n"
            "1) A small set of findings for the first 1-2 steps of the plan (no more).\n"
            "2) A 'Remaining plan' section listing the remaining steps to do later.\n\n"
            "Output format:\n"
            "Findings:\n"
            "- ...\n"
            "Remaining plan:\n"
            "1) ...\n"
            "2) ...\n"
        ),
        max_tool_iterations=5,
    )

    # Second executor pass to continue remaining plan
    step_executor_2 = AgentNode(
        name="step_executor_2",
        llm=llm,
        node_prompt=(
            "Continue executing the remaining research plan steps step by step.\n\n"
            "Use the previous findings and the remaining plan.\n"
            "Produce findings for the next 1-2 steps, and then provide:\n"
            "- A final 'All findings' consolidated list\n"
            "- 'Open questions' if anything critical is still missing\n\n"
            "Output format:\n"
            "All findings:\n"
            "- ...\n"
            "Open questions:\n"
            "- ... (or 'None')\n"
        ),
        max_tool_iterations=5,
    )

    # Final synthesis
    synthesize = AgentNode(
        name="synthesize",
        llm=llm,
        node_prompt=(
            "You are a research synthesizer.\n"
            "Synthesize the research findings into ONE final answer.\n\n"
            "Requirements:\n"
            "- 1 paragraph synthesis\n"
            "- Then 3-6 bullet points: key takeaways\n"
            "- If there are open questions, include a brief 'What to check next' list\n"
        ),
    )

    # Panel-like parallelism for verification (lightweight): critique + evidence checklist
    critique = AgentNode(
        name="critique",
        llm=llm,
        node_prompt=(
            "You are a skeptical reviewer.\n"
            "Given the question and the synthesized findings (or partial findings),\n"
            "list the top 2-3 weaknesses or assumptions that could reduce reliability.\n"
            "Output:\n"
            "- Weakness 1: ...\n"
            "- Weakness 2: ...\n"
            "- Weakness 3 (optional): ...\n"
        ),
    )

    evidence_check = AgentNode(
        name="evidence_check",
        llm=llm,
        node_prompt=(
            "You are an evidence checklist.\n"
            "Given the question and the synthesized findings,\n"
            "produce a short checklist of what types of evidence would support the claims\n"
            "and which claims are most/least supported.\n\n"
            "Output format:\n"
            "Most supported:\n"
            "- ...\n"
            "Least supported:\n"
            "- ...\n"
            "Evidence to look for next:\n"
            "- ...\n"
        ),
    )

    # A final integrator that merges synthesis with critique and evidence checklist.
    final_answer = AgentNode(
        name="final_answer",
        llm=llm,
        node_prompt=(
            "Integrate the synthesized answer with the critique and evidence checklist.\n"
            "Produce the final response to the user.\n\n"
            "Output:\n"
            "Final Answer (one paragraph):\n"
            "...\n\n"
            "Key Takeaways:\n"
            "- ...\n\n"
            "What to Check Next (optional):\n"
            "- ...\n"
        ),
    )

    # Wire graph:
    # start -> plan_maker -> decide_depth
    # if brief -> brief_answer -> final_answer (as only synthesis input)
    # if deep -> step_executor -> step_executor_2 -> synthesize
    # then in both cases, run critique + evidence_check in parallel, join to final_answer
    #
    # Note: For the deep route, we go through synthesize before critique/evidence_check.
    # For the brief route, we still pass through synthesize-like final steps via brief_answer
    # directly into critique/evidence_check (they will adapt to what's provided).
    decide_depth["brief"] > brief_answer
    decide_depth["deep"] > step_executor
    plan_maker > decide_depth

    # Deep route continuation
    step_executor > step_executor_2 > synthesize

    # Both routes converge into the critique/evidence_check fanout
    # (we connect synthesize and brief_answer as inputs to the fanout join node)
    # We'll do: (brief_answer > fanout(critique, evidence_check) > final_answer)
    # and: (synthesize > fanout(critique, evidence_check) > final_answer)
    #
    # But the same critique/evidence_check nodes can't have two parents unless the framework allows
    # concurrent writes to messages; here both routes are exclusive by DecisionNode, so it is safe.
    # Still, we wire both to the same fanout input by chaining through an explicit gate node.
    gate = AgentNode(
        name="gate",
        llm=llm,
        node_prompt=(
            "Gate node. Pass through the best available draft answer/finding(s) to support later review.\n"
            "Do not add new content beyond lightly normalizing into a coherent research notes form."
        ),
    )

    brief_answer > gate
    synthesize > gate

    gate > fanout(critique, evidence_check) > final_answer

    return AgenticGraph(start_node=plan_maker, end_nodes={final_answer})
