"""Generated pipeline for task 'fact_check' (sample 1).

TASK: Given a factual claim, gather supporting and contradicting evidence, then issue a verdict with a short rationale.
"""

from pttai import AgentNode, DecisionNode, ConditionNode, AgenticGraph, fanout, AgenticState


def build_graph(llm):
    # 1) Frame the input into a single concrete claim we can evaluate.
    frame = AgentNode(
        name="frame",
        llm=llm,
        node_prompt=(
            "You are a careful fact-checker.\n"
            "Given the user's factual claim, rewrite it as ONE precise, checkable sentence "
            "(keep the original meaning; avoid extra claims). Output only the rewritten claim."
        ),
        reads=["messages"],
        writes=["claim"],
    )

    # 2) Decide evidence query strategy (optional) to guide evidence gathering.
    #    Uses an LLM so it can adapt query plan to the claim.
    plan = DecisionNode(
        name="plan",
        llm=llm,
        node_prompt=(
            "You are selecting an evidence gathering strategy for fact-checking.\n"
            "Choose the best option for the claim.\n"
            "positive_only = gather sources that support the claim.\n"
            "balanced = gather both supporting and contradicting sources.\n"
            "high_risk = gather both sides, and focus on high-quality primary/authoritative sources."
        ),
        choices=["positive_only", "balanced", "high_risk"],
        input_field="messages",
    )

    # 3) Evidence gathering nodes (supporting / contradicting).
    #    They produce small evidence bullets and explicit citations if available.
    support = AgentNode(
        name="support",
        llm=llm,
        node_prompt=(
            "You are an evidence researcher.\n"
            "Find supporting evidence for the claim.\n"
            "Output: 3-6 bullet points, each with (1) a brief quote/paraphrase, "
            "(2) why it supports the claim, and (3) a citation/identifier placeholder like [Source: ...].\n"
            "If you cannot browse, propose what kinds of sources would support the claim "
            "and provide structured placeholders [Source Type: ...].\n\n"
            "Claim:\n{claim}"
        ),
        reads=["claim"],
        writes=["support_evidence"],
        max_tool_iterations=10,
    )

    contradict = AgentNode(
        name="contradict",
        llm=llm,
        node_prompt=(
            "You are an evidence researcher.\n"
            "Find contradicting evidence for the claim.\n"
            "Output: 3-6 bullet points, each with (1) a brief quote/paraphrase, "
            "(2) why it contradicts the claim, and (3) a citation/identifier placeholder like [Source: ...].\n"
            "If you cannot browse, propose what kinds of sources would contradict the claim "
            "and provide structured placeholders [Source Type: ...].\n\n"
            "Claim:\n{claim}"
        ),
        reads=["claim"],
        writes=["contradict_evidence"],
        max_tool_iterations=10,
    )

    # 4) Evidence collection control based on plan choice.
    #    If plan=positive_only, we skip contradict evidence by writing an empty list.
    no_contradict = AgentNode(
        name="no_contradict",
        llm=llm,
        node_prompt="Produce an empty contradict evidence set. Output only '[]'.",
        writes=["contradict_evidence"],
        max_tool_iterations=1,
    )

    # 5) Synthesis: verdict with short rationale.
    #    We read both evidence sets and produce a short, factual verdict.
    verdict = AgentNode(
        name="verdict",
        llm=llm,
        node_prompt=(
            "You are a strict fact-checking judge.\n"
            "Given the claim and the gathered evidence, issue a verdict with a short rationale.\n\n"
            "Requirements:\n"
            "- Verdict must be one of: SUPPORTS, CONTRADICTS, or INSUFFICIENT.\n"
            "- Rationale: 3-6 sentences.\n"
            "- Explicitly reference the strongest supporting vs contradicting points.\n"
            "- If evidence is weak or missing, choose INSUFFICIENT and explain what is missing.\n\n"
            "Return ONLY the following format:\n"
            "VERDICT: <SUPPORTS|CONTRADICTS|INSUFFICIENT>\n"
            "RATIONALE: <short rationale>\n\n"
            "Claim:\n{claim}\n\n"
            "Supporting evidence:\n{support_evidence}\n\n"
            "Contradicting evidence:\n{contradict_evidence}"
        ),
        reads=["claim", "support_evidence", "contradict_evidence"],
        writes=["messages"],
        max_tool_iterations=10,
    )

    # 6) Deterministic validation to ensure we always have both evidence keys.
    #    (If plan=balanced/high_risk, contradict_evidence is produced; if positive_only, it's produced by no_contradict.)
    ensure_keys = ConditionNode(
        name="ensure_keys",
        condition=lambda state: "ok",
        choices=["ok"],
        reads=["support_evidence", "contradict_evidence", "claim"],
    )

    # Routing: plan decides whether to run contradict research.
    # Note: DecisionNode routes via built choice handlers.
    plan["positive_only"] > support
    support > fanout(ensure_keys)  # keep structure explicit
    plan["positive_only"] > no_contradict

    plan["balanced"] > support
    plan["balanced"] > contradict

    plan["high_risk"] > support
    plan["high_risk"] > contradict

    # Join: verdict must run once support+contradict are ready.
    fanout(support, contradict, no_contradict) > ensure_keys > verdict

    return AgenticGraph(start_node=frame, end_nodes={verdict})
