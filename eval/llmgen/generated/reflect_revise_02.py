"""Generated pipeline for task 'reflect_revise' (sample 2).

TASK: Draft an answer, then critique the draft, then revise the answer using the critique before returning it.
"""

from pttai import AgentNode, DecisionNode, ConditionNode, AgenticGraph, fanout


def build_graph(llm):
    # 1) Draft an answer to the user's request
    draft = AgentNode(
        name="draft",
        llm=llm,
        node_prompt=(
            "You are a careful assistant. Draft a complete, accurate answer to the user. "
            "If information is missing, make the best assumptions and clearly label them. "
            "Use concise structure (headings or bullets if helpful)."
        ),
        # Use incoming user content from the conversation history:
        # (pttai default reads/input_field is "messages", and it appends replies back)
        # This node's output will be added to state["messages"] by default.
    )

    # 2) Critique the draft
    critique = AgentNode(
        name="critique",
        llm=llm,
        node_prompt=(
            "You are a reviewer. Critique the assistant's last draft for:\n"
            "1) factuality (what might be wrong or unsupported),\n"
            "2) completeness,\n"
            "3) clarity and structure,\n"
            "4) any safety/policy issues,\n"
            "5) whether it answers the user's question directly.\n"
            "Be specific and list concrete fixes."
        ),
    )

    # 3) Decide whether revision is needed (LLM-based constrained routing)
    needs_revision = DecisionNode(
        name="needs_revision",
        llm=llm,
        node_prompt=(
            "Decide whether the draft should be revised after the critique. "
            "Return one choice: 'revise' if meaningful improvements are needed; "
            "otherwise return 'keep'."
        ),
        choices=["keep", "revise"],
        # Read conversation history (default input_field="messages") so it can compare draft vs critique.
    )

    # 4a) If keeping, finalize by presenting the draft (plus minimal rationale)
    keep_final = AgentNode(
        name="keep_final",
        llm=llm,
        node_prompt=(
            "Based on the critique, decide to KEEP the draft. "
            "Return the final answer to the user. "
            "Only include brief notes about why no revision was necessary."
        ),
    )

    # 4b) If revising, produce an improved answer
    revise_final = AgentNode(
        name="revise_final",
        llm=llm,
        node_prompt=(
            "Based on the critique, revise the draft into an improved final answer. "
            "Make sure the final answer directly addresses the user's question. "
            "If you make assumptions, label them; remove unsupported claims."
        ),
    )

    # Wire: draft -> (critique in parallel-ready slot) -> needs_revision -> choice handlers
    # We keep it simple and sequential: draft then critique then route.
    draft > critique > needs_revision
    needs_revision["keep"] > keep_final
    needs_revision["revise"] > revise_final

    return AgenticGraph(start_node=draft, end_nodes={keep_final, revise_final})
