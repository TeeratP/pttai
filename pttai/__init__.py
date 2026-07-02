"""pttai — Pythonic Topology Tools for AI.

Write agent workflows in the ergonomic layer you *want* to write in — wire nodes
with the `>` operator, fan out with `fanout(...)` / `[a, b]`, map-reduce with
`node.map("field")` — and `AgenticGraph` compiles it down to a native LangGraph
`StateGraph`. No lock-in: the whole LangGraph ecosystem (streaming, async,
checkpointers, LangSmith) is right underneath.

Public API::

    from pttai import (
        AgenticGraph,   # compiles the wired nodes into a LangGraph StateGraph
        AgentNode,      # LLM node: tool-call loop, typed multi-key reads/writes
        DecisionNode,   # constrained structured-output routing
        ConditionNode,  # deterministic routing by a Python predicate (no LLM)
        HumanNode,      # resumable human-in-the-loop (interrupt/resume)
        fanout,         # explicit parallel fan-out: a > fanout(b, c) > d
        AgenticState,   # default reduced-channel state (messages / log / token)
    )
"""

from pttai.graph import AgenticGraph
from pttai.node import fanout
from pttai.nodes import AgentNode, DecisionNode, ConditionNode, HumanNode
from pttai.state import AgenticState

__version__ = "0.1.3"

__all__ = [
    "AgenticGraph",
    "AgentNode",
    "DecisionNode",
    "ConditionNode",
    "HumanNode",
    "fanout",
    "AgenticState",
]
