"""Default state schema and channel reducers for pttai.

Defines `AgenticState`, the `TypedDict` of reduced channels every graph uses
unless you pass your own `state=`: `messages` (the conversation, via LangGraph's
`add_messages`), `log` (per-node trace lines, concatenated), and `token`
(per-model usage totals). The reducers are the merge functions that make
parallel branches, subgraph composition, and checkpointing correct rather than
racy — `merge_token_usage` deep-sums usage across every LLM call, and
`accumulate` collects one value per parallel writer into a list (used for
auto-registered map-reduce collection channels). `RESERVED` lists the
framework-managed channels a user schema/invoke must not clobber.
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


# Framework-managed channels. They are written/seeded internally and must not be
# created or overwritten by USER node declarations or invoke extra kwargs.
# (`messages` is exempt from the user-declaration guard — it is the standard
# conversation channel and the default node read/write — but it is still
# supplied through the dedicated input/`message=` path, not as an extra kwarg.)
RESERVED = {"messages", "log", "token"}


def _deep_sum(a: dict, b: dict) -> dict:
    """Recursively add `b` into `a`: numeric fields are summed, nested dicts
    (e.g. `input_token_details`) are deep-summed, other values take `b`."""
    out = dict(a)
    for k, v in b.items():
        cur = out.get(k)
        if isinstance(cur, dict) and isinstance(v, dict):
            out[k] = _deep_sum(cur, v)
        elif isinstance(cur, (int, float)) and isinstance(v, (int, float)):
            out[k] = cur + v
        else:
            out[k] = v
    return out


def merge_token_usage(left, right) -> dict:
    """Reducer for the `token` channel: merge two `{model_name: usage}` dicts.

    Union by model key; when the same model appears on both sides its usage
    breakdown is deep-summed (top-level token counts AND nested `*_details`).
    Associative/commutative so parallel fan-in merges cleanly. A missing/None
    `left` (the first update) is treated as `{}`.
    """
    merged = dict(left or {})
    for model, usage in (right or {}).items():
        if model in merged:
            merged[model] = _deep_sum(merged[model], usage)
        else:
            merged[model] = dict(usage)
    return merged


def accumulate(left, right):
    """Reducer for a map-reduce collection channel: each parallel writer
    contributes ONE value, accumulated into a list (arrival order). A
    missing/None left (the first update) is treated as []."""
    return (left or []) + [right]


class AgenticState(TypedDict):
    # add_messages appends new messages, replaces an existing message when IDs
    # match, and coerces bare strings to HumanMessage. It also merges updates
    # from parallel branches instead of clobbering them.
    messages: Annotated[list[AnyMessage], add_messages]
    # operator.add concatenates log fragments from parallel branches.
    log: Annotated[list[str], operator.add]
    # Per-model token tally: {model_name: usage_metadata}. The reducer unions by
    # model and deep-sums a model's usage across every LLM call in the run (incl.
    # tool-loop calls and parallel fan-out), so out["token"] is the run total.
    token: Annotated[dict, merge_token_usage]
