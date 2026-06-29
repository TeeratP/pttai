"""Shared read-side helpers for multi-key state IO (reads/writes)."""

from langchain_core.messages import BaseMessage


def partition_reads(state, keys):
    """Split the requested read keys into conversation history and scalars.

    Dispatch is by runtime VALUE type, not key name (a custom key can hold a
    message list): a non-empty list of BaseMessage is treated as conversation
    history (all such reads are concatenated in declared order); anything else
    is a scalar collected into a dict for prompt interpolation.

    Args:
        state: the current graph state.
        keys: the state keys this node reads, in declared order.

    Returns:
        (history, scalars): the concatenated message history and the scalar dict.

    Raises:
        ValueError: if any requested key is missing from state.
    """
    history = []
    scalars = {}
    for k in keys:
        if k not in state:
            raise ValueError(f"State must contain a {k!r} key")
        v = state[k]
        if isinstance(v, list) and v and all(isinstance(m, BaseMessage) for m in v):
            history += v
        else:
            scalars[k] = v
    return history, scalars
