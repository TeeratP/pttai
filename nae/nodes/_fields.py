"""Shared read-side helpers for multi-key state IO (reads/writes)."""

import string
import typing

from langchain_core.messages import AnyMessage, BaseMessage
from langgraph.graph.message import add_messages


def partition_reads(state, keys):
    """Split the requested read keys into conversation history and scalars.

    Dispatch is by runtime VALUE type, not key name (a custom key can hold a
    message list): a list of BaseMessage (including an EMPTY list, vacuously) is
    treated as conversation history (all such reads are concatenated in declared
    order); anything else is a scalar collected into a dict for prompt
    interpolation.

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
        if isinstance(v, list) and all(isinstance(m, BaseMessage) for m in v):
            history += v
        else:
            scalars[k] = v
    return history, scalars


def prompt_placeholders(prompt: str) -> set:
    """The `{name}` field names referenced in `prompt` (`str.format` style).

    Uses `string.Formatter().parse` so escaped braces (`{{` / `}}`) are
    ignored and only the base name of dotted/indexed fields (`{a.b}`/`{a[0]}`)
    is returned. Positional/empty fields (`{}`/`{0}`) are skipped.
    """
    names = set()
    for _, field, _, _ in string.Formatter().parse(prompt):
        if not field:
            continue
        base = field.split(".")[0].split("[")[0]
        if base:  # skip positional `{}` / `{0}`
            names.add(base)
    return names


def is_history_annotation(annotation) -> bool:
    """True if a state schema annotation is a message-list/history channel
    (reducer is `add_messages`, or the type is `list[AnyMessage|BaseMessage]`).
    Such reads become conversation history and are never interpolated."""
    meta = getattr(annotation, "__metadata__", ())
    if any(m is add_messages for m in meta):
        return True
    base = typing.get_args(annotation)[0] if meta else annotation
    if typing.get_origin(base) in (list, typing.List):
        args = typing.get_args(base)
        if args and args[0] in (AnyMessage, BaseMessage):
            return True
    return False
