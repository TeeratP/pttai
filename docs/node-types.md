# Node types

All nodes are callables (`__call__(state) -> delta`) invoked by LangGraph with
the shared state. They return **only the keys they update**; the reducers merge
them. Nodes never mutate state in place — that is what keeps checkpointing,
parallel-branch merges, and subgraph composition correct rather than racy.

## The `LLMNode` base

The two LLM-backed nodes — `AgentNode` and `DecisionNode` — share a common
`LLMNode` base that owns the **model binding** and the **tool-call loop**.
Constructing an `LLMNode` subclass normalizes `tools` (a `StructuredTool` /
`BaseTool` is used as-is; a plain callable is wrapped via
`StructuredTool.from_function`) and wires the internal loop capped by
`max_tool_iterations` (default 25). `ConditionNode` and `HumanNode` do not use
an LLM and subclass `Node` directly.

## Overview

| Node | Purpose |
|------|---------|
| **`AgentNode`** | Prepends `node_prompt` as a `SystemMessage`, calls the LLM, returns a delta. Pass `tools=[...]` to wrap bare callables as `StructuredTool`s and run an internal tool-call loop. `reads` / `writes` give typed multi-key IO. Optional `reasoning_effort` for gpt-5.x. |
| **`DecisionNode`** | LLM branching. Reads from `input_field`, writes its choice to `decision`, routes via conditional edges over a `Literal[*choices]` structured output — the model can only return a valid branch. Also accepts `tools=[...]` (gathers context first, *then* routes). |
| **`ConditionNode`** | Deterministic branching with **no LLM**. A Python predicate `condition(state) -> str` returns one of `choices`; routing is free, deterministic, and prompt-less. |
| **`HumanNode`** | Resumable human-in-the-loop via LangGraph's `interrupt()`. Surfaces a message for review; the human's reply lands `into` `messages` (or any key a router can gate on). |

## `AgentNode`

The workhorse. Prepends `node_prompt` as a `SystemMessage` to the history (read
from `input_field`, default `messages`), calls the LLM, and returns a delta.

```python
AgentNode(
    name=None,
    llm=None,
    node_prompt="you are a helpful assistant",
    tools=None,                 # bare callables or BaseTool/StructuredTool
    max_tool_iterations=25,     # cap on the internal tool-call loop
    input_field="messages",
    output_field="messages",    # a non-default key writes final content there
    reads=None,                 # typed multi-key inputs
    writes=None,                # list[str] (str-only) or {"key": type} (typed)
    reasoning_effort=None,      # "low" / "medium" / "high" (gpt-5.x)
    cache_ttl=None,
    retry=False,
)
```

- **Tools.** When `tools=[...]` is set, the node runs an internal tool-call loop:
  it executes every requested tool, appends `ToolMessage`s, and re-invokes the
  model until it stops requesting tools (or hits `max_tool_iterations`, which
  then raises). Tools cannot be combined with multi-field structured `writes`.
- **Typed IO.** `reads=[...]` / `writes=[...]` give one node multi-key IO
  dispatched **by value type** — a message-list read is treated as history, a
  scalar read is interpolated into the prompt. `writes={"score": int}` returns
  native-typed structured output (a real `int`, not a string).
- **`output_field`.** When it is the default `messages`, produced messages are
  appended to the conversation; any other key writes the final response there
  instead (a transform node).

## `DecisionNode`

LLM branching only. Reads from `input_field`, writes its choice to the dedicated
`decision` state field (never into `messages`), and routes via LangGraph
conditional edges. The LLM is wrapped with `with_structured_output` over a
`Literal[*choices]`, so the model **can only return a valid branch**.

```python
DecisionNode(
    name=None,
    llm=None,
    node_prompt="",     # required, non-empty
    choices=[],         # required, non-empty
    tools=None,         # optional: gather context BEFORE routing
    input_field="messages",
    reads=None,
    cache_ttl=None,
    retry=False,
)
```

Wire a `DecisionNode` by **indexing a choice** — `decision["positive"] > handler`
sets that choice's child. `decision > x` is an error. If `tools=[...]` is passed,
the node runs a tool-gathering loop first and *then* routes via structured output
(the two never share a call). `reasoning_effort` is not accepted — it conflicts
with structured output on current OpenAI models.

## `ConditionNode`

Deterministic branching with **no LLM**: a Python predicate decides the branch,
so routing is free, deterministic, and prompt-less. Handy for capping loops.

```python
ConditionNode(
    name=None,
    condition=None,     # condition(state) -> str, returns one of `choices`
    choices=[],
    reads=None,         # declare keys so the validator can check availability
    cache_ttl=None,
    retry=False,
)
```

Wired like a `DecisionNode` — index the choice: `cond["retry"] > worker`.

## `HumanNode`

Resumable human-in-the-loop via LangGraph's `interrupt()`.

```python
HumanNode(
    name=None,
    node_prompt="Please review the following message and provide feedback.",
    n=1,                # surface state["messages"][-n]; n<=0 shows nothing
    show=None,          # None => the n-th message; a str => literal; a callable => show(state)
    into="messages",    # where the human reply lands ("messages" wraps a HumanMessage)
    cache_ttl=None,
    retry=False,
)
```

Resume works when the graph is built with a `checkpointer` and invoked with a
`thread_id`: build `AgenticGraph(..., checkpointer=InMemorySaver())`, then resume
via `Command(resume=value)`. Routing `into` a non-`messages` key lets a router
gate on the human's answer.

## Shared knobs and composition

All node types also accept `cache_ttl` (LangGraph `CachePolicy` — caches the
node's result) and `retry` (LangGraph `RetryPolicy` — retries on exception);
`AgenticGraph` auto-provides an `InMemoryCache` when any node sets `cache_ttl`.

An `AgenticGraph` can itself be embedded as a node in a larger graph
(`graph_0 > graph_1`), and RAG helpers (`make_retriever_tool`, optional
`ChromaRAG`) wrap any LangChain retriever as a tool you hand to
`AgentNode(tools=[...])`.

See these nodes in action, one at a time, in the [Basics gallery](examples/basics.md).
