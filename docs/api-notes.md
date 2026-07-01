# API notes

Reference for the runtime pieces you interact with beyond the node types: the
state channels, compile-time validation, `summary()`, token usage, and prompt
caching.

## State channels

`AgenticState` is a `TypedDict` of **reduced** channels; custom schemas just add
more. Nodes return deltas (only the keys they update) and never mutate state in
place — that is what keeps checkpointing, parallel-branch merges, and subgraph
composition correct rather than racy.

- **`messages`** — `add_messages` reducer: appends new messages, replaces by
  matching `id`, coerces bare strings to `HumanMessage`, and merges
  parallel-branch updates instead of clobbering.
- **`log`** — `operator.add`: every node appends a trace line
  (`"{name}:{content}"`, tool calls, decisions). Seed it with `[]` on invoke to
  capture the trace.
- **`decision`** — transient routing key written by `DecisionNode`, read by its
  `route()`. Plain last-writer-wins, which is correct because `route` runs
  immediately after the same node writes it.
- **`token`** — per-model usage totals accumulated across every node.

## Invoking a graph

`AgenticGraph` runs via `invoke` / `stream` / `ainvoke` / `astream` (all accept
an optional LangGraph `config` and `durability`).

- **`message=` shorthand** — `graph.invoke(message="...")` wraps a string (or a
  list of messages) onto `messages`. The full `invoke({...})` state form still
  works; seed `log=[]` if you want the trace.
- **Checkpointing / resume** — pass `checkpointer=...` to the constructor to
  enable `HumanNode` interrupt/resume; invoke with a `thread_id` config and
  resume via `Command(resume=value)`. Without a checkpointer, the simple no-config
  `invoke({...})` path works.
- **Async** is graph-level only: `ainvoke` / `astream` run the sync nodes in
  LangGraph's threadpool; true per-node async LLM calls aren't implemented.

## Compile-time validation

The `AgenticGraph` constructor runs a **forward dataflow analysis** and **fails
the build** if a node reads a key that nothing produces upstream — naming both
the offending key and the real writer. Bugs surface before you ever invoke,
rather than at runtime. You can also call `graph.validate()` directly.

## `summary()`

`summary()` prints a Keras-`model.summary()`-style topology table — a static
view of the DAG (per-node reads / writes / available keys):

```text
AgenticGraph 'graph'   state=AgenticState
initial: decision, log, messages, token
--------------------------------------------------------------------------
node        type       reads     writes        available
frame       AgentNode  messages  log,messages  decision,log,messages,token
optimist    AgentNode  messages  log,messages  decision,log,messages,token
verdict     AgentNode  messages  log,messages  decision,log,messages,token
skeptic     AgentNode  messages  log,messages  decision,log,messages,token
pragmatist  AgentNode  messages  log,messages  decision,log,messages,token
--------------------------------------------------------------------------
5 nodes · 0 errors · 0 warning(s)
```

## Token usage

`out["token"]` is a `{model: {total_tokens / input_tokens / output_tokens}}`
breakdown accumulated across every node in the run — per-model, so a graph that
mixes models keeps them separate.

## Prompt caching

Opt-in OpenAI prompt caching: `AgenticGraph(..., prompt_cache=True)` threads one
cache key through every OpenAI `AgentNode` call in the graph.

## Node policies and composition

All node types accept `cache_ttl` (LangGraph `CachePolicy`) and `retry`
(`RetryPolicy`); `AgenticGraph` auto-provides an `InMemoryCache` when any node
sets `cache_ttl`. An `AgenticGraph` can itself be embedded as a node in a larger
graph (`graph_0 > graph_1`), and RAG helpers (`make_retriever_tool`, optional
`ChromaRAG`) wrap any LangChain retriever as a tool for `AgentNode(tools=[...])`.

## Limitations

Kept honest on purpose:

- **Structured multi-write list fields are `str`-only.** `writes=["a", "b"]`
  produces one `str` field per key; use the dict form `writes={"a": int}` for
  native-typed structured output.
- **Map workers don't echo their source item** and must output `messages` (the
  default) — a non-message write would race N parallel workers on a plain key.
- **`[b, c] > [d, e]` isn't supported.** Two fan-outs chained directly is
  Python's element-wise list compare, not join wiring — insert a node between.
- **Async is graph-level only** (see above).
- **`reasoning_effort` is `AgentNode`-only** — it conflicts with `DecisionNode`'s
  structured output on current OpenAI models.

## Testing

The **164-test** suite (`python -m pytest tests/`) runs with **no API calls** — a
scripted `FakeLLM` stands in. It covers state reducers, graph construction,
routing, the tool-call loop, interrupt/resume, RAG tool wiring, streaming/async,
configurable fields, parallel fan-out/join, map-reduce, multi-key IO, static
validation, and node caching/retry/`reasoning_effort`/`durability`.
