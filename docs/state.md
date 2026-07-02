# State & observability

Every pttai graph shares one state object across its nodes. By default that is
[`AgenticState`](api/state/agenticstate.md), a `TypedDict` of **reduced
channels**: each key carries a reducer that merges node outputs, so parallel
branches combine instead of overwriting each other.

Nodes return deltas — only the keys they update — and never mutate state in
place. The reducers do the merging, which is what keeps checkpointing,
parallel-branch joins, and subgraph composition correct rather than racy.

## The channels

- **`messages`** — the conversation, reduced with LangGraph's `add_messages`:
  it appends new messages, replaces by matching `id`, coerces bare strings to
  `HumanMessage`, and merges parallel branches.
- **`log`** — a trace, reduced with `operator.add`: every node appends a line
  (`"{name}:{content}"`, tool calls, routing decisions). Seed it with `[]` on
  invoke to capture the trace.
- **`token`** — per-model usage totals, accumulated across every node.
- **`decision_{name}`** — each router (`DecisionNode`/`ConditionNode`) writes
  its choice to its own per-node channel, auto-registered as a plain
  last-writer-wins key and read by that node's `route()`. There is no shared
  `decision` channel, so multiple routers never clobber each other.

You rarely declare a schema: nodes auto-register the keys they `read`/`write`.
When you do pass a custom `state=` schema, it is the same pattern — more
reduced channels (e.g. a `summary: str` field for a node with
`writes=["summary"]`).

## Free observability: `token` and `log`

Total token spend and a per-node trace come with every run — no callbacks, no
custom reducers to wire:

```python
out = graph.invoke({"messages": ["..."], "log": []})   # seed log=[] to capture it
print(out["token"])   # {'gpt-5.4-nano': {'input_tokens': 42, 'output_tokens': 88, 'total_tokens': 130, ...}}
print(out["log"])     # ['frame:...', 'optimist:...', ...] — one line per node/tool call
```

`token` is the run total: it accumulates across every node, tool-loop call, and
parallel branch, deep-summed per model. In raw LangGraph you would hand-wire a
usage callback plus a custom summing channel to get the same numbers.

Offline, with no API key, the fake model reports no usage, so `token` is an
empty `{}` — the accounting runs, there is nothing to count. Side-by-side with
the raw-LangGraph version:
[`examples/basics/13_token_and_log.py`](https://github.com/TeeratP/pttai/blob/main/examples/basics/13_token_and_log.py).

The validator reads the same declared `reads`/`writes` that drive these
channels — see [The validator](validator.md) for what it checks.

## Limitations

Kept honest on purpose:

- **Structured multi-write list fields are `str`-only.** `writes=["a", "b"]`
  produces one `str` field per key; use the dict form `writes={"a": int}` for
  native-typed structured output.
- **Map workers don't echo their source item** — return only what you computed;
  the item you mapped over doesn't need to appear in the worker's output.
- **`[b, c] > [d, e]` isn't supported.** Two fan-outs chained directly is
  Python's element-wise list compare, not join wiring — insert a node between.
- **Async is graph-level only.** `ainvoke`/`astream` run the sync nodes in
  LangGraph's threadpool; true per-node async LLM calls aren't implemented.
- **`reasoning_effort` is `AgentNode`-only** — it conflicts with
  `DecisionNode`'s structured output on current OpenAI models.
