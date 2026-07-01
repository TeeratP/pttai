# pttai

**The easiest way to build multi-agent systems in Python.** Every node is a
self-contained tool-using agent, composed into a *visible* DAG you can fan out,
map-reduce, and validate at compile time — and it all compiles down to a native
LangGraph `StateGraph`, so you keep the whole ecosystem (streaming, async,
checkpointers, LangSmith) with zero lock-in.

If you know LangGraph, think of pttai as **Keras for LangGraph**: an ergonomic
default layer over the same runtime. The value is everything you *don't* write —
`add_node` / `add_edge` / `add_conditional_edges`, `Send` fan-out plumbing,
structured-output routing, the tool-call loop — plus a build-time validator that
catches read-before-written dataflow bugs before you ever invoke.

## The 3-vs-10 story

The same tool-using agent — an LLM that calls `add` / `multiply` in a loop until
it has the answer — is **3 lines in pttai vs. 10 in raw LangGraph**:

```python
from pttai import AgentNode, AgenticGraph

agent = AgentNode(name="agent", llm=llm, tools=[add, multiply])
graph = AgenticGraph(start_node=agent, end_nodes={agent})   # schema-free

graph.invoke(message="What is 21 + 21, then times 3?")      # -> 126
```

The full side-by-side comparison lives in [Getting started](getting-started.md).

## Where to go next

- **[Getting started](getting-started.md)** — install, and the full pttai-vs-LangGraph comparison.
- **[Node types](node-types.md)** — `AgentNode`, `DecisionNode`, `ConditionNode`, `HumanNode`, and the shared `LLMNode` base.
- **Examples → [Basics](examples/basics.md)** — one primitive per file, pttai and raw-LangGraph side by side.
- **Examples → [Architectures](examples/architectures.md)** — famous agent patterns (router, reflection, orchestrator-workers, …) built end-to-end.
- **[API notes](api-notes.md)** — state channels, compile-time validation, `summary()`, token usage, prompt caching.

## What you get

- **`>` wiring** — `a > b > c` builds the graph; branches index by choice. No
  `add_node` / `add_edge` boilerplate.
- **Parallel `fanout(...)` + deferred join** — `start > fanout(a, b) > combine`
  runs `a` and `b` concurrently and joins **once** after both finish (the
  bracket form `start > [a, b] > combine` wires identically).
- **`worker.map("field")` map-reduce** — `dispatch > summarize.map("docs") > reduce`
  fans a worker out over a state list via LangGraph `Send`, once per item, in
  parallel, then joins once.
- **Schema-free typed state** — nodes default to `messages`; `reads=[...]` /
  `writes=[...]` give one node multi-key IO dispatched **by value type**.
  `writes={"score": int}` returns native-typed structured output.
- **`message=` invoke shorthand** — `graph.invoke(message="...")` wraps a string
  (or list of messages) onto `messages`; the full `invoke({...})` state form
  still works.
- **Per-model token usage** — `out["token"]` is a
  `{model: {total/input/output_tokens}}` breakdown accumulated across every node.
- **Opt-in OpenAI prompt caching** — `AgenticGraph(..., prompt_cache=True)`.
- **Compile-time validation + `summary()`** — the constructor runs a forward
  dataflow analysis and **fails the build** if a node reads a key nothing
  produces upstream.

## How it works

`a > b` doesn't build an edge — it just sets `a.children = [b]` and returns `b`,
so `a > b > c` builds a linked structure in memory. `AgenticGraph(...)` walks
that structure **once** at construction, emits the real LangGraph
`add_node` / `add_edge` / `Send` calls, runs the dataflow validator, and
`compile()`s to a native `StateGraph` — which `AgenticGraph` subclasses. So
pttai is a build-time convenience that disappears at runtime: the execution
underneath is plain LangGraph (streaming, async, durability, checkpointers,
LangSmith all included), and you can drop down to it anytime. No lock-in.

!!! note "About Material for MkDocs"
    Material for MkDocs entered maintenance mode in early 2026; it still builds
    fine and is the right choice for this site today. The successor, Zensical,
    reads the existing `mkdocs.yml` and is the later migration target.

## License

MIT — see [LICENSE](https://github.com/TeeratP/pttai/blob/main/LICENSE).
