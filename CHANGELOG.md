# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0]

Initial release. **nae** (Nodes and Edges) is a thin, declarative DSL layer over
[LangGraph](https://langchain-ai.github.io/langgraph/): wire self-contained,
tool-using agent nodes together with a `>` operator into a visible DAG, then
hand the start/end nodes to `AgenticGraph`, which compiles down to a native
LangGraph `StateGraph`.

### Added

- **`>` wiring DSL** — `a > b > c` builds the graph structure in memory;
  `AgenticGraph(start_node=..., end_nodes=...)` materializes it into a LangGraph
  `StateGraph` and compiles it. No `add_node` / `add_edge` boilerplate.
- **Compile-time dataflow validator** — the constructor runs a forward
  `may`/`must` availability analysis and **fails the build**
  (`GraphValidationError`) on read-before-write, undeclared-key reads/writes,
  concurrent writes to a reducer-less key, dangling decision choices, dead-end
  nodes, duplicate node names, and prompt-placeholder mismatches. `summary()`
  prints a topology table.
- **Node types** — `AgentNode` (tool-call loop, typed multi-key `reads`/`writes`,
  `output_field`, optional `reasoning_effort`), `DecisionNode` (constrained
  structured-output routing over a `Literal` of choices), `ConditionNode`
  (deterministic branching via a Python predicate, no LLM), and `HumanNode`
  (resumable human-in-the-loop via `interrupt()`).
- **Parallelism** — `fanout(a, b, c)` / `[a, b, c]` for fixed parallel branches
  with a deferred single join, and `worker.map("field")` for dynamic map-reduce
  fan-out over a state list via LangGraph `Send`.
- **Schema-free typed state** — `AgenticState` with reduced `messages` / `log` /
  `token` channels; undeclared keys a node reads/writes are auto-registered as
  plain channels, so most graphs need no hand-written schema.
- **Subgraph composition** — an `AgenticGraph` composes as a node inside a
  larger `AgenticGraph` (`graph_0 > graph_1`).
- **Invoke ergonomics** — `invoke(message="...")` string/list shorthand
  alongside the full `invoke({...})` state form; `stream` / `ainvoke` /
  `astream` variants.
- **Per-model token usage** — `out["token"]` accumulates a
  `{model: {total/input/output_tokens}}` breakdown across every node.
- **Graph rendering** — `AgenticGraph` renders its compiled topology (Mermaid /
  PNG) for quick visual inspection of the wired DAG.
- **Per-node `cache_ttl` / `retry`** — LangGraph `CachePolicy` / `RetryPolicy`
  wiring, with an auto-provided `InMemoryCache` when any node sets `cache_ttl`.
- **RAG helpers** — `make_retriever_tool` wraps any LangChain retriever as a
  tool; `ChromaRAG` is a convenience wrapper (optional `rag` extra).
- **Auto-generated API reference** — `AgenticGraph`, all node types, state
  helpers, and internals render from docstrings via mkdocstrings on a MkDocs
  Material site (`pip install "nae[docs]"`), alongside getting-started,
  node-types, validator, quickstart, LangGraph-migration, and example galleries.

[0.1.0]: https://github.com/TeeratP/nae/releases/tag/v0.1.0
