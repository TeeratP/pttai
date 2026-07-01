# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-07-01

Initial release. A thin, declarative DSL layer over
[LangGraph](https://langchain-ai.github.io/langgraph/): wire nodes with `>`,
compile to a native `StateGraph`.

### Added

- **`>` wiring DSL** — `a > b > c` builds the graph structure in memory;
  `AgenticGraph(start_node=..., end_nodes=...)` materializes it into a LangGraph
  `StateGraph` and compiles it. No `add_node` / `add_edge` boilerplate.
- **Compile-time dataflow validator** — the constructor runs a forward
  `may`/`must` availability analysis and **fails the build**
  (`GraphValidationError`) on read-before-write, undeclared-key reads/writes,
  concurrent writes to a reducer-less key, dangling decision choices, dead-end
  nodes, duplicate node names, and prompt-placeholder mismatches. `summary()`
  prints a Keras-style topology table.
- **Node types** — `AgentNode` (tool-call loop, typed multi-key `reads`/`writes`,
  `output_field`, optional `reasoning_effort`), `DecisionNode` (constrained
  structured-output routing over a `Literal` of choices), `ConditionNode`
  (deterministic branching via a Python predicate, no LLM), and `HumanNode`
  (resumable human-in-the-loop via `interrupt()`).
- **Parallelism** — `fanout(a, b, c)` / `[a, b, c]` for fixed parallel branches
  with a deferred single join, and `worker.map("field")` for dynamic map-reduce
  fan-out over a state list via LangGraph `Send`.
- **Schema-free typed state** — `AgenticState` with reduced `messages` / `log` /
  `decision` / `token` channels; undeclared keys a node reads/writes are
  auto-registered as plain channels, so most graphs need no hand-written schema.
- **Subgraph composition** — an `AgenticGraph` composes as a node inside a
  larger `AgenticGraph` (`graph_0 > graph_1`).
- **Invoke ergonomics** — `invoke(message="...")` string/list shorthand
  alongside the full `invoke({...})` state form; `stream` / `ainvoke` /
  `astream` variants.
- **Per-model token usage** — `out["token"]` accumulates a
  `{model: {total/input/output_tokens}}` breakdown across every node.
- **Opt-in OpenAI prompt caching** — `AgenticGraph(..., prompt_cache=True)`.
- **Per-node `cache_ttl` / `retry`** — LangGraph `CachePolicy` / `RetryPolicy`
  wiring, with an auto-provided `InMemoryCache` when any node sets `cache_ttl`.
- **RAG helpers** — `make_retriever_tool` wraps any LangChain retriever as a
  tool; `ChromaRAG` is a convenience wrapper (optional `rag` extra).
- **Docs** — MkDocs Material site (getting started, node types, validator,
  quickstart, LangGraph migration, examples galleries).

[Unreleased]: https://github.com/TeeratP/pttai/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/TeeratP/pttai/releases/tag/v0.1.0
