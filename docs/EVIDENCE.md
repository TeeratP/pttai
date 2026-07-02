# External grounding: wild bugs + framework feature matrix

This document grounds pttai's benchmark taxonomy in reality. Part 1 shows that the
bug classes pttai catches at **build time** are real failures people hit in the
wild with raw LangGraph (not author-imagined). Part 2 positions pttai against
peer frameworks on *what kind of static checking they do*, qualitatively.

The bug taxonomy and the measured build-vs-runtime numbers live in
[`docs/validator.md`](validator.md), [`docs/COMPARISON.md`](COMPARISON.md), and
[`eval/bugbench/results.csv`](../eval/bugbench/results.csv). Our own bugbench
corpus already reproduces the verbatim LangGraph runtime error
`InvalidUpdateError: At key 'result': Can receive only one value per step. Use an
Annotated key to handle multiple values.` — that string anchored the search below.

## Part 1 — Wild-bug mining

> **This is an illustrative sample, not a frequency claim.** These are hand-picked
> public reports that each map to a bug class pttai fails the build on — they show
> the classes are *real and hit in the wild*, but there is **no denominator**
> (they are 6 issues drawn from thousands, not a measured incidence rate). Read
> this as existence proof, not prevalence.

Every row is a real, publicly reported issue/discussion (verified to resolve as
of 2026-07). Each maps to one of **three** dataflow/structural bug classes pttai
fails the build on, that raw LangGraph surfaces only at runtime (after burning
model calls).

| pttai bug class | In-the-wild report | One-line description |
|---|---|---|
| **read-before-write** (runtime `KeyError` on a state key) | [langgraph #5179](https://github.com/langchain-ai/langgraph/issues/5179) | `KeyError: 'context'` — a node/consumer reads a state key the summarizer never populated. |
| **read-before-write** | [langgraph #1798](https://github.com/langchain-ai/langgraph/issues/1798) | `KeyError: 'documents'` reading `state["documents"]` before any node produced it; "fix" is defensive `state.get("documents", [])`. |
| **concurrent-write / no reducer** (`InvalidUpdateError`) | [langgraph #2336](https://github.com/langchain-ai/langgraph/issues/2336) | `InvalidUpdateError: … Can receive only one value per step` — parallel branches write the same plain key in one step. |
| **concurrent-write / no reducer** | [deepagents #96](https://github.com/langchain-ai/deepagents/issues/96) | `INVALID_CONCURRENT_GRAPH_UPDATE` — the `todos` state key has no reducer, so concurrent node updates are rejected. |
| **dangling conditional edge** (`dangling-choice`) | [langgraph #987](https://github.com/langchain-ai/langgraph/issues/987) | `add_conditional_edges(...)` with no `path_map`: routing fans out to *all* nodes instead of the router's real destinations — a mis-wired graph structure. |
| **dangling conditional edge** | [langgraph #450](https://github.com/langchain-ai/langgraph/issues/450) | `KeyError: 'action'` — the routing function returns a destination that isn't in the graph's edge map. |

**Genuine instances shown: 6**, across three classes (two independent reports each):

| Bug class | Wild reports shown |
|---|:--:|
| read-before-write (`KeyError` on state key) | 2 |
| concurrent-write / no reducer (`InvalidUpdateError`) | 2 |
| dangling conditional edge | 2 |

Each class here has at least two real, independent reports — none is
author-imagined, and each maps to a bug pttai fails the **build** on while raw
LangGraph only surfaces at runtime. The other pttai classes were **out of scope
for this search** and are not padded: *cyclic-read-before-write* is the loop-carried
case of the first class; *duplicate node names* is caught at build by **both**
frameworks (see [`docs/COMPARISON.md`](COMPARISON.md) §d — LangGraph raises
`ValueError('Node \`x\` already present.')`), so it isn't a differentiator; and
*prompt-placeholder / scalar-read mismatch* is a pttai-internal reformulation of a
generic Python `str.format` `KeyError` with no dedicated LangGraph-specific report.

### What pttai does NOT catch (honest limitations)

- **Non-termination / runaway loops** (e.g. [langgraph #6731](https://github.com/langchain-ai/langgraph/issues/6731) — an agent that never hits its stop condition and loops to the recursion limit) is **not** a class pttai catches. Cycles validate clean, and the dead-end check is a structural terminal check, **not** a termination-guarantee analysis. A graph that loops forever passes pttai's build.
- **`dead-end-node`** (a non-end node with no outgoing edge) is a pttai **DSL-strictness** rejection, not a LangGraph bug: in LangGraph a node with no outgoing edge is a *legal implicit terminal* that compiles and runs (verified empirically — see footnote 1). It is therefore excluded from the differentiator numbers in [`eval/bugbench/`](../eval/bugbench/).

## Part 2 — Framework feature matrix

Qualitative only. **No LOC ports** to DSPy/Haystack/etc. — different execution
models make line counts apples-to-oranges and invite methodological attack. This
table is about *what class of static checking each framework does before it runs*.

Columns:
- **Declarative topology** — you *declare* the graph/flow structure (nodes + edges), rather than expressing control flow as imperative code.
- **Build-time STRUCTURAL check** — before running, validates graph well-formedness (reachability, dead ends, entry points, connection validity).
- **Build-time DATAFLOW check** — before running, validates *state-key dataflow*: that every key a node reads is produced upstream, and that concurrent writes to a plain key have a reducer. (This is read-before-write / concurrent-write detection — **not** event-type or socket-type checking.)
- **False-positive guarantee** — the dataflow analysis is *sound*: it never rejects a graph that would actually run (pttai uses a `may`-set fixpoint; measured 0 false positives on 19 valid pipelines).
- **Compiles-to-native-runtime** — lowers to an established execution engine rather than shipping its own.

| Framework | Declarative topology | Build-time STRUCTURAL check | Build-time DATAFLOW check | False-positive guarantee | Compiles-to-native-runtime |
|---|:--:|:--:|:--:|:--:|:--:|
| **LangGraph** | yes | partial¹ | **no** | n/a | is the runtime² |
| **DSPy** | no³ | no | no | n/a | no⁴ |
| **LlamaIndex Workflows** | partial⁵ | yes⁶ | no | n/a | own runtime |
| **Haystack** | yes | yes⁷ | partial⁸ | n/a | own runtime |
| **CrewAI** | no⁹ | no | no | n/a | own runtime |
| **AutoGen (GraphFlow)** | yes | yes¹⁰ | no | n/a | own runtime |
| **pydantic-graph (PydanticAI)** | yes | **yes (type-check time)**¹² | **no**¹² | n/a | own runtime |
| **pttai** | **yes** | **yes** | **yes** ⭐ | **yes** | **yes** ¹¹ |

**The narrow, defensible claim: pttai is the only framework we surveyed that
performs a build-time *state-key availability (dataflow)* check** — verifying that
every key a node reads is produced upstream and that concurrent writes to a plain
key have a reducer. We do **not** claim to be the first build-time check of *any*
kind: **pydantic-graph** already does a genuine build-time graph-**structure**
check (its node edges are encoded in `run`-method return-type annotations, so a
standard static type checker flags mis-wired edges before the graph runs¹²), and
LlamaIndex/Haystack/AutoGen validate event/socket/graph structure at build. What
none of them do is a per-node *dataflow* availability analysis over shared state —
that is pttai's distinct contribution, and it is the only one with a
false-positive guarantee (measured 0 FPs on 19 valid pipelines) because it is the
only one running a state-key dataflow analysis to be sound *about*.

### Notes & citations

1. **LangGraph — structural: partial.** `compile()` runs "basic checks on the structure" but does **not** raise on a node with no outgoing edge / no path to `END`: such a node is a *legal implicit terminal* that compiles and runs the graph to a normal halt (verified empirically on LangGraph 1.2.x — `compile()` does not raise and `invoke()` returns without error; see [`eval/bugbench/`](../eval/bugbench/) `de-*` items). It also does **not** flag orphan/unreachable nodes; adding richer structural validation is an open feature request. → [Graph API docs](https://docs.langchain.com/oss/python/langgraph/graph-api), [feature request #6735](https://github.com/langchain-ai/langgraph/issues/6735). Dataflow: `compile()` explicitly does **not** validate which state keys are produced before they're read — that's the entire premise of [`docs/COMPARISON.md`](COMPARISON.md).
2. **LangGraph — is the runtime.** pttai compiles *to* LangGraph's `StateGraph`/Pregel engine; LangGraph doesn't lower to anything further. → [Graph API docs](https://docs.langchain.com/oss/python/langgraph/graph-api). (Be fair: pttai adds a validation pass *on top of* LangGraph, then hands the compiled graph to it — streaming, async, checkpointers all still work.)
3. **DSPy — declarative signatures, imperative topology.** Signatures/modules are declarative, but control flow is "arbitrary Python — you compose modules using standard Python statements." There is no declared graph to statically check. → [dspy.ai](https://dspy.ai/), [DSPy paper](https://arxiv.org/abs/2310.03714).
4. **DSPy — compiles to prompts, not a runtime.** "Compile" optimizes prompts/demos against a metric until quality converges; it does not lower to a graph execution engine. → [dspy.ai](https://dspy.ai/).
5. **LlamaIndex — topology partial.** Structure is *implicit* in the event types each `@step` consumes/emits, not an explicitly declared edge list. → [Workflow API reference](https://developers.llamaindex.ai/python/workflows-api-reference/workflow/).
6. **LlamaIndex — structural: yes.** Validates the event graph before running: steps reachable from an input event, only output events terminal, and an emitted event with no consumer is an error. This is **event-type** validation, not state-key dataflow. → [Workflow API reference](https://developers.llamaindex.ai/python/workflows-api-reference/workflow/).
7. **Haystack — structural: yes.** When the pipeline is built, "the Pipeline validates the components without running them yet." → [Creating Pipelines](https://docs.haystack.deepset.ai/docs/creating-pipelines).
8. **Haystack — dataflow: partial (type-level only).** `connect()` validates that a producer's output socket **type** is compatible with the consumer's input socket type before running — a type check on explicit connections, not a state-key availability analysis (its explicit-DAG model makes read-before-write structurally different from LangGraph's shared-state model). → [Creating Pipelines](https://docs.haystack.deepset.ai/docs/creating-pipelines).
9. **CrewAI — no declared graph.** Orchestration is a `Process` (sequential or hierarchical/manager-delegated), not a declared node/edge graph; docs describe no build-time graph or dataflow validation. → [Processes](https://docs.crewai.com/en/concepts/processes).
10. **AutoGen (GraphFlow) — structural: yes.** `DiGraphBuilder(...).build()` validates the directed graph's structure (entry points, cycles) before use; it does not validate shared-state dataflow. → [GraphFlow docs](https://microsoft.github.io/autogen/stable//user-guide/agentchat-user-guide/graph-flow.html).
11. **pttai — all yes.** Declarative `>` wiring; structural checks (dead-end, dangling choice, duplicate names, unreachable); a `may`/`must` dataflow fixpoint (read-before-write incl. the cyclic loop-carried case, concurrent-write-no-reducer, prompt-placeholder); sound `may`-set errors with **0 false positives measured** on 19 valid pipelines; and it compiles down to a plain LangGraph `StateGraph`. → [`docs/validator.md`](validator.md), [`eval/bugbench/results.csv`](../eval/bugbench/results.csv).
12. **pydantic-graph (PydanticAI) — structural: yes (type-check time); dataflow: no.** In `pydantic-graph`, "nodes and edges are defined using type hints", and "the return type of the node's `run` method … is used to determine the outgoing edges of the node." Because edges *are* return-type annotations, a standard static type checker (mypy/pyright) flags a node that returns an undeclared successor — a build-time graph-**structure** check. But state is a single shared object (`GraphRunContext[StateT]`, "an optional way to access and mutate an object … as nodes run") passed uniformly to every node; there is **no** per-node read/write availability or state-key dataflow analysis. So it checks *wiring*, not *dataflow*. → [pydantic-graph docs](https://pydantic.dev/docs/ai/graph/graph/).
