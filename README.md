# pttai

**`pttai` — Keras for LangGraph.** You write agent graphs in the layer you *want*
to write in — a declarative `>` DSL with parallelism, typed state, and a compiler
that fails on dataflow bugs — and it compiles down to a native LangGraph
`StateGraph`, so you keep the whole ecosystem (streaming, async, checkpointers,
LangSmith) with zero lock-in.

![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-1.0-orange)
![tests](https://img.shields.io/badge/tests-69%20passing-green)
![License](https://img.shields.io/badge/license-MIT-green)

Like Keras over TensorFlow, `pttai` is an *ergonomic default layer*, not a faster
runtime: the execution underneath is plain LangGraph. The value is everything you
*don't* write — `add_node`/`add_edge`/`add_conditional_edges`, `Send` fan-out
plumbing, structured-output routing, parallel-join bookkeeping — and a build-time
validator that catches read-before-written bugs before you ever invoke.

```python
from pttai import AgentNode, AgenticGraph, AgenticState, fanout

# Parallel fan-out + join — both forms wire identically:
start > fanout(worker_a, worker_b) > combine     #  (or:  start > [worker_a, worker_b] > combine)

# Map-reduce — run a worker once per item, in parallel, then join:
dispatch > summarize.map("docs") > reduce

graph = AgenticGraph(state=AgenticState, start_node=start, end_nodes={combine})
graph.summary()   # Keras-style table of every node's reads/writes/available keys
#  ...and the constructor FAILS the build if any node reads a key nothing produces.
```

That's the whole pitch: parallelism in one line, and the compiler has your back.

## Install

Requires **Python ≥ 3.10**. Core deps: LangGraph ≥ 1.0, langchain-core ≥ 1.0, Pydantic 2.

```bash
pip install pttai                  # core
pip install "pttai[openai]"        # + langchain-openai & python-dotenv (examples/notebook)
pip install "pttai[rag]"           # + langchain-chroma (ChromaRAG)
pip install "pttai[dev]"           # + pytest
```

Local dev: `python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev,openai]"`.
For live model calls, copy `.env.example` to `.env` and set `OPENAI_API_KEY`.

## Quickstart

A tool call → decision → branch graph, wired entirely with `>`:

```python
import random
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from pttai import AgenticGraph, AgentNode, DecisionNode, AgenticState


def random_number(maximum: int) -> int:
    """Return a random integer between 1 and `maximum` (inclusive)."""
    return random.randint(1, maximum)


llm = ChatOpenAI(model="gpt-5.4-nano")

# An agent that can call a tool. bind_tools wraps the bare function as a
# StructuredTool and runs the call/respond loop automatically.
randomizer = AgentNode(
    name="randomizer", llm=llm,
    node_prompt="Use the random_number tool to pick a number, then state it.",
)
randomizer.bind_tools([random_number])

# A decision node returns one of `choices` (constrained structured output).
classifier = DecisionNode(
    name="classifier", llm=llm,
    node_prompt="If the number is greater than 5 answer 'positive', else 'negative'.",
    choices=["positive", "negative"],
)

positive_handler = AgentNode(name="positive_handler", llm=llm,
                             node_prompt="Report the number in an upbeat tone.")
negative_handler = AgentNode(name="negative_handler", llm=llm,
                             node_prompt="Report the number in a gloomy tone.")

# Wire it: `>` builds the graph, branches index by choice.
randomizer > classifier
classifier["positive"] > positive_handler
classifier["negative"] > negative_handler

graph = AgenticGraph(state=AgenticState, start_node=randomizer,
                     end_nodes={positive_handler, negative_handler})

result = graph.invoke({
    "messages": [HumanMessage(content="Give me a random number up to 10.")],
    "log": [],
})
print("routed to :", result["decision"])
print("final reply:", result["messages"][-1].content)
```

The offline, no-API-key tour of the parallel features lives in
[`examples/parallel_usage.py`](examples/parallel_usage.py); the live tool→decision→branch
demo is [`examples/sample_usage.py`](examples/sample_usage.py).

## Parallel topology

Three ways to run nodes concurrently, all wired with `>`:

```python
# Fan-out / join — two equivalent forms; `combine` is a deferred join that runs
# ONCE after both branches finish (LangGraph reducers merge their deltas):
start > fanout(worker_a, worker_b) > combine
start > [worker_a, worker_b] > combine            # identical wiring

# Multi-node branches — each arm can be its own chain; fan-out goes to the chain
# HEADS, and the join collects the chain TAILS:
a > fanout(b > c > d > e, f > g) > h

# Map-reduce — fan a worker out over a state list via LangGraph `Send`, once per
# item, in parallel; the collector runs once after:
dispatch > summarize.map("docs") > reduce         # "docs" must be a state channel
```

`fanout(...)` returns a single object, so it composes cleanly in a chained `>`;
the bracket-list form relies on Python's chained-comparison semantics and wires
identically. (`[b, c] > [d, e]` — two fan-outs back-to-back — isn't supported;
insert a node between them.)

## Typed multi-key state IO

By default a node reads/writes `messages`. `reads=[...]` / `writes=[...]` let one
node touch several state keys, dispatched **by value type**, not key name:

```python
classify = AgentNode(
    name="classify", llm=llm,
    node_prompt="Rate the {topic} discussion. Return sentiment and score.",
    reads=["messages", "topic"],      # messages-list -> history; scalar -> interpolated
    writes=["sentiment", "score"],    # two scalars -> structured output
)
```

- **Reads** — a read whose runtime value is a non-empty list of messages becomes
  conversation history (all such reads concatenated in order); any other value is
  a scalar `.format_map`-interpolated into `node_prompt`. So a custom key holding
  a message list is treated as history automatically.
- **Writes — three modes:**
  | `writes=` | behavior |
  |-----------|----------|
  | `["messages"]` (default) | append the produced messages to the conversation |
  | one scalar key | write the final response *content* to that key (transform node) |
  | two+ scalar keys | structured output — one `str` field per key, no tool loop |
- **Tools XOR structured.** A node either runs the free-form tool-call loop
  (`bind_tools`) or emits multi-field structured output — not both (they conflict
  on current OpenAI models). Combining them raises.

## Static validation — the differentiator

When you build a graph, `pttai` runs a forward dataflow analysis and **fails the
build** if any node reads a state key that nothing produces upstream, writes a key
the schema doesn't declare, or has two parallel branches racing on a non-reduced
key. The bug surfaces at construction, with a message that names the key and the
real writer — not as a `KeyError` three nodes into a run.

```python
early > late          # `early` reads "summary"; its only writer `late` runs AFTER it
AgenticGraph(state=SummaryState, start_node=early, end_nodes={late})
# raises GraphValidationError:
#   [error] early: reads computed key 'summary' but no upstream node produces it
#   before this node (produced by: ['late'], none of which are upstream);
#   available keys here: ['decision', 'log', 'messages']
```

- `AgenticGraph(..., validate=True)` is the default; pass `validate=False` to opt out.
- `graph.validate()` returns a `ValidationReport` (`.ok`, `.errors`, `.warnings`)
  without raising.
- `graph.summary()` prints a Keras-`model.summary()`-style table — every node's
  reads, writes, and the keys guaranteed available where it runs:

```
AgenticGraph 'graph'   state=AgenticState
initial: decision, log, messages
------------------------------------------------------------------
node      type       reads     writes        available
start     AgentNode  messages  log,messages  decision,log,messages
worker_a  AgentNode  messages  log,messages  decision,log,messages
combine   AgentNode  messages  log,messages  decision,log,messages
worker_b  AgentNode  messages  log,messages  decision,log,messages
------------------------------------------------------------------
4 nodes · 0 errors · 0 warning(s)
```

- `inputs={"cfg"}` declares a *plain* (non-reduced) key you seed at `invoke()` that
  a node also writes — so the validator treats it as provided at entry, not as a
  read-before-written bug. (Reduced channels and keys no node writes are inferred
  as inputs automatically.)
- Hard errors come only from the precise (`may`) analysis — zero false positives;
  the all-paths (`must`) analysis drives warnings only.

## Node types

All nodes are callables (`__call__(state) -> delta`) invoked by LangGraph with the
shared state. They return **only the keys they update**; reducers merge them.

| Node | Purpose |
|------|---------|
| **`AgentNode`** | Prepends `node_prompt` as a `SystemMessage`, calls the LLM, returns a delta. `bind_tools(...)` wraps bare callables as `StructuredTool`s and runs an internal tool-call loop (capped by `max_tool_iterations`, default 25). `reads=[...]`/`writes=[...]` give typed multi-key IO (history vs. scalar reads; messages / single-scalar / structured writes). Optional `reasoning_effort` (`"low"`/`"medium"`/`"high"`) passed per-call for gpt-5.x. |
| **`DecisionNode`** | Branching only. Reads from `input_field`, writes its choice to `decision`, routes via conditional edges over a `Literal[*choices]` structured output — the model can only return a valid branch. Wired by indexing a choice (`decision["x"] > handler`); `decision > x` is an error. |
| **`InputNode`** | Human-in-the-loop via LangGraph's `interrupt()`. Resumes when the graph is built with a `checkpointer` and invoked with a `thread_id`, via `Command(resume=value)`. |

All node types also accept `cache_ttl` (LangGraph `CachePolicy`) and `retry`
(`RetryPolicy`); `AgenticGraph` auto-provides an `InMemoryCache` when any node sets
`cache_ttl`. An `AgenticGraph` can itself be embedded as a node in a larger graph
(`graph_0 > graph_1`), and RAG helpers (`make_retriever_tool`, optional `ChromaRAG`)
wrap any LangChain retriever as a bindable tool.

## State

`AgenticState` is a `TypedDict` of reduced channels; custom schemas just add more:

- **`messages`** — `add_messages` reducer: appends, replaces by matching `id`,
  coerces bare strings to `HumanMessage`, and merges parallel branches.
- **`log`** — `operator.add`: every node appends a trace line (`"{name}:{content}"`,
  tool calls, decisions). Seed it with `[]` on invoke to capture the trace.
- **`decision`** — transient routing key written by `DecisionNode`, read by its
  `route()`.

Nodes return deltas and never mutate state in place — that's what keeps
checkpointing, parallel-branch merges, and subgraph composition correct rather
than racy.

## Design decisions

The framework is small on purpose; the value is in a few load-bearing choices.

- **Deferred wiring, then compile-once.** `a > b` only sets `a.children = [b]` and
  returns `b`, so `a > b > c` builds a linked structure in memory with no edges
  yet. `AgenticGraph(...)` walks it once at construction, emits the real
  `add_node`/`add_edge`/`Send` calls, validates, and `compile()`s. Wiring and
  dataflow bugs surface immediately, not mid-run. Revisited nodes are
  short-circuited by name, so cycles just work; names must be unique.
- **Reducer-based state deltas.** Nodes return only the keys they change; LangGraph
  reducers merge them. This is what makes parallel fan-out/join and map-reduce
  correct — concurrent branches contribute deltas instead of clobbering state.
- **Routing constrained to a `Literal`.** `DecisionNode` wraps the LLM with
  `with_structured_output` over a `Literal[*choices]` field, so the model can only
  return a valid branch; the label lands in `decision` and never pollutes `messages`.
- **Compiles to plain LangGraph — no lock-in.** `AgenticGraph` *is* a `StateGraph`
  subclass. The whole LangGraph ecosystem (streaming, async, durability, LangSmith,
  checkpointers) is underneath, and you can drop down anytime.

## Running it

```bash
python -m pytest tests/                # full suite, no API calls (a scripted FakeLLM stands in)
python examples/parallel_usage.py      # offline tour: parallel + map-reduce + validation
python examples/sample_usage.py        # live end-to-end demo (needs OPENAI_API_KEY)
```

The **69-test** suite covers state reducers, graph construction, routing, the
tool-call loop, interrupt/resume, RAG tool wiring, streaming/async, configurable
fields, parallel fan-out/join, map-reduce, multi-key IO, static validation, and
node caching/retry/`reasoning_effort`/`durability`.

## Limitations

Kept honest on purpose:

- **Structured multi-write fields are `str`-only in v1.** `writes=[a, b]` produces
  one `str` field per key; typed/nested structured output (a planned
  `output_model=` escape hatch) isn't built yet.
- **Map workers don't echo their source item.** A `.map(...)` worker receives each
  item but returns only its reply (the `Send` payload is the worker's input, never
  a state update), and it must output `messages` (the default) — a non-message
  write would race N parallel workers on the same plain key.
- **`must` (all-paths) analysis is imprecise for decision→handler→merge.** A
  handler reaches the merge by a sequential edge, so its writes union into the
  merge as if unconditional — this can only *under*-warn (warning-only); it never
  produces a wrong hard error (those come only from the precise `may` analysis).
- **`[b, c] > [d, e]` isn't supported.** Two fan-outs chained directly is Python's
  element-wise list compare, not our join wiring — insert a node between them.
- **Async is graph-level only.** `ainvoke`/`astream` run the sync nodes in
  LangGraph's threadpool; true per-node async LLM calls aren't implemented (which
  is also why a per-node `timeout` isn't exposed — LangGraph only times out async
  nodes).
- **`reasoning_effort` is `AgentNode`-only** — it conflicts with `DecisionNode`'s
  structured output on current OpenAI models.

## Project layout

```
pttai/
  graph.py            # AgenticGraph — walks the wiring, validates, compiles to StateGraph
  node.py             # base Node, the `>` operator, fanout()/Branch/Spread (map-reduce)
  validation.py       # compile-time dataflow analysis + ValidationReport + summary()
  state.py            # AgenticState (reduced channels)
  nodes/              # AgentNode, DecisionNode, InputNode
  nodes/_fields.py    # type-based read partitioning (history vs. scalar)
  tools/              # RAG tool helpers (make_retriever_tool, ChromaRAG)
examples/parallel_usage.py   # offline tour: parallel + map-reduce + validation
examples/sample_usage.py     # live tool -> decision -> branch demo
tests/                # pytest suite (no live API calls)
docs/project.md       # living notes: setup, status, roadmap, rough edges
```

See [`docs/project.md`](docs/project.md) for setup, status, and roadmap.

## License

MIT — see [LICENSE](LICENSE).
