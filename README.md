# pttai

**pttai** — *Pythonic Topology Tools for AI*.

**A declarative DSL over LangGraph whose typed node-IO enables a build-time
dataflow lint.** Every node is a self-contained tool-using agent, composed with a
`>` operator into a *visible* DAG you can fan out, map-reduce, and — because each
node declares the state keys it reads and writes — statically check for dataflow
bugs *before* you invoke. It all compiles down to a native LangGraph
`StateGraph`, so you keep the whole ecosystem (streaming, async, checkpointers,
LangSmith) with zero lock-in.

If you know LangGraph, think of pttai as **Keras for LangGraph**: an ergonomic
default layer over the same runtime. The value is everything you *don't* write —
`add_node`/`add_edge`/`add_conditional_edges`, `Send` fan-out plumbing,
structured-output routing, the tool-call loop — plus a build-time validator that
catches read-before-written dataflow bugs before you ever invoke.

![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-1.0-orange)
![tests](https://img.shields.io/badge/tests-166%20passing-green)
![License](https://img.shields.io/badge/license-MIT-green)

<p align="center">
  <img src="https://raw.githubusercontent.com/TeeratP/agentic-framework/main/figures/architecture.png" width="100%"
       alt="pttai architecture: the > DSL builds a linked node structure in memory; AgenticGraph's _build_graph walks it into add_node/add_edge/Send calls; a build-time dataflow validator gates the build (GraphValidationError on failure, before any model call); on success it compiles to a native LangGraph StateGraph run via invoke/stream/async.">
</p>

<p align="center"><em>The <code>&gt;</code> DSL → <code>_build_graph</code> → <strong>build-time validator gate</strong> → native LangGraph <code>StateGraph</code>. (<a href="https://github.com/TeeratP/agentic-framework/blob/main/figures/architecture.svg">SVG</a> · <a href="https://github.com/TeeratP/agentic-framework/blob/main/figures/architecture.mmd">source</a>)</em></p>

## pttai vs. raw LangGraph

The same tool-using agent — an LLM that calls `add` / `multiply` in a loop until
it has the answer. Ask it *"What is 21 + 21, then times 3?"* and both print
**126**. The only thing that differs is how much graph plumbing you write:
**3 lines vs. 10.**

```python
# pttai
from pttai import AgentNode, AgenticGraph

agent = AgentNode(name="agent", llm=llm, tools=[add, multiply])
graph = AgenticGraph(start_node=agent, end_nodes={agent})   # schema-free

graph.invoke(message="What is 21 + 21, then times 3?")      # -> 126
```

```python
# raw LangGraph
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

llm_with_tools = llm.bind_tools([add, multiply])

def call_model(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_node("tools", ToolNode([add, multiply]))
builder.add_edge(START, "call_model")
builder.add_conditional_edges("call_model", tools_condition)  # tools? -> "tools" : END
builder.add_edge("tools", "call_model")                       # loop back to the model
graph = builder.compile()

graph.invoke({"messages": [{"role": "user", "content": "What is 21 + 21, then times 3?"}]})  # -> 126
```

Identical behavior — same tools, same loop, same answer. pttai folds the model
node, the `ToolNode`, the `tools_condition` edge and the loop-back edge into
**one `AgentNode`** with a built-in tool-call loop, and infers the state schema
for you. Both versions run side by side in
[`examples/vs_langgraph.py`](https://github.com/TeeratP/agentic-framework/blob/main/examples/vs_langgraph.py).

## The validator: bugs caught before you ever invoke

The line count is the *secondary* win. The one you can't get from raw LangGraph
or `create_react_agent` is a **build-time dataflow analysis** that *fails the
build* on read-before-write (including the cyclic, loop-carried case), dangling
branches, and concurrent unreduced writes — before a single model call. Raw
LangGraph compiles the same bugs and only trips at runtime, on the first input
that exercises the broken path.

<p align="center">
  <img src="https://raw.githubusercontent.com/TeeratP/agentic-framework/main/figures/validator_before_after.png" width="100%"
       alt="Two panels on the same buggy RAG pipeline. Left (pttai): the AgenticGraph constructor raises GraphValidationError at build time with 0 model calls. Right (raw LangGraph): compile() succeeds with no dataflow check, then a runtime KeyError on the first invoke; across 12 such bugs all 12 fail at runtime and 8 (simulated) LLM calls are burned.">
</p>

Measured on a 36-pipeline benchmark ([`eval/bugbench/`](https://github.com/TeeratP/agentic-framework/blob/main/eval/bugbench/)): pttai
catches **12/12** pttai-only dataflow bugs at build with **0 false positives** on
19 valid pipelines, while raw LangGraph catches **0** of those at build (all 12
surface at runtime) and burns **8 model calls** — a *simulated, worst-case-ordered*
figure from an offline fake LLM, not measured real-model cost. And the DSL is
~**60% less code** — 113 vs 281 LOC across 12 pipelines. (Duplicate node names
are caught by *both* frameworks at build; the dead-end class is legal LangGraph
behavior, not a defect — both are excluded above.)

<p align="center">
  <img src="https://raw.githubusercontent.com/TeeratP/agentic-framework/main/figures/bug_catch.png" width="49%"
       alt="Bar chart: pttai catches 12/12 pttai-only dataflow bugs at build; raw LangGraph catches 0 at build (all 12 surface at runtime) and burns 8 simulated LLM calls; pttai has 0 false positives on 19 valid pipelines.">
  <img src="https://raw.githubusercontent.com/TeeratP/agentic-framework/main/figures/loc_comparison.png" width="49%"
       alt="Grouped bar chart of lines of code per pipeline: pttai 113 LOC vs raw LangGraph 281 LOC across 12 pipelines, about 60% fewer lines.">
</p>

Full methodology and per-class results are in
[`docs/COMPARISON.md`](https://github.com/TeeratP/agentic-framework/blob/main/docs/COMPARISON.md); regenerate the charts from the
committed CSV/JSON with `python figures/make_charts.py`.

## Examples

Two runnable galleries make the "pttai vs. LangGraph" story concrete:

- **[`examples/basics/`](https://github.com/TeeratP/agentic-framework/blob/main/examples/basics/)** — one file per feature, each showing
  the pttai version *and* the equivalent raw-LangGraph version side by side. The
  fastest way to see exactly what plumbing pttai folds away — tool loops,
  fan-out/join, map-reduce, structured-output routing, typed state IO,
  human-in-the-loop — one concept at a time.
- **[`examples/architectures/`](https://github.com/TeeratP/agentic-framework/blob/main/examples/architectures/)** — famous agent
  patterns (router, evaluator-optimizer, orchestrator-workers, reflection, and
  more) built end-to-end in pttai, so you can lift a whole topology instead of a
  single node.

Start with `basics/` to learn the primitives, then reach for `architectures/`
when you're wiring a real system.

## Interactive playground

`python demo/app.py` launches a local Gradio playground: paste a `>`-DSL snippet,
click **Build + Validate**, and see the compiled LangGraph diagram and the
build-time validator output side by side — no API key needed. See [`demo/`](https://github.com/TeeratP/agentic-framework/blob/main/demo/).

<p align="center">
  <img src="https://raw.githubusercontent.com/TeeratP/agentic-framework/main/figures/demo_screenshot.png" width="100%"
       alt="pttai playground: (a) a working RAG QA pipeline (retrieve > rerank > answer) compiles clean with a rendered LangGraph diagram and a green validator; (b) a broken pipeline (rerank wired before retrieve) fails the build — the offending 'rerank' node is painted red and the read-before-write error is shown, before the graph is ever invoked.">
</p>

<p align="center"><em>The playground: (a) a working pipeline compiles clean; (b) a broken one fails the build with the offending node painted <strong>red</strong> and the error attached.</em></p>

## Install

Not on PyPI yet — install from source:

```bash
git clone https://github.com/TeeratP/agentic-framework && cd agentic-framework
python -m venv .venv && source .venv/bin/activate
pip install -e ".[openai]"          # core + langchain-openai & python-dotenv
```

Requires **Python ≥ 3.10** (core deps: LangGraph ≥ 1.0, langchain-core ≥ 1.0,
Pydantic 2). Other extras: `[rag]` (langchain-chroma for `ChromaRAG`), `[dev]`
(pytest). For live model calls, set `OPENAI_API_KEY` in your environment or a
`.env` file.

## 30-second example: a multi-agent panel

A question goes to `frame` (which sharpens it into one concrete decision), fans
out to three rival personas — optimist / skeptic / pragmatist — who argue
**concurrently**, then `verdict` weighs every argument into a one-paragraph
ruling. The whole thing is the one wiring line at the bottom.

```python
from pttai import AgentNode, AgenticGraph, fanout
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-5.4-nano")

frame = AgentNode(name="frame", llm=llm, node_prompt=(
    "Restate the user's question as ONE sharp, concrete decision. One sentence."))
optimist = AgentNode(name="optimist", llm=llm, node_prompt=(
    "Relentless optimist. Argue FOR the bold move — two strongest upsides."))
skeptic = AgentNode(name="skeptic", llm=llm, node_prompt=(
    "Hard-nosed skeptic. Argue AGAINST — the two biggest risks."))
pragmatist = AgentNode(name="pragmatist", llm=llm, node_prompt=(
    "Pragmatist. Propose the smallest concrete next step that de-risks it."))
verdict = AgentNode(name="verdict", llm=llm, node_prompt=(
    "You are the chair. Weigh all three above into a balanced one-paragraph verdict."))

# The line that matters: the three personas run IN PARALLEL, then join at `verdict`.
frame > fanout(optimist, skeptic, pragmatist) > verdict

panel = AgenticGraph(start_node=frame, end_nodes={verdict})   # schema-free

out = panel.invoke(message="Should an early-stage SaaS rewrite its monolith into microservices?")
print(out["messages"][-1].content)        # the verdict
panel.summary()                           # the topology table (below)
print(out["token"])                       # per-model token totals
```

Runs from a single paste with just `OPENAI_API_KEY` set. Full version:
[`examples/panel.py`](https://github.com/TeeratP/agentic-framework/blob/main/examples/panel.py).

## What you get

- **`>` wiring** — `a > b > c` builds the graph; branches index by choice. No
  `add_node`/`add_edge` boilerplate.
- **Parallel `fanout(...)` + deferred join** — `start > fanout(a, b) > combine`
  runs `a` and `b` concurrently and joins **once** after both finish (the
  bracket form `start > [a, b] > combine` wires identically).
- **`worker.map("field")` map-reduce** — `dispatch > summarize.map("docs") > reduce`
  fans a worker out over a state list via LangGraph `Send`, once per item, in
  parallel, then joins once.
- **Schema-free typed state** — nodes default to `messages`; `reads=[...]` /
  `writes=[...]` give one node multi-key IO dispatched **by value type** (a
  message-list read is history, a scalar read is interpolated into the prompt).
  `writes={"score": int}` returns native-typed structured output (a real `int`,
  not a string).
- **`message=` invoke shorthand** — `graph.invoke(message="...")` wraps a string
  (or list of messages) onto `messages`; the full `invoke({...})` state form
  still works.
- **Per-model token usage** — `out["token"]` is a `{model: {total/input/output_tokens}}`
  breakdown accumulated across every node.
- **Opt-in OpenAI prompt caching** — `AgenticGraph(..., prompt_cache=True)`
  threads one cache key through every OpenAI `AgentNode` call.
- **Compile-time validation + `summary()`** — the constructor runs a forward
  dataflow analysis and **fails the build** if a node reads a key nothing
  produces upstream (with the offending key and real writer named), and
  `summary()` prints a Keras-`model.summary()`-style topology table:

```
AgenticGraph 'graph'   state=AgenticState
initial: log, messages, token
-----------------------------------------------------------------
node        type       reads     writes        available
frame       AgentNode  messages  log,messages  log,messages,token
optimist    AgentNode  messages  log,messages  log,messages,token
verdict     AgentNode  messages  log,messages  log,messages,token
skeptic     AgentNode  messages  log,messages  log,messages,token
pragmatist  AgentNode  messages  log,messages  log,messages,token
-----------------------------------------------------------------
5 nodes · 0 errors · 0 warning(s)
```

The offline, no-API-key tour of parallelism, map-reduce, typed IO, and
validation lives in [`examples/parallel_usage.py`](https://github.com/TeeratP/agentic-framework/blob/main/examples/parallel_usage.py).

## vs. LangChain's Functional API

The closest comparison isn't raw graphs — it's LangChain's own Functional API
(`@entrypoint` / `@task`), which also lets you skip explicit graph wiring. The
difference is **visibility of control flow**:

| | Functional API (`@entrypoint`/`@task`) | pttai |
|---|---|---|
| Control flow | hidden in plain Python (loops, `if`, `await`) | an explicit, declarative DAG |
| Fan-out / join | you orchestrate futures by hand | `fanout(...)` / `.map("field")`, one line |
| Inspect the topology | run it and trace | `summary()` prints the static DAG |
| Catch dataflow bugs | at runtime | at **compile time**, before any invoke |

Both are concise. pttai's edge is that the topology is *inspectable and
validatable* — you can see the fan-out/join/map-reduce structure, render it, and
have the compiler reject read-before-written bugs — whereas the Functional API
hides the graph inside ordinary Python, so you lose the auditable DAG.

## How it works

`a > b` doesn't build an edge — it just sets `a.children = [b]` and returns `b`,
so `a > b > c` builds a linked structure in memory. `AgenticGraph(...)` walks
that structure **once** at construction, emits the real LangGraph
`add_node`/`add_edge`/`Send` calls, runs the dataflow validator, and `compile()`s
to a native `StateGraph` — which `AgenticGraph` subclasses. So pttai is a
build-time convenience that disappears at runtime: the execution underneath is
plain LangGraph (streaming, async, durability, checkpointers, LangSmith all
included), and you can drop down to it anytime. No lock-in.

## Node types

All nodes are callables (`__call__(state) -> delta`) invoked by LangGraph with
the shared state. They return **only the keys they update**; reducers merge them.

The two LLM-backed nodes (`AgentNode`, `DecisionNode`) share a common `LLMNode`
base that owns the model binding and the tool-call loop.

| Node | Purpose |
|------|---------|
| **`AgentNode`** | Prepends `node_prompt` as a `SystemMessage`, calls the LLM, returns a delta. Pass `tools=[...]` in the constructor to wrap bare callables as `StructuredTool`s and run an internal tool-call loop (capped by `max_tool_iterations`, default 25). `reads`/`writes` give typed multi-key IO. Optional `reasoning_effort` (`"low"`/`"medium"`/`"high"`) for gpt-5.x. |
| **`DecisionNode`** | LLM branching. Reads from `input_field`, writes its choice to its per-node `decision_{name}` channel, routes via conditional edges over a `Literal[*choices]` structured output — the model can only return a valid branch. Also accepts `tools=[...]`: it runs a tool-gathering loop first, *then* routes via structured output (the two never share a call). Wired by indexing a choice (`decision["x"] > handler`); `decision > x` is an error. |
| **`ConditionNode`** | Deterministic branching with **no LLM**. A Python predicate `condition(state) -> str` returns one of `choices`; routing is free, deterministic, and prompt-less. Wired like `DecisionNode` (`cond["x"] > handler`). |
| **`HumanNode`** | Resumable human-in-the-loop via LangGraph's `interrupt()`. Surfaces a message (or a custom `show`) for review; the human's reply lands `into` `messages` (or any key a router can gate on). Resumes when the graph is built with a `checkpointer` and invoked with a `thread_id`, via `Command(resume=value)`. |

All node types also accept `cache_ttl` (LangGraph `CachePolicy`) and `retry`
(`RetryPolicy`); `AgenticGraph` auto-provides an `InMemoryCache` when any node
sets `cache_ttl`. An `AgenticGraph` can itself be embedded as a node in a larger
graph (`graph_0 > graph_1`), and RAG helpers (`make_retriever_tool`, optional
`ChromaRAG`) wrap any LangChain retriever as a tool you hand to
`AgentNode(tools=[...])`.

## State

`AgenticState` is a `TypedDict` of reduced channels; custom schemas just add more:

- **`messages`** — `add_messages` reducer: appends, replaces by matching `id`,
  coerces bare strings to `HumanMessage`, and merges parallel branches.
- **`log`** — `operator.add`: every node appends a trace line. Seed it with `[]`
  on invoke to capture the trace.
- **`decision_{name}`** — each router (`DecisionNode`/`ConditionNode`) writes its
  choice to its OWN per-node channel `decision_{node_name}` (auto-registered as a
  plain last-writer-wins key), read by that node's `route()`. There is no shared
  `decision` channel, so multiple routers never clobber each other.
- **`token`** — per-model usage totals accumulated across nodes.

Nodes return deltas and never mutate state in place — that's what keeps
checkpointing, parallel-branch merges, and subgraph composition correct rather
than racy.

### Free observability: the `token` and `log` channels

Total token spend and a per-node trace come **automatically — no callbacks, no
custom reducers**. Every LLM call's usage is summed into `token`; every node
appends a trace line to `log`:

```python
out = graph.invoke({"messages": ["..."], "log": []})   # seed log=[] to capture it
print(out["token"])   # {'gpt-5.4-nano': {'input_tokens': 42, 'output_tokens': 88, 'total_tokens': 130, ...}}
print(out["log"])     # ['frame:...', 'optimist:...', ...] — one line per node/tool call
```

`token` is the **run total** — it accumulates across every node, tool-loop
call, and parallel branch, deep-summed per model. In raw LangGraph you'd hand-wire
a usage callback plus a custom summing channel to get this. (Offline, with no
`OPENAI_API_KEY`, the fake model reports no usage, so `token` is an empty `{}` —
the accounting runs, there's just nothing to count; a real model fills it as
shown.) Side-by-side vs. raw LangGraph:
[`examples/basics/13_token_and_log.py`](https://github.com/TeeratP/agentic-framework/blob/main/examples/basics/13_token_and_log.py).

## Limitations

Kept honest on purpose:

- **Structured multi-write list fields are `str`-only.** `writes=["a", "b"]`
  produces one `str` field per key; use the dict form `writes={"a": int}` for
  native-typed structured output.
- **Map workers don't echo their source item** and must output `messages` (the
  default) — a non-message write would race N parallel workers on a plain key.
- **`[b, c] > [d, e]` isn't supported.** Two fan-outs chained directly is
  Python's element-wise list compare, not join wiring — insert a node between.
- **Async is graph-level only.** `ainvoke`/`astream` run the sync nodes in
  LangGraph's threadpool; true per-node async LLM calls aren't implemented.
- **`reasoning_effort` is `AgentNode`-only** — it conflicts with `DecisionNode`'s
  structured output on current OpenAI models.

## Running it

```bash
python -m pytest tests/                # full suite, no API calls (a scripted FakeLLM stands in)
python examples/parallel_usage.py      # offline tour: parallel + map-reduce + validation
python examples/panel.py               # live multi-agent panel (needs OPENAI_API_KEY)
python examples/vs_langgraph.py        # the 3-vs-10 comparison, both ways (needs OPENAI_API_KEY)
```

The **166-test** suite covers state reducers, graph construction, routing, the
tool-call loop, interrupt/resume, RAG tool wiring, streaming/async, configurable
fields, parallel fan-out/join, map-reduce, multi-key IO, static validation, and
node caching/retry/`reasoning_effort`/`durability`.

## License

MIT — see [LICENSE](https://github.com/TeeratP/agentic-framework/blob/main/LICENSE).
