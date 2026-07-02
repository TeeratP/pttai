# **pttai** — *Pythonic Topology Tools for AI*

**Build LLM agent graphs in a few lines.** pttai is a small declarative DSL over
[LangGraph](https://langchain-ai.github.io/langgraph/): wire self-contained,
tool-using agent nodes together with a `>` operator into a *visible* DAG —
fan-out, map-reduce, structured-output routing, human-in-the-loop — and skip the
`add_node` / `add_edge` / `add_conditional_edges` / `Send` boilerplate. It all
compiles down to a native LangGraph `StateGraph`.

pttai **builds on LangGraph, it doesn't replace it.** LangGraph is well worth
learning directly — especially for experienced users who want full control.
pttai's job is to make that capability more *accessible*: the value is
everything you *don't* write, folded into a small `>`-DSL. And because each node
declares the state keys it reads and writes, pttai statically checks your graph
for read-before-written dataflow bugs *before* you invoke. You keep the whole
LangGraph ecosystem (streaming, async, checkpointers, LangSmith) and can drop
down to raw LangGraph anytime — no lock-in.

Full documentation: **[teeratp.github.io/pttai](https://teeratp.github.io/pttai/)**.

[![PyPI](https://img.shields.io/pypi/v/pttai)](https://pypi.org/project/pttai/)
[![Docs](https://img.shields.io/badge/docs-teeratp.github.io%2Fpttai-blue)](https://teeratp.github.io/pttai/)
[![CI](https://github.com/TeeratP/pttai/actions/workflows/ci.yml/badge.svg)](https://github.com/TeeratP/pttai/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-1.0-orange)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/TeeratP/pttai/blob/main/LICENSE)

## pttai vs. raw LangGraph

The same tool-using agent — an LLM that calls `add` / `multiply` in a loop until
it has the answer. Ask it *"What is 21 + 21, then times 3?"* and both print
**126**. The only thing that differs is how much graph plumbing you write:
**3 lines vs. 10.**

Shared setup for both versions — the model and the two tools:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-5.4-nano")
# swap for any LangChain chat model, e.g.:
#   from langchain_anthropic import ChatAnthropic;              llm = ChatAnthropic(model="claude-opus-4-8")
#   from langchain_google_genai import ChatGoogleGenerativeAI;  llm = ChatGoogleGenerativeAI(model="gemini-...")

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b
```

```python
# pttai
from pttai import AgentNode, AgenticGraph

agent = AgentNode(llm=llm, tools=[add, multiply])   # name inferred from the variable -> "agent"
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
[`examples/vs_langgraph.py`](https://github.com/TeeratP/pttai/blob/main/examples/vs_langgraph.py).

## Catch dataflow bugs at build time

Because every node declares what it reads and writes, `AgenticGraph(...)` runs a
forward dataflow analysis **at construction** and *fails the build* — before a
single model call — if a node reads a state key that nothing upstream produces:

```python
from pttai import AgentNode, AgenticGraph

early = AgentNode(llm=llm, reads=["summary"])    # name inferred -> "early"; reads 'summary'...
late  = AgentNode(llm=llm, writes=["summary"])   # ...-> "late"; but it's produced downstream
early > late

AgenticGraph(start_node=early, end_nodes={late})
# GraphValidationError: 'early' reads computed key 'summary' but no upstream node produces it
```

Raw LangGraph compiles the same graph and only trips at runtime, on the first
input that exercises the broken path. pttai names the offending node and key at
build time; `graph.validate()` runs the same check on demand. Every bug class
it catches, with the verbatim errors:
**[The validator](https://teeratp.github.io/pttai/validator/)**.

## Examples

Two runnable galleries make the "pttai vs. LangGraph" story concrete:

- **[`examples/basics/`](https://github.com/TeeratP/pttai/blob/main/examples/basics/)** — one file per feature, each showing
  the pttai version *and* the equivalent raw-LangGraph version side by side. The
  fastest way to see exactly what plumbing pttai folds away — tool loops,
  fan-out/join, map-reduce, structured-output routing, typed state IO,
  human-in-the-loop — one concept at a time.
- **[`examples/architectures/`](https://github.com/TeeratP/pttai/blob/main/examples/architectures/)** — famous agent
  patterns (router, evaluator-optimizer, orchestrator-workers, reflection, and
  more) built end-to-end in pttai, so you can lift a whole topology instead of a
  single node.

Start with `basics/` to learn the primitives, then reach for `architectures/`
when you're wiring a real system.

## Interactive playground

`python demo/app.py` launches a local Gradio playground: paste a `>`-DSL snippet,
click **Build + Validate**, and see the compiled LangGraph diagram and the
build-time validator output side by side — no API key needed. See [`demo/`](https://github.com/TeeratP/pttai/blob/main/demo/).

## Install

```bash
pip install pttai

# pttai works with ANY LangChain chat model — install the provider you want:
pip install langchain-openai python-dotenv   # OpenAI (used in the examples below)
pip install langchain-anthropic              # Anthropic
pip install langchain-google-genai           # Google
```

Nodes take the model via `llm=` — pass any LangChain `BaseChatModel`. (A
`pttai[openai]` extra exists as a shortcut for `langchain-openai` +
`python-dotenv`, but it's convenience only, not a requirement.) Note:
`reasoning_effort` and structured-output routing are tuned to OpenAI gpt-5.x
semantics; core wiring, the tool-call loop, and routing are otherwise
provider-neutral.

Or from source (for development):

```bash
git clone https://github.com/TeeratP/pttai && cd pttai
python -m venv .venv && source .venv/bin/activate
pip install -e ".[openai,dev]"
```

Requires **Python ≥ 3.10** (core deps: LangGraph ≥ 1.0, langchain-core ≥ 1.0,
Pydantic 2). Other extras: `[rag]` (langchain-chroma for `ChromaRAG`), `[dev]`
(pytest). For live model calls, set `OPENAI_API_KEY` (or your provider's key)
in your environment or a `.env` file.

## 30-second example: a multi-agent panel

A question goes to `frame` (which sharpens it into one concrete decision), fans
out to three rival personas — optimist / skeptic / pragmatist — who argue
**concurrently**, then `verdict` weighs every argument into a one-paragraph
ruling. The whole thing is the one wiring line at the bottom.

```python
from pttai import AgentNode, AgenticGraph, fanout
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-5.4-nano")   # swap for any LangChain chat model — see Install above

# names are inferred from the variables (frame, optimist, ...) — no name= needed
frame = AgentNode(llm=llm, node_prompt=(
    "Restate the user's question as ONE sharp, concrete decision. One sentence."))
optimist = AgentNode(llm=llm, node_prompt=(
    "Relentless optimist. Argue FOR the bold move — two strongest upsides."))
skeptic = AgentNode(llm=llm, node_prompt=(
    "Hard-nosed skeptic. Argue AGAINST — the two biggest risks."))
pragmatist = AgentNode(llm=llm, node_prompt=(
    "Pragmatist. Propose the smallest concrete next step that de-risks it."))
verdict = AgentNode(llm=llm, node_prompt=(
    "You are the chair. Weigh all three above into a balanced one-paragraph verdict."))

frame > fanout(optimist, skeptic, pragmatist) > verdict   # parallel, then join

panel = AgenticGraph(start_node=frame, end_nodes={verdict})   # schema-free

out = panel.invoke(message="Should an early-stage SaaS rewrite its monolith into microservices?")
print(out["messages"][-1].content)        # the verdict
panel.summary()                           # the topology table (below)
print(out["token"])                       # per-model token totals
```

Runs from a single paste with only `OPENAI_API_KEY` set. Full version:
[`examples/panel.py`](https://github.com/TeeratP/pttai/blob/main/examples/panel.py).

The graph also renders itself. In a notebook, end a cell with the graph — or
call `display(panel)` — and you get the compiled DAG:

```python
from IPython.display import display
display(panel)    # or make `panel` the last expression of the cell
```

![pttai renders your graph](https://raw.githubusercontent.com/TeeratP/pttai/main/docs/assets/graph-example.png)

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
  `summary()` prints a `model.summary()`-style topology table:

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
validation lives in [`examples/parallel_usage.py`](https://github.com/TeeratP/pttai/blob/main/examples/parallel_usage.py).

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

`a > b` doesn't build an edge — it sets `a.children = [b]` and returns `b`,
so `a > b > c` builds a linked structure in memory. `AgenticGraph(...)` walks
that structure **once** at construction, emits the real LangGraph
`add_node`/`add_edge`/`Send` calls, runs the dataflow validator, and `compile()`s
to a native `StateGraph` — which `AgenticGraph` subclasses. So pttai is a
build-time convenience that disappears at runtime: the execution underneath is
plain LangGraph (streaming, async, durability, checkpointers, LangSmith all
included), and you can drop down to it anytime. No lock-in.

## Node types

Four node types cover the graphs above: **`AgentNode`** (an LLM step with an
optional built-in tool-call loop, capped by `max_tool_iterations=25`),
**`DecisionNode`** (LLM branching via structured output, constrained to valid
choices), **`ConditionNode`** (deterministic branching from a plain Python
predicate — no model call), and **`HumanNode`** (resumable human-in-the-loop
via `interrupt()`). The LLM-backed nodes share an `LLMNode` base; all take
`cache_ttl`/`retry`, and an `AgenticGraph` can itself be embedded as a node.
Each type with a runnable snippet:
**[Node types](https://teeratp.github.io/pttai/node-types/)**.

## State

`AgenticState` is a `TypedDict` of reduced channels — `messages`
(`add_messages`), a `log` trace (`operator.add`), per-model `token` totals, and
a private `decision_{name}` key per router. Nodes return deltas, never mutate
state in place, and reducers merge them — the full channel model is in
**[State & observability](https://teeratp.github.io/pttai/state/)**.

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
call, and parallel branch, deep-summed per model. In raw LangGraph you'd
hand-wire a usage callback plus a custom summing channel to get this.
Side-by-side vs. raw LangGraph:
[`examples/basics/13_token_and_log.py`](https://github.com/TeeratP/pttai/blob/main/examples/basics/13_token_and_log.py).

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
- **`reasoning_effort` is `AgentNode`-only** — it conflicts with `DecisionNode`'s
  structured output on current OpenAI models.

## Running it

```bash
python -m pytest tests/                # full suite, no API calls (a scripted FakeLLM stands in)
python examples/parallel_usage.py      # offline tour: parallel + map-reduce + validation
python examples/panel.py               # live multi-agent panel (needs OPENAI_API_KEY)
python examples/vs_langgraph.py        # the 3-vs-10 comparison, both ways (needs OPENAI_API_KEY)
```

The **165-test** suite covers state reducers, graph construction, routing, the
tool-call loop, interrupt/resume, RAG tool wiring, streaming/async, configurable
fields, parallel fan-out/join, map-reduce, multi-key IO, static validation, and
node caching/retry/`reasoning_effort`/`durability`.

## License

MIT — see [LICENSE](https://github.com/TeeratP/pttai/blob/main/LICENSE).
