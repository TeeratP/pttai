# Agentic Framework

A thin, declarative DSL over [LangGraph](https://langchain-ai.github.io/langgraph/) for building agent workflows. You instantiate `Node` objects, wire them with the `>` operator, and hand the start/end nodes to `AgenticGraph`, which compiles down to a native LangGraph `StateGraph`.

The value-add over raw LangGraph is the **`>` wiring syntax** and a small set of node abstractions: an automatic tool-call loop, structured-output routing, resumable human-in-the-loop, and reducer-based state. Because it compiles to plain LangGraph, you keep checkpointing, streaming, and LangSmith tracing for free.

```python
randomizer > classifier
classifier["positive"] > positive_handler
classifier["negative"] > negative_handler
```

That's the whole mental model: `>` builds a graph, branches index by choice.

## Features

- **`>` wiring DSL** — build graphs by writing `a > b`; branches index by choice (`decision["x"] > handler`). Compiles to a native LangGraph `StateGraph`.
- **Automatic tool-call loop** — `AgentNode.bind_tools([fn])` wraps bare callables as `StructuredTool`s and runs the call/respond loop for you (capped by `max_tool_iterations`).
- **Structured-output routing** — `DecisionNode` constrains the LLM to a `Literal` of your choice names, so it can only return a valid branch; the label lands in a dedicated `decision` field, never in `messages`.
- **Resumable human-in-the-loop** — `InputNode` uses LangGraph's `interrupt()`; resume with a `checkpointer` + `Command(resume=...)`.
- **Reducer-based state** — deltas merged by `add_messages` / `operator.add`, which keeps checkpointing, parallel branches, and subgraph composition correct.
- **Graphs as nodes** — an `AgenticGraph` can be embedded inside another (`graph_0 > graph_1`).
- **RAG helpers** — `make_retriever_tool` wraps any LangChain retriever; `ChromaRAG` is an optional Chroma convenience.
- **Per-node knobs** — `cache_ttl`, `retry`, and `reasoning_effort` (gpt-5.x) per node.
- **Free LangGraph plumbing** — streaming (`stream`/`astream`), async (`ainvoke`/`astream`), `durability`, and LangSmith tracing, because it's plain LangGraph underneath.

## How it works

1. **Wiring is deferred.** `a > b` just sets `a.child = b` and returns `b`, so chains like `a > b > c` build a linked list in memory — no edges exist yet.
2. **`AgenticGraph(...)` materializes the graph.** At construction it walks the `.child` pointers, emits the real LangGraph `add_node`/`add_edge` calls, and `compile()`s. The graph is frozen at build time.
3. **Termination is explicit.** Nodes you pass in `end_nodes` get an edge to `END`; everything else recurses into its `.child`.
4. **Loops are allowed.** Revisited nodes are short-circuited by name, so cycles (e.g. a research loop) just work. Node names must be unique within a graph.
5. **Decision nodes route by structured output.** `decision["positive"] > handler` sets the branch; the LLM is wrapped with `with_structured_output` over a `Literal` of the choice names, so it can only return a valid choice. The label is written to a dedicated `decision` state field — it never pollutes `messages`.

## Install

Requires **Python ≥ 3.10**. Core deps are LangGraph ≥ 1.0, LangChain-core ≥ 1.0, and Pydantic 2.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                 # core
pip install -e ".[openai]"       # + langchain-openai & python-dotenv (for the example/notebook)
pip install -e ".[rag]"          # + langchain-chroma (for ChromaRAG)
pip install -e ".[dev]"          # + pytest
```

For live model calls, copy `.env.example` to `.env` and set `OPENAI_API_KEY`.

## Quickstart

A tool call → decision → branch graph, wired entirely with `>` (full runnable version in [`examples/sample_usage.py`](examples/sample_usage.py)):

```python
import random
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from agentic_framework.graph import AgenticGraph
from agentic_framework.nodes import AgentNode, DecisionNode
from agentic_framework.state import AgenticState


def random_number(maximum: int) -> int:
    """Return a random integer between 1 and `maximum` (inclusive)."""
    return random.randint(1, maximum)


llm = ChatOpenAI(model="gpt-5.4-nano")

# An agent that can call a tool. bind_tools wraps the bare function as a
# StructuredTool and runs the tool-call loop automatically.
randomizer = AgentNode(
    name="randomizer",
    llm=llm,
    node_prompt="Use the random_number tool to pick a number, then state it.",
)
randomizer.bind_tools([random_number])

# A decision node returns one of `choices` into the `decision` state field.
classifier = DecisionNode(
    name="classifier",
    llm=llm,
    node_prompt="If the number is greater than 5 answer 'positive', else 'negative'.",
    choices=["positive", "negative"],
)

positive_handler = AgentNode(name="positive_handler", llm=llm,
                             node_prompt="Report the number in an upbeat tone.")
negative_handler = AgentNode(name="negative_handler", llm=llm,
                             node_prompt="Report the number in a gloomy tone.")

# Wire it.
randomizer > classifier
classifier["positive"] > positive_handler
classifier["negative"] > negative_handler

graph = AgenticGraph(
    state=AgenticState,
    start_node=randomizer,
    end_nodes={positive_handler, negative_handler},
)

result = graph.invoke({
    "messages": [HumanMessage(content="Give me a random number up to 10.")],
    "log": [],
})
print("routed to :", result["decision"])
print("final reply:", result["messages"][-1].content)
for line in result["log"]:          # per-node + tool-call trace
    print("  -", line)
```

## Node types

All nodes are callables (`__call__(state) -> delta`) invoked by LangGraph with the shared state. They return **only the keys they update** — the reducers merge them; nodes never mutate state in place.

| Node | Purpose |
|------|---------|
| **`AgentNode`** | Prepends its `node_prompt` as a `SystemMessage`, calls the LLM, returns a delta. `bind_tools(...)` wraps bare callables and runs an internal tool-call loop (capped by `max_tool_iterations`, default 25), accumulating every message produced this turn into one delta. A non-default `output_field` writes the final response *content* to that key instead of `messages` (transform nodes). Optional `reasoning_effort` (`"low"`/`"medium"`/`"high"`) is passed per-call for gpt-5.x. |
| **`DecisionNode`** | Branching only. Reads from `input_field`, writes its choice to the `decision` field, and routes via conditional edges. Wired by indexing a choice (`decision["x"] > handler`); `decision > x` is an error. |
| **`InputNode`** | Human-in-the-loop via LangGraph's `interrupt()`. Resumes when the graph is built with a `checkpointer` and invoked with a `thread_id`, via `Command(resume=value)`. |

All node types also accept `cache_ttl` (LangGraph `CachePolicy`) and `retry` (LangGraph `RetryPolicy`); `AgenticGraph` auto-provides an `InMemoryCache` when any node sets `cache_ttl`.

## State

`AgenticState` is a `TypedDict` with reduced channels:

- **`messages`** — `add_messages` reducer: appends, replaces by matching `id`, coerces bare strings to `HumanMessage`, and merges parallel branches.
- **`log`** — `operator.add`: every node appends a trace line (`"{name}:{content}"`, tool calls, decisions). Seed it with `[]` on invoke to capture the trace.
- **`decision`** — transient routing key written by `DecisionNode`, read by its `route()`.

Reducer-based deltas are what make checkpointing, parallel branches, and subgraph composition correct. Custom schemas just add more reduced channels (e.g. a `summary` field for an `output_field` node).

## Running it

```bash
python -m pytest tests/            # full suite, no API calls (a scripted FakeLLM stands in)
python examples/sample_usage.py    # live end-to-end demo (needs OPENAI_API_KEY)
```

The 33-test suite covers state reducers, graph construction, routing, the tool-call loop, interrupt/resume, RAG tool wiring, streaming/async, configurable fields, and node caching/retry/`reasoning_effort`/`durability`.

## Limitations

- **Async is graph-level only.** `ainvoke`/`astream` run the sync nodes in LangGraph's threadpool; true per-node async LLM calls aren't implemented (which is also why a per-node `timeout` isn't exposed — LangGraph only times out async nodes).
- **`reasoning_effort` is `AgentNode`-only** — it conflicts with `DecisionNode`'s structured output on current OpenAI models.
- **`ChromaRAG` is untested end-to-end** — only `make_retriever_tool` is covered (against a fake retriever); the live Chroma path needs real embeddings.

## Composition & graphs as nodes

An `AgenticGraph` can itself be a node inside a larger `AgenticGraph` — `graph_0 > graph_1` wires them, and the parent graph embeds the child's compiled graph. Run any graph via `invoke` / `stream` / `ainvoke` / `astream` (all accept an optional LangGraph `config` and `durability`). Pass `checkpointer=...` to enable `InputNode` interrupt/resume.

## Project layout

```
agentic_framework/
  graph.py            # AgenticGraph — walks `.child`, compiles to StateGraph
  node.py             # base Node + the `>` operator
  state.py            # AgenticState (reduced channels)
  nodes/              # AgentNode, DecisionNode, InputNode
  tools/              # RAG tool helpers (make_retriever_tool, ChromaRAG)
examples/sample_usage.py
tests/                # pytest suite (no live API calls)
docs/project.md       # living notes: setup, status, roadmap, rough edges
CLAUDE.md             # stable architecture & conventions
```

See [`docs/project.md`](docs/project.md) for setup details, current status, and the roadmap, and [`CLAUDE.md`](CLAUDE.md) for the architecture in depth.

## License

MIT — see [LICENSE](LICENSE).
