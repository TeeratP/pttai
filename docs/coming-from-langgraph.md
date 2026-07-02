# Coming from LangGraph

pttai *is* LangGraph underneath — `AgenticGraph` subclasses `StateGraph` and
compiles down to one. Nothing you know is wasted: pttai replaces the imperative
`add_node`/`add_edge`/`add_conditional_edges`/`Send` calls with a declarative
`>` layer and adds a build-time validator on top. This page maps the LangGraph
API you already use to its pttai equivalent.

## Mapping table

| LangGraph | pttai | Notes |
|---|---|---|
| `builder.add_node("a", fn)` | `a = AgentNode(name="a", llm=llm, ...)` | The node is the object; you don't register it by name. |
| `builder.add_edge("a", "b")` | `a > b` | `>` sets `a.children = [b]` and returns `b`, so `a > b > c` chains. No edge exists until `AgenticGraph(...)` walks the chain. |
| `builder.add_edge(START, "a")` | `AgenticGraph(start_node=a, ...)` | The start is a constructor arg, not an edge. |
| `builder.add_edge("z", END)` | `AgenticGraph(..., end_nodes={z})` | Termination is controlled by `end_nodes`, not by `.child == None`. |
| fan-out: two `add_edge("a", ...)` calls | `a > fanout(b, c) > d` (or `a > [b, c] > d`) | Branches run **concurrently**; the join `d` is deferred until **all** finish (fires once, not once per branch). |
| `add_conditional_edges("a", router_fn, [...])` | `DecisionNode` (LLM) or `ConditionNode` (predicate) | Index the choices to wire them: `decide["yes"] > handler`. |
| `Send("worker", payload)` + a conditional edge | `dispatch > worker.map("field") > reduce` | `.map("field")` fans `worker` out over `state["field"]`, one `Send` per item, in parallel, then joins once at `reduce`. |
| `ToolNode([tools])` + `tools_condition` + loop-back edge | `AgentNode(tools=[...])` | The tool-call loop (execute → append `ToolMessage` → re-invoke) is built into the node, capped by `max_tool_iterations`. |
| `with_structured_output(Literal[...])` + a router fn | `DecisionNode(choices=[...])` | pttai builds the `Literal` structured output *and* the conditional edges; the model can only return a valid branch. |
| `builder.compile()` | happens inside `AgenticGraph(...)` | Construction walks the wiring, **runs the validator**, and compiles. |
| a subgraph added via `add_node(compiled_subgraph)` | `graph_0 > graph_1` | An `AgenticGraph` composes as a node inside a larger `AgenticGraph`. |
| `MessagesState` | `AgenticState` (the default) | `messages` + reduced `log` / `token` channels; routers auto-register a per-node `decision_{name}` key. Schema-free by default; nodes auto-register keys. |
| — (runtime `KeyError` when you get it wrong) | **compile-time `GraphValidationError`** | The dataflow validator has no LangGraph equivalent. See [the validator](validator.md). |

## When to use which

- **`>` (sequential edge)** — one step then the next. The 90% case.
- **`fanout(a, b, c)` / `[a, b, c]`** — a *fixed, known* set of branches that
  should run in parallel and rejoin (e.g. three critics, then a verdict). Both
  forms wire identically; `fanout(...)` reads better inline.
- **`worker.map("field")`** — a *dynamic* number of parallel tasks: one worker
  applied to every item of a state list (map-reduce / orchestrator-workers).
  Use this instead of hand-rolling `Send`.
- **`DecisionNode`** — the branch should be chosen by an **LLM** (classify,
  triage, route). Constrained to valid choices via structured output. Can take
  `tools=` to gather context before routing.
- **`ConditionNode`** — the branch is decided by **plain Python** (a counter, a
  flag, a length check). Free, deterministic, no model call — ideal for capping
  loops.
- **`AgentNode(tools=[...])`** — any tool-using / ReAct agent. Replaces the
  `ToolNode` + `tools_condition` + loop-back triad with one node.
- **Drop back to raw LangGraph** anytime — the compiled graph is a real
  `StateGraph`, so streaming, async, `durability`, checkpointers, and LangSmith
  all work unchanged. No lock-in.

## Under the hood

`a > b` doesn't build an edge — it sets `a.children = [b]` and returns `b`, so
`a > b > c` builds a linked structure in memory. `AgenticGraph(...)` walks that
structure **once** at construction, emits the real LangGraph
`add_node`/`add_edge`/`Send` calls, runs the dataflow validator, and
`compile()`s to a native `StateGraph` — which `AgenticGraph` subclasses.

So pttai is a build-time convenience that disappears at runtime: the execution
underneath is plain LangGraph, and you can drop down to it anytime.

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

Both are concise. The trade pttai makes is keeping the topology *inspectable
and validatable*: you can see the fan-out/join/map-reduce structure, render it,
and have the build reject read-before-written bugs — where the Functional API
hides the graph inside ordinary Python.

## Why not raw LangGraph?

You keep the entire LangGraph runtime either way — pttai disappears at compile
time. What it buys you: (1) less wiring boilerplate, (2) an inspectable
topology (`summary()` prints the DAG, and the graph renders itself in a
notebook), and (3) a [dataflow validator](validator.md) that rejects
read-before-write and unwired-branch bugs before you spend a token — which has
no equivalent in raw LangGraph.

A worked comparison — the same tool-using agent in 3 lines vs. 10 — is on the
[home page](index.md#the-same-agent-both-ways), and every file in the
[Examples](examples.md) galleries pairs the pttai version with a
`# --- equivalent in raw LangGraph ---` block.
