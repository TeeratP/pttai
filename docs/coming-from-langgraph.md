# Coming from LangGraph

pttai *is* LangGraph underneath — `AgenticGraph` subclasses `StateGraph` and
compiles down to one. So nothing you know is wasted; pttai just replaces the
imperative `add_node`/`add_edge`/`add_conditional_edges`/`Send` plumbing with a
declarative `>` layer, and adds a build-time validator on top. This page maps
the LangGraph API you already use to its pttai equivalent.

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

## A worked comparison

The canonical tool-using agent — 10 lines of LangGraph, 3 of pttai — is in
[Getting started](getting-started.md), and every architecture in the
[Architectures gallery](examples/architectures.md) shows the pttai version
followed by a `# --- equivalent in raw LangGraph ---` block in the same file.

## The reviewer's objection: "why not raw LangGraph?"

You keep the entire LangGraph runtime — this is a build-time convenience that
disappears at compile time. What you gain is: (1) far less wiring boilerplate,
(2) an *inspectable* topology (`summary()` prints the DAG), and (3) a
**dataflow validator that rejects read-before-write and unwired-branch bugs
before you spend a single token**. That third item has no equivalent in raw
LangGraph and is the reason to adopt pttai.
