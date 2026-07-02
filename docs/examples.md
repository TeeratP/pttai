# Examples

The repo ships two runnable galleries. Every file shows the **pttai version
first**, then a runnable `# --- equivalent in raw LangGraph ---` block in the
same file, so you can read the same behavior built both ways:

- [`examples/basics/`](https://github.com/TeeratP/pttai/tree/main/examples/basics)
  — one feature per file, 13 files. Learn the primitives one at a time.
- [`examples/architectures/`](https://github.com/TeeratP/pttai/tree/main/examples/architectures)
  — the canonical agent patterns from Anthropic's
  [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
  and the common LangGraph tutorials, 8 files. Lift a whole topology.

## Running offline

Everything runs **offline with no API key**: the shared helper
`examples/_llm.py` (imported as `get_llm()`) returns a scripted fake chat model
when `OPENAI_API_KEY` is unset. Set the key and the exact same files call real
OpenAI (`gpt-5.4-nano`) instead — no code change.

```bash
python examples/basics/01_single_agent.py                   # one file
for f in examples/basics/[0-9]*.py; do python "$f"; done    # all basics
for f in examples/architectures/*.py; do python "$f"; done  # all architectures
```

The snippets on this page are lifted from those files, so they use `get_llm()`
— substitute your own `llm` to run them standalone.

### Capping loops offline

The looping architectures need care offline: the scripted fake always picks a
model's first structured choice, so a loop gated by a `DecisionNode` would
never choose "stop". The evaluator–optimizer and supervisor therefore gate with
a [`ConditionNode`](node-types.md#conditionnode) that *counts iterations* in
the reduced `log` channel — every node appends a trace line there, so counting
lines that start with a node's name counts its runs. The cap is a hard
termination guarantee that costs nothing and works the same with a real model:

```python
MAX_ROUNDS = 2

def _gate(state) -> str:
    rounds = sum(1 for line in state["log"] if line.startswith("generate"))
    return "accept" if rounds >= MAX_ROUNDS else "refine"
```

## Basics: one primitive per file

| File | Feature |
|------|---------|
| [`01_single_agent.py`](https://github.com/TeeratP/pttai/blob/main/examples/basics/01_single_agent.py) | One `AgentNode` — smallest useful graph |
| [`02_tools.py`](https://github.com/TeeratP/pttai/blob/main/examples/basics/02_tools.py) | `AgentNode(tools=[...])` with a built-in tool-call loop |
| [`03_sequential.py`](https://github.com/TeeratP/pttai/blob/main/examples/basics/03_sequential.py) | Sequential chain with the `a > b > c` operator |
| [`04_decision_routing.py`](https://github.com/TeeratP/pttai/blob/main/examples/basics/04_decision_routing.py) | `DecisionNode` — LLM picks the branch (structured output) |
| [`05_condition_routing.py`](https://github.com/TeeratP/pttai/blob/main/examples/basics/05_condition_routing.py) | `ConditionNode` — deterministic Python predicate, no LLM |
| [`06_parallel_fanout.py`](https://github.com/TeeratP/pttai/blob/main/examples/basics/06_parallel_fanout.py) | `start > fanout(a, b) > join` — concurrent branches + deferred join |
| [`07_map_reduce.py`](https://github.com/TeeratP/pttai/blob/main/examples/basics/07_map_reduce.py) | `worker.map("field") > collector` — fan out over a list, join once |
| [`08_typed_io.py`](https://github.com/TeeratP/pttai/blob/main/examples/basics/08_typed_io.py) | `reads=[...]`, `writes={"score": int}` — typed structured I/O |
| [`09_human_in_the_loop.py`](https://github.com/TeeratP/pttai/blob/main/examples/basics/09_human_in_the_loop.py) | `HumanNode` + `InMemorySaver` + `Command(resume=...)` |
| [`10_graph_composition.py`](https://github.com/TeeratP/pttai/blob/main/examples/basics/10_graph_composition.py) | An `AgenticGraph` wired in as a node in a bigger graph |
| [`11_node_policies.py`](https://github.com/TeeratP/pttai/blob/main/examples/basics/11_node_policies.py) | `cache_ttl` / `retry` / `reasoning_effort` per-node knobs |
| [`12_validation_summary.py`](https://github.com/TeeratP/pttai/blob/main/examples/basics/12_validation_summary.py) | `graph.validate()` + `graph.summary()` introspection |
| [`13_token_and_log.py`](https://github.com/TeeratP/pttai/blob/main/examples/basics/13_token_and_log.py) | `state['token']` / `state['log']` — token accounting + per-node trace |

For example, the whole map-reduce pattern
([`07_map_reduce.py`](https://github.com/TeeratP/pttai/blob/main/examples/basics/07_map_reduce.py))
is one wiring line — `.map("docs")` fans the worker out over `state["docs"]`,
one parallel LangGraph `Send` per item, and the collector joins once:

```python
dispatch = AgentNode(llm=get_llm(), node_prompt="Kick off the summaries.")
summarize = AgentNode(llm=get_llm(), node_prompt="Summarize this document.")
collect = AgentNode(llm=get_llm(), node_prompt="Combine the summaries into a digest.")

dispatch > summarize.map("docs") > collect        # fan out over state["docs"]

graph = AgenticGraph(start_node=dispatch, end_nodes={collect})
out = graph.invoke("Summarize the docs.", docs=["alpha memo", "beta memo", "gamma memo"])
```

## Architectures: whole topologies

| File | Pattern | Flow |
|------|---------|------|
| [`react_agent.py`](https://github.com/TeeratP/pttai/blob/main/examples/architectures/react_agent.py) | Tool-calling ReAct loop — one `AgentNode(tools=[...])` runs think→act→observe until it answers. | `agent ⇄ tools → answer` |
| [`prompt_chaining.py`](https://github.com/TeeratP/pttai/blob/main/examples/architectures/prompt_chaining.py) | Fixed sequence of steps, each building on the last, with an optional early-exit gate. | `extract → gate → expand → finalize` |
| [`routing.py`](https://github.com/TeeratP/pttai/blob/main/examples/architectures/routing.py) | Classify then dispatch; `DecisionNode(tools=...)` gathers context with a tool, then routes. | `triage → {billing, technical, general}` |
| [`parallelization.py`](https://github.com/TeeratP/pttai/blob/main/examples/architectures/parallelization.py) | Sectioning / voting — branches run concurrently, then join. | `frame → (optimist ∥ skeptic ∥ pragmatist) → verdict` |
| [`orchestrator_workers.py`](https://github.com/TeeratP/pttai/blob/main/examples/architectures/orchestrator_workers.py) | Dynamic fan-out over sub-tasks via `worker.map("field")`, then synthesize. | `plan → worker×N → synthesize` |
| [`evaluator_optimizer.py`](https://github.com/TeeratP/pttai/blob/main/examples/architectures/evaluator_optimizer.py) | Generate → evaluate → refine loop; a `ConditionNode` caps the rounds. | `generate → evaluate → gate ⟳ / accept → finalize` |
| [`reflection.py`](https://github.com/TeeratP/pttai/blob/main/examples/architectures/reflection.py) | Generate → self-critique → revise, one terminating pass. | `draft → critique → revise` |
| [`supervisor.py`](https://github.com/TeeratP/pttai/blob/main/examples/architectures/supervisor.py) | A supervisor `DecisionNode` delegates to worker agents in a capped loop. | `supervisor → {researcher, writer} → gate ⟳ / done → report` |

Two of them, inline. The ReAct agent
([`react_agent.py`](https://github.com/TeeratP/pttai/blob/main/examples/architectures/react_agent.py))
— the whole think→act→observe loop lives inside one node:

```python
def search(query: str) -> str:
    """Look up a fact."""
    return f"(stub) top result for {query!r}: 42"

def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression."""
    return str(eval(expression, {"__builtins__": {}}, {}))

agent = AgentNode(llm=get_llm(), tools=[search, calculator],
                  node_prompt="Reason step by step and use tools to answer.")
graph = AgenticGraph(start_node=agent, end_nodes={agent})
print(graph.invoke("What is 6 times 7?")["messages"][-1].content)
```

And the evaluator–optimizer
([`evaluator_optimizer.py`](https://github.com/TeeratP/pttai/blob/main/examples/architectures/evaluator_optimizer.py)),
using the `_gate` counter from [Capping loops offline](#capping-loops-offline)
above:

```python
generate = AgentNode(name="generate", llm=get_llm(), node_prompt="Write or improve the draft.")
evaluate = AgentNode(name="evaluate", llm=get_llm(), node_prompt="Critique the draft; list concrete fixes.")
gate = ConditionNode(name="gate", condition=_gate, choices=["refine", "accept"])
finalize = AgentNode(name="finalize", llm=get_llm(), node_prompt="Return the polished final draft.")

generate > evaluate > gate
gate["refine"] > generate      # loop back to improve
gate["accept"] > finalize

graph = AgenticGraph(start_node=generate, end_nodes={finalize})
```

## Topology diagrams

```text
react_agent
    user ─▶ [ agent ]──tool_call──▶ (tool) ──result──┐
              ▲                                       │
              └───────────────── loop ────────────────┘
              └──▶ final answer

prompt_chaining
    input ─▶ [ extract ] ─▶ <gate?> ──pass──▶ [ expand ] ─▶ [ finalize ] ─▶ out
                                └──fail──▶ [ reject ] ─▶ out

routing
    input ─▶ [ triage ]──(lookup tool)──┬─billing──▶ [ billing ]  ─▶ out
                                        ├─technical▶ [ technical ]─▶ out
                                        └─general──▶ [ general ]  ─▶ out

parallelization
                  ┌─▶ [ optimist ]   ──┐
    [ frame ] ────┼─▶ [ skeptic ]    ──┼─▶ [ verdict ] ─▶ out
                  └─▶ [ pragmatist ] ──┘
                  (join deferred until all three finish)

orchestrator_workers
    [ plan ] ─▶ worker.map("subtasks") ═══▶ [ worker ] × N ═══▶ [ synthesize ] ─▶ out

evaluator_optimizer
    ┌──────────────── refine ────────────────┐
    ▼                                         │
    [ generate ] ─▶ [ evaluate ] ─▶ <gate> ──┘
                                      └──accept──▶ [ finalize ] ─▶ out

reflection
    [ draft ] ─▶ [ critique ] ─▶ [ revise ] ─▶ out

supervisor
    ┌─────────────── continue ────────────────┐
    ▼                                          │
    [ supervisor ] ─┬─researcher─▶ [ researcher ]─┐
                    └─writer─────▶ [ writer ]────┴▶ <gate>
                                                     └─done─▶ [ report ] ─▶ out
```

There is also a set of NLP-focused pipelines in
[`examples/nlp/`](https://github.com/TeeratP/pttai/tree/main/examples/nlp)
(document triage, extract-and-summarize, and a full RAG QA pipeline), plus
[`examples/parallel_usage.py`](https://github.com/TeeratP/pttai/blob/main/examples/parallel_usage.py),
an offline tour of parallelism, map-reduce, typed IO, and validation in one
script.
