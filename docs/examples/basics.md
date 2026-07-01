# Examples: Basics

The [`examples/basics/`](https://github.com/TeeratP/pttai/tree/main/examples/basics)
gallery has **one feature per file**. Each file shows the **pttai version first**,
then a runnable `# --- equivalent in raw LangGraph ---` block, so you can see the
same behavior built both ways side by side — the fastest way to see exactly what
plumbing pttai folds away.

## Running

Everything runs **offline** with no API key — the shared helper
(`examples/_llm.py`, imported as `get_llm()`) returns a scripted fake chat model
when `OPENAI_API_KEY` is unset. Set `OPENAI_API_KEY` and the exact same examples
call **real OpenAI** (`gpt-5.4-nano`) instead — no code change.

```bash
python examples/basics/01_single_agent.py                  # one file
for f in examples/basics/[0-9]*.py; do python "$f"; done    # all of them
```

## Index

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

None of the examples require an API key; real OpenAI is used automatically if
`OPENAI_API_KEY` is set. Once you know the primitives, reach for the
[Architectures gallery](architectures.md) to lift whole topologies.
