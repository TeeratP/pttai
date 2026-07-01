# Examples: Architectures

The [`examples/architectures/`](https://github.com/TeeratP/pttai/tree/main/examples/architectures)
gallery has the canonical agent patterns from Anthropic's
[Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
and the common LangGraph tutorials, each in **one runnable file**: the **pttai
version first**, then a `# --- equivalent in raw LangGraph ---` block, so you can
read the same topology built both ways.

## Running

Everything runs **offline** with no API key — the shared helper
(`examples/_llm.py`, imported as `get_llm()`) returns a scripted fake chat model
when `OPENAI_API_KEY` is unset. Set `OPENAI_API_KEY` and the exact same files
call **real OpenAI** (`gpt-5.4-nano`) — no code change.

```bash
python examples/architectures/react_agent.py               # one file
for f in examples/architectures/*.py; do python "$f"; done  # all of them
```

Every loop is **capped** so it terminates under the offline fake (which always
picks a model's first structured choice): the evaluator–optimizer and supervisor
count iterations in the reduced `log` channel via a `ConditionNode` and stop at a
fixed cap.

## Index

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

## ASCII diagrams

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
    ┌▶ [ optimist ]  ─┐
    [ frame ] ─┼▶ [ skeptic ]   ─┼▶ [ verdict ] ─▶ out
    └▶ [ pragmatist ]─┘   (join deferred until all three finish)

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
