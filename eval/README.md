# `eval/` — reproducible evaluation harness

Every number the EMNLP 2026 demo paper cites is regenerated here. All scripts run
**offline** (no `OPENAI_API_KEY`): `examples/_llm.py` supplies a scripted fake
chat model, and model calls are counted with a wrapper so "wasted LLM calls" is a
real measurement, not an estimate. Numbers are counted/measured, never invented.

## Regenerate every number (one command)

From the repo root, with the project virtualenv:

```bash
PYTHONPATH=. .venv/bin/python eval/loc_compare.py    && \
PYTHONPATH=. .venv/bin/python eval/validator_bugs.py && \
PYTHONPATH=. .venv/bin/python eval/bugbench/run.py   && \
PYTHONPATH=. .venv/bin/python eval/overhead.py
```

## What each script produces

| Script | Output | Claim it backs |
|---|---|---|
| `loc_compare.py` | Markdown table + `loc_results.csv` | Lines-of-code / node-count reduction vs. raw LangGraph, across every `examples/architectures/*.py` and selected `examples/basics/*.py`. Reconciles the three numbers hardcoded in `docs/COMPARISON.md` (harness is source of truth). |
| `validator_bugs.py` | Summary table + verbatim errors | The headline demo: build-time dataflow bugs pttai's validator catches. One buggy graph per class; each pttai build is asserted to fail at construction (with timing + verbatim message), then the equivalent raw-LangGraph graph is run to show whether it fails at build, only at runtime (after ≥1 wasted model call), or silently. |
| `bugbench/run.py` | `bugbench/results.{csv,json}` + tables | The full **labeled benchmark** that extends `validator_bugs.py`: 17 buggy pipelines across 7 classes + 19 clean/valid (the real `examples/`). Reports build-time **catch rate** (17/17 all buggy; **12/12** on the pttai-only differentiator subset), **false-positive rate** (0/19), and a **simulated** wasted-cost figure (offline fake LLM, worst-case-ordered — 8 calls on the differentiator subset). Labels `duplicate-node-names` as caught by *both* frameworks and `dead-end-node` as pttai DSL strictness (a legal LangGraph implicit terminal, excluded from the differentiator numbers). See `bugbench/README.md`. |
| `overhead.py` | Console report | pttai's build time is negligible and it compiles to a native LangGraph `CompiledStateGraph` with runtime parity (same model-call count, identical output). |

## Method notes

- **LOC** = non-blank, non-comment physical source lines in the `pttai_version()`
  / `langgraph_version()` function body, counted with `ast` (a leading docstring
  and the `def` line are excluded). This rule reproduces `docs/COMPARISON.md`'s
  hand-cited numbers exactly.
- **Node count** — pttai counts node constructors
  (`AgentNode`/`DecisionNode`/`ConditionNode`/`HumanNode`); LangGraph counts
  `builder.add_node(...)` calls. Note pttai sometimes shows *more* nodes: a gate
  or router is a first-class node in pttai but is expressed as a conditional
  *edge* (no `add_node`) in raw LangGraph. LOC, not node count, is the reduction
  story.
- **Wasted LLM calls** — a `CountingLLM` wrapper increments a shared counter on
  every `.invoke` (including `bind_tools` / `with_structured_output` variants),
  so the count survives across a run and reflects real calls made before a raw
  LangGraph graph fails.
