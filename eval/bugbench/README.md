# `eval/bugbench/` — a labeled dataflow-bug benchmark for LLM agent graphs

A reproducible corpus of agent-graph pipelines, each labeled **buggy** (with a
known dataflow-bug class) or **clean/valid**, scored for **build-time bug
detection**. It answers the paper's "must report evaluation" with real metrics:
a build-time **catch rate**, a **false-positive rate**, and a **wasted-cost**
table — all measured offline with a counting fake LLM, never invented.

This extends `eval/validator_bugs.py` (the #26 four-class demo) into a proper
labeled corpus with multiple instances per class, a valid-pipeline control set,
and machine-readable output (`results.json`) for the paper's charts.

## Reproduce (one command, offline — no API key)

```bash
PYTHONPATH=. .venv/bin/python eval/bugbench/run.py
```

Writes `eval/bugbench/results.csv` (per-item) and `eval/bugbench/results.json`
(per-item + summary), and prints the tables below.

## Corpus

**34 items:** 15 buggy across 6 classes + 19 clean/valid (every runnable
`examples/architectures/*` and `examples/basics/*` pipeline — the false-positive
control).

| Bug class | instances | pttai-only at build? | what raw LangGraph does |
|---|:--:|:--:|---|
| `read-before-write` (reads a computed key before its producer runs) | 3 | **yes** | runtime `KeyError` (0–1 wasted calls) |
| `dangling-choice` (a decision/condition choice with no wired node) | 3 | **yes** | runtime `KeyError` on the omitted branch (1 wasted call) |
| `dead-end-node` (a non-end node with no outgoing edge) | 3 | **yes** | **silent** — graph just halts (2–3 wasted calls) |
| `concurrent-write-no-reducer` (parallel branches write one reducer-less key) | 2 | **yes** | runtime `InvalidUpdateError` (1 wasted call) |
| `prompt-placeholder-mismatch` (`{name}` in prompt, no matching scalar read) | 2 | **yes** | runtime `KeyError` when the prompt is formatted |
| `duplicate-node-names` | 2 | **NO — caught by both** | **build** `ValueError('Node \`x\` already present.')` |

### Honesty notes

- **`duplicate-node-names` is caught by BOTH frameworks at build.** Raw
  LangGraph's `add_node` raises `Node \`x\` already present.` It is included in
  the corpus for completeness but is **not** a pttai differentiator. (`docs/COMPARISON.md`
  previously claimed LangGraph silently merges duplicate names — that was wrong
  and has been corrected.)
- **There is no true value-*type* mismatch detection.** The validator checks
  state-key *presence / availability*, not Python value types. The closest
  firing check is `prompt-placeholder-mismatch` (a guaranteed runtime
  `KeyError`), included as its own class.
- The **clean/valid set is the real, shipped `examples/`** — the strongest
  false-positive control available.

## Results (measured, offline fake LLM)

### Catch rate by class

| bug class | pttai-only? | n | pttai caught | LG build | LG runtime | LG silent | LG wasted calls |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| read-before-write | yes | 3 | 3/3 | 0 | 3 | 0 | 2 |
| dangling-choice | yes | 3 | 3/3 | 0 | 3 | 0 | 3 |
| dead-end-node | yes | 3 | 3/3 | 0 | 0 | 3 | 8 |
| duplicate-node-names | NO (both) | 2 | 2/2 | 2 | 0 | 0 | 0 |
| concurrent-write-no-reducer | yes | 2 | 2/2 | 0 | 2 | 0 | 2 |
| prompt-placeholder-mismatch | yes | 2 | 2/2 | 0 | 2 | 0 | 0 |

### Headline

| metric | value |
|---|---|
| pttai build-time catch rate (all 15 buggy) | **15/15 = 100%** |
| …of which raw LangGraph also catches at build | 2 (the duplicate-name class — both) |
| **pttai-only-at-build differentiator subset** | **13/13 caught**; on the same 13, LangGraph = 0 build / 10 runtime / 3 silent |
| **pttai false-positive rate** (19 clean/valid) | **0/19 = 0%** |

### Wasted cost

| framework | LLM calls burned before the bug surfaces (15 buggy items) |
|---|---|
| raw LangGraph | **15 model calls** — and 3 items (`dead-end-node`) surface **silently**, never erroring |
| pttai | **0 model calls** — every buggy pipeline is rejected at build, before any `invoke()` |

The `dead-end-node` class is the sharpest illustration: raw LangGraph runs the
graph to a silent halt (no error, wrong/empty output) after wasting the most
calls; pttai refuses to build it.
