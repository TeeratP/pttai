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

**36 items:** 17 buggy across 7 classes + 19 clean/valid (every runnable
`examples/architectures/*` and `examples/basics/*` pipeline — the false-positive
control). Each buggy item has a **category**: `differentiator` (pttai-only at
build), `both` (caught at build by both frameworks), or `dsl-strictness` (a pttai
DSL requirement that is **not** a LangGraph bug).

| Bug class | instances | category | what raw LangGraph does |
|---|:--:|:--:|---|
| `read-before-write` (reads a computed key before its producer runs) | 3 | **differentiator** | runtime `KeyError` (0–1 wasted calls) |
| `cyclic-read-before-write` (reads a key produced only by a downstream node that loops back) | 2 | **differentiator** | runtime `KeyError` on the first iteration (0–1 wasted calls) |
| `dangling-choice` (a decision/condition choice with no wired node) | 3 | **differentiator** | runtime `KeyError` on the omitted branch (1 wasted call) |
| `concurrent-write-no-reducer` (parallel branches write one reducer-less key) | 2 | **differentiator** | runtime `InvalidUpdateError` (1 wasted call) |
| `prompt-placeholder-mismatch` (`{name}` in prompt, no matching scalar read) | 2 | **differentiator** | runtime `KeyError` when the prompt is formatted |
| `duplicate-node-names` | 2 | **both** | **build** `ValueError('Node \`x\` already present.')` |
| `dead-end-node` (a non-end node with no outgoing edge) | 3 | **dsl-strictness** | **compiles + runs** to a legal implicit-terminal halt (see note) |

### Honesty notes

- **`dead-end-node` is NOT a LangGraph bug.** A node with no outgoing edge is a
  *legal implicit terminal* in LangGraph: `compile()` does **not** raise and
  `invoke()` returns without error (verified empirically on LangGraph 1.2.x). The
  benchmark classifies it `dsl-strictness` because pttai rejects it only per its
  own rule that every terminal must be declared in `end_nodes` — not because
  LangGraph did anything wrong. It is therefore **excluded** from the
  differentiator headline and the wasted-call total. The `silent` phase and the
  8 "calls" it records are just the graph running its two nodes to completion —
  normal work, not waste.
- **`cyclic-read-before-write` is a genuine pttai-only-at-build catch** (issue
  #34): a cycle where a node reads a key produced **only** by a downstream node
  that loops back. On the first iteration the key is absent, so raw LangGraph
  `KeyError`s at runtime; pttai's back-edge-aware availability fixpoint rejects
  it at build.
- **`duplicate-node-names` is caught by BOTH frameworks at build.** Raw
  LangGraph's `add_node` raises `Node \`x\` already present.` It is included for
  completeness but is **not** a pttai differentiator.
- **There is no true value-*type* mismatch detection.** The validator checks
  state-key *presence / availability*, not Python value types. The closest
  firing check is `prompt-placeholder-mismatch` (a guaranteed runtime
  `KeyError`), included as its own class.
- The **clean/valid set is the real, shipped `examples/`** — the strongest
  false-positive control available.
- **The "wasted calls" figure is SIMULATED, not real-model cost.** Every
  LangGraph baseline runs against a counting *fake* LLM (offline), and the
  dangling-choice baselines deterministically hit the omitted branch because the
  fake `with_structured_output` picks the first `Literal` value and the corpus
  orders the unwired choice first — a **worst-case ordering**, disclosed here, not
  an observation of a real model.

## Results (measured, offline fake LLM)

### Catch rate by class

| bug class | category | n | pttai caught | LG build | LG runtime | LG silent | LG wasted calls |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| read-before-write | differentiator | 3 | 3/3 | 0 | 3 | 0 | 2 |
| cyclic-read-before-write | differentiator | 2 | 2/2 | 0 | 2 | 0 | 1 |
| dangling-choice | differentiator | 3 | 3/3 | 0 | 3 | 0 | 3 |
| concurrent-write-no-reducer | differentiator | 2 | 2/2 | 0 | 2 | 0 | 2 |
| prompt-placeholder-mismatch | differentiator | 2 | 2/2 | 0 | 2 | 0 | 0 |
| duplicate-node-names | both | 2 | 2/2 | 2 | 0 | 0 | 0 |
| dead-end-node | dsl-strictness | 3 | 3/3 | 0 | 0 | 3 | 8¹ |

¹ Not waste: LangGraph compiles and runs the dead-end graph to a normal halt.

### Headline

| metric | value |
|---|---|
| pttai build-time catch rate (all 17 buggy) | **17/17 = 100%** |
| …of which raw LangGraph also catches at build | 2 (the duplicate-name class — both) |
| **pttai-only-at-build differentiator subset** | **12/12 caught**; on the same 12, LangGraph = 0 build / 12 runtime / 0 silent |
| **pttai false-positive rate** (19 clean/valid) | **0/19 = 0%** |

### Wasted cost (simulated, offline fake LLM — worst-case-ordered)

| framework | LLM calls burned before the bug surfaces (12 differentiator items) |
|---|---|
| raw LangGraph | **8 simulated model calls** (all 12 surface at runtime) |
| pttai | **0 model calls** — every buggy pipeline is rejected at build, before any `invoke()` |

This is a simulated worst case, not measured real-model cost (see the honesty
note on the fake LLM above). The `dead-end-node` class is deliberately **not**
counted here — it is legal LangGraph behavior, not a defect.
