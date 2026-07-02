# Submission video — shot list (target ≤ 2:30)

Mandatory demo video for EMNLP 2026 System Demonstrations. Reviewers often watch
**muted**, so every point must be legible on screen: large fonts (≥ 24 pt for
captions), high-contrast, code zoomed so it is readable, and a burned-in caption
on every shot. No reliance on narration.

Narrative arc: **(1) pttai is a real agent framework** (a pipeline that actually
calls the model and produces output), **(2) miswired pipelines fail late and
expensively**, **(3) the money shot — pttai catches the bug at build with zero
model calls.** Establishing (1) first is deliberate: it forestalls the "this is
just a linter, not a framework" objection before the validator is shown.

Total budget: **150 s** (30 + 20 + 55 + 30 + 15).

---

## Shot 1 — pttai IS an agent framework (0:00–0:30, 30 s)

**On screen:** run the `parallelization` example live in a terminal (a real
multi-branch pipeline that calls the model on each branch and reduces the
results). Show the model calls happening and the real answer printed.

**Caption (persistent):**
> pttai wires LLM steps with `>` into a pipeline that runs on native LangGraph —
> streaming, tools, fan-out, map-reduce.

**Beats:**
- 0:00 title card: *"pttai — a declarative DSL over LangGraph, with a build-time
  dataflow validator."*
- 0:08 the parallelization pipeline runs; branches call the model concurrently.
- 0:22 the reduced final answer prints. Caption: *"A working agent, in a few
  lines. It compiles to a plain LangGraph `StateGraph`."*

## Shot 2 — The problem (0:30–0:50, 20 s)

**On screen:** a raw-LangGraph pipeline with one miswired state key. It makes the
model calls, then, several seconds in, crashes with a red traceback:
`KeyError: 'passages'`.

**Caption (persistent):**
> LLM pipelines fail *after* the model runs. A miswired state key = a runtime
> KeyError — tokens, latency, and tool side effects already spent.

**Beats:**
- 0:30 the graph runs, spinner on the model call.
- 0:42 the traceback lands. Freeze-frame + red box around `KeyError: 'passages'`.
- Caption sting: *"The bug existed at build time — but it cost model calls to
  find."*

## Shot 3 — The playground, the money shot (0:50–1:45, 55 s)

**On screen:** the browser playground (the paper's Figure 2). The playground
never calls a model — it builds, validates, and draws.

**Beats:**
- 0:50 paste a **working** RAG-QA `pttai` snippet. Submit. Green: the compiled
  LangGraph diagram renders + the `summary()` table (reads / writes / available
  keys per node). Caption: *"Clean build → the compiled LangGraph."*
- 1:15 edit **one** thing on camera: reorder so the answer node reads
  `passages` before the retriever runs. Submit.
- 1:25 the diagram redraws from the pre-compile wiring with the **offending node
  painted red** and the exact error attached:
  `reads computed key 'passages' but no upstream node produces it`.
  Caption (big): *"Rejected at BUILD time. Zero model calls."*
- 1:38 show the "model calls: 0" indicator; contrast callout to Shot 2's crash.
  Caption: *"Same bug. Caught before anything ran."*

Keep the red-node moment on screen ≥ 4 s — it is the whole demo.

## Shot 4 — The numbers (1:45–2:15, 30 s)

**On screen:** a single static results slide (readable, muted-friendly).

**Slide content (verbatim, matches the paper / `eval/bugbench/results.json`):**
- **17 / 17** buggy pipelines rejected at build time
- **12 / 12** differentiating bug cases caught at build; raw LangGraph catches
  **0** at build (all 12 surface only at runtime)
- **0** false positives on 19 valid pipelines
- **8 → 0** *simulated* wasted model calls on the differentiator subset (offline
  fake model, worst-case ordering — not a measured cost)
- Compiles to a native LangGraph `StateGraph`; build overhead within noise

Caption: *"Hard errors come only from the may-analysis — 0 false positives
observed on our valid corpus."*

> Do **not** put an LLM-generated-study number on this slide: the primary
> `eval/llmgen` study awaits a keyed run (paper §5). The bugbench figures above
> are the only headline numbers shown.

## Shot 5 — Install + link (2:15–2:30, 15 s)

**On screen:** terminal + repo URL.
- `pip install -e .` (or the published package name)
- `github.com/TeeratP/agentic-framework`  ·  live demo link
- Caption: *"Try the playground. MIT licensed."*

---

### Production notes
- Record the playground at 1080p, browser zoom ~125% so DSL + error are legible.
- All numbers on the results slide are the measured values in
  `eval/bugbench/results.json`. The dead-end class is **excluded** from the
  differentiator count (it is a pttai DSL-strictness choice, not a LangGraph
  bug), and the wasted-call figure is labeled *simulated*. The
  LLM-generation headline number is intentionally **not** shown — it awaits a
  keyed run (see paper §5).
