# `eval/llmgen/` — the validator as a guardrail for AI-generated agent code

An **externally-grounded** evaluation of pttai's build-time validator. A
frontier LLM generates N pttai pipelines from short, natural NLP task specs —
given only the docs, and **never told to write bugs** — then every generated
pipeline is built and the validator's verdict is recorded and adjudicated.

## Why this exists

`eval/bugbench/` measures a 100% catch rate on a corpus **we authored**. The
obvious objection is circularity: *of course* the validator catches the bugs its
author planted. This study removes the author from the loop. The bug
distribution is produced by a frontier model solving ordinary tasks, so whatever
the validator catches (or misses, or wrongly flags) is a fact about **real
AI-written agent code**, not about a hand-picked corpus. That reframes the
validator as its most 2026-relevant thing: **a static guardrail that rejects
broken AI-generated agent pipelines before they ever run.**

## How it works

1. **`gen.py`** — prompts a frontier model (via `examples/_llm.py` `get_llm()`,
   real `ChatOpenAI` when `OPENAI_API_KEY` is set) with the pttai docs
   (`docs/*.md`) + a short task spec (RAG QA, multi-hop, extract→summarize,
   triage, rerank, tool-use, reflect/revise, plan/execute, sentiment routing,
   map-reduce). Each reply is saved as a standalone module in `generated/`
   defining `build_graph(llm) -> AgenticGraph`.
2. **`score.py`** — builds every pipeline (pttai's validator runs inside
   `AgenticGraph.__init__`, so **construction alone scores it — no API key, no
   model call**) and buckets each outcome:
   - **clean** — validator passed;
   - **flagged** — a validator/structural build error rejected it (classified by
     bug class, and mapped to the phase raw LangGraph would surface the same bug
     in: `runtime` / `silent` / `build`, per the measured `eval/bugbench` findings);
   - **malformed** — the model emitted broken/non-pttai code (syntax/import/name
     error). **Excluded** from the false-positive math — counting generation
     noise as flags would inflate the result.
3. **`adjudicate.md`** — the protocol for labeling each flag `true-bug` vs
   `false-positive` (mostly decidable by inspection). Labels go in
   `adjudication.csv`; `score.py` reads it and reports the human-adjudicated
   false-positive rate.

## Reproduce (one command, needs your key)

```bash
export OPENAI_API_KEY=sk-...
bash eval/llmgen/run_study.sh            # gen (~50 pipelines) -> score
# bigger sample:  bash eval/llmgen/run_study.sh 10   # ~100 pipelines
```

Then adjudicate per [`adjudicate.md`](adjudicate.md) (write `adjudication.csv`)
and re-run scoring for the human-adjudicated FP rate:

```bash
PYTHONPATH=. .venv/bin/python eval/llmgen/score.py --gen-dir generated
```

Outputs: `results.csv` (per pipeline), `results.json` (per pipeline + summary),
and `flagged/` (each flagged snippet + its error, ready to adjudicate).

## Verify the scoring path offline (no key)

The scoring path is proven against a committed set of hand-placed sample
pipelines in [`samples/`](samples/) (4 clean + 5 buggy, one per bug class,
reused from `examples/` and the `eval/bugbench` corpus):

```bash
PYTHONPATH=. .venv/bin/python eval/llmgen/score.py --gen-dir samples
```

Measured offline result on the sample set (this is a **harness self-test**, not
the study): 9 scored → 4 clean, **5 flagged (0 false positives)**, 0 malformed;
of the 5 flags, raw LangGraph would surface 4 only at runtime, and the 5th
(dead-end) is a pttai **DSL-strictness** rejection LangGraph runs fine — not a
pttai-only differentiator. This is what the committed `results.{csv,json}`
currently contain.

## Headline results — populate by running `run_study.sh`

> **These numbers require your `OPENAI_API_KEY`.** The build environment has no
> key, so the study was **not** run here. Run `run_study.sh`, adjudicate, and
> fill this table from `results.json`. Do not copy the sample-set self-test
> numbers above into here — those are the harness check, not the study.

| metric | value |
|---|---|
| model / date | _populate via run_study.sh_ |
| pipelines generated (N) | _populate_ |
| buildable (clean + flagged) | _populate_ |
| malformed (excluded) | _populate_ |
| **% of buildable pipelines flagged** | _populate_ |
| **false-positive rate among flags (adjudicated)** | _populate — target ≈ 0_ |
| flags raw LangGraph would only hit at runtime | _populate_ |
| flags raw LangGraph would surface silently (never) | _populate_ |
| flags raw LangGraph also catches at build | _populate — not a differentiator_ |

### Honest framing

- The false-positive rate is only meaningful **after adjudication**
  (`adjudication.csv`). An unadjudicated run reports a heuristic 0% and marks the
  summary `adjudicated: false` — that is not the headline.
- **Malformed generations are excluded**, not counted as catches. The claim is
  about the validator's precision on *buildable* pipelines.
- The LangGraph phase per flag is a **lookup** from the measured `eval/bugbench`
  mapping (same bug classes), not a re-measurement per generated pipeline.
- No study numbers are fabricated. The table above stays a placeholder until a
  real keyed run fills it.
