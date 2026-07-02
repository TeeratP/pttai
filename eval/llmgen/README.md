# `eval/llmgen/` — the validator as a guardrail for AI-generated agent code

An **externally-grounded** evaluation of pttai's build-time validator. A small
OpenAI model (`gpt-5.4-nano` by default, overridable via `--model`) generates N
pttai pipelines from short, natural NLP task specs — given only the docs, and
**never told to write bugs** — then every generated pipeline is built and the
validator's verdict is recorded and adjudicated.

## Why this exists

`eval/bugbench/` measures a 100% catch rate on a corpus **we authored**. The
obvious objection is circularity: *of course* the validator catches the bugs its
author planted. This study removes the author from the loop. The bug
distribution is produced by a model solving ordinary tasks, so whatever
the validator catches (or misses, or wrongly flags) is a fact about **real
AI-written agent code**, not about a hand-picked corpus. That reframes the
validator as its most 2026-relevant thing: **a static guardrail that rejects
broken AI-generated agent pipelines before they ever run.**

## How it works

1. **`gen.py`** — prompts a small OpenAI model (`gpt-5.4-nano` by default via
   `examples/_llm.py` `get_llm()`, real `ChatOpenAI` when `OPENAI_API_KEY` is
   set; override with `--model`) with a compact **API
   cheatsheet** + the pttai prose docs (`docs/*.md`) + one short, natural task
   spec drawn from the **20-spec set** (see [What the model is given](#what-the-model-is-given)).
   Each reply is saved as a standalone module in `generated/` defining
   `build_graph(llm) -> AgenticGraph`.
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

## What the model is given

To keep the study honest, this is disclosed in full — including the fact that the
cheatsheet **coaches correct pttai usage** (see the honest disclosure below). We do
**not** claim the setup is non-circular or bug-neutral.

**It IS given:**

- **An accurate, compact API cheatsheet** (`API_CHEATSHEET` in `gen.py`): the real
  top-level exports, each node type's real constructor signature and accepted
  kwargs (e.g. `DecisionNode` takes `choices=` and does **not** take `writes`;
  `ConditionNode`/`HumanNode` take no `llm`/`tools`), the `>` wiring rules
  (`router["choice"] > node`, `fanout(...)`, `.map(...)`), `reads`/`writes`
  semantics, and the instruction to use the injected `llm`. Every line is verified
  against the pttai source.

  **Honest disclosure — the cheatsheet coaches valid usage.** It states core API
  and wiring rules, several of which map **directly onto validator error classes**:
  "a scalar key you read must be produced by an upstream node or seeded at invoke"
  (read-before-write), "wire EACH choice by name" (dangling-choice), and
  "`tools=[...]` CANNOT be combined with multi-field structured writes" (a build
  error). So the cheatsheet does reduce some error classes — telling the model the
  real API is legitimate, but we do **not** pretend this is a clean-room test of
  the model's unaided memory of pttai. What we do claim: the model is never told to
  write bugs, never handed a state key or a graph, and never steered toward any
  particular bug class; whatever bugs remain are the model's.
- **The full prose docs** (`docs/*.md`), as before.
- **One short, natural task spec** per generation, from the 20-spec set below.

**It is NOT given (deliberately):**

- **No complete, bug-free, copyable pipelines.** The cheatsheet is API surface +
  1–2 line snippets only. Handing over whole validated pipelines would suppress
  the natural dataflow mistakes the study exists to observe.
- **No mention of bugs, of any state key to use, or of the validator.** The specs
  are goals, never graphs. The model is **never told to introduce bugs** — every
  bug that appears is model-produced.

**The 20 task specs** (terse goals; grouped only for readability — the model sees
one at a time with no group label):

| id | goal |
|---|---|
| `rag_qa` | retrieve passages with a search tool, answer from them only |
| `multi_hop` | two sequential lookups: find an intermediate fact, then the final answer |
| `extract_summarize` | extract entities/claims, then summarize using them |
| `rerank` | retrieve candidates, rerank by relevance, answer from the top one |
| `plan_execute` | plan a research question, execute step by step, synthesize |
| `translate_glossary` | detect language, translate to English, build a term glossary |
| `fact_check` | gather supporting + contradicting evidence, issue a verdict |
| `dialogue_next_turn` | infer requested action + missing details, write the next turn |
| `triage` | classify a support message (billing/technical/other), route to a handler |
| `sentiment_route` | classify review sentiment, route to a matching responder |
| `intent_router` | classify a chat message into one of several intents, route it |
| `change_review_route` | decide bug-fix/feature/refactor, route to a fitting reviewer |
| `escalation` | judge severity, then auto-resolve / clarify / escalate to a human |
| `map_reduce_summ` | summarize several docs in parallel, reduce to one summary |
| `aspect_review` | review writing on clarity/grammar/tone in parallel, merge notes |
| `entity_dedup` | extract entities from docs in parallel, reconcile + deduplicate |
| `reflect_revise` | draft, critique, then revise using the critique |
| `refine_until_good` | draft, score, loop refining until good enough or rounds run out |
| `tool_use_math` | calculator agent: call add/multiply tools in a loop to the answer |
| `interview_grader` | grade an answer on rubric dimensions, give a hire recommendation |

The set spans chains, multi-branch routing, parallel fan-out/join, map-reduce,
reflection/evaluator loops, tool use, and human-in-the-loop — so model mistakes
can land across many bug classes rather than concentrating in one.

## Harness revision (pre-registered vs revised — disclosed)

The current harness is **not** the one we first ran, and we disclose the change so
a reader knows the design was revised after seeing results:

- **v1 (initial):** docs only, **no** API cheatsheet, **10** task specs. It
  produced **few catches** — most generations were malformed (API hallucinations)
  rather than buildable-but-buggy, which is generation noise, not the dataflow
  signal the study is about.
- **v2 (revised, current):** an **accurate API cheatsheet** was added (so the
  model stops hallucinating the API and instead emits buildable pipelines whose
  bugs are dataflow bugs), and the spec set was **broadened to 20** shapes.

This revision happened **after** seeing v1's results, so treat v2 as a *revised*
design, not the pre-registered one. The cheatsheet coaches correct usage (see the
honest disclosure above), which is exactly why we surface this: the strengthened
harness was not the original plan.

## Reproduce (one command, needs your key)

```bash
export OPENAI_API_KEY=sk-...
bash eval/llmgen/run_study.sh            # gen (20 specs x 5 = ~100 pipelines) -> score
# bigger sample:  bash eval/llmgen/run_study.sh 10   # ~200 pipelines
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
  (`adjudication.csv`). The summary is marked `adjudicated: true` **only when
  every flagged id has an explicit verdict** (an `adjudication_coverage` field
  reports the fraction labeled); a partial file keeps `adjudicated: false`. An
  unadjudicated (or partially adjudicated) run reports a heuristic 0% — that is
  not the headline.
- **Malformed generations are excluded**, not counted as catches. The claim is
  about the validator's precision on *buildable* pipelines.
- The LangGraph phase per flag is a **lookup** from the measured `eval/bugbench`
  mapping (same bug classes), not a re-measurement per generated pipeline.
- No study numbers are fabricated. The table above stays a placeholder until a
  real keyed run fills it.
