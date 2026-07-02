# Adjudicating the flags — true bug vs. false positive

`score.py` copies every flagged pipeline (plus its error) into `flagged/`. The
headline claim of this study is that the validator's flags are **real bugs, not
noise** — i.e. the false-positive rate among flags is ~0. That claim only holds
if each flag is adjudicated. This file is the labeling protocol.

## The rule — avoid circularity

The trap: if "true bug" just restates the flag condition (e.g. dead-end true-bug =
"a reachable non-end node has no outgoing edge" = *the flag itself*), then a false
positive is definitionally impossible and the 0% FP rate is a **tautology**. This
protocol avoids that by judging each flag against **raw LangGraph semantics**, not
against the validator's own rule.

Sort every flag into exactly one of three outcomes:

1. **True differentiator bug** — the pipeline, run as written, **genuinely fails or
   misbehaves under LangGraph semantics too**: a real runtime error (`KeyError`,
   `InvalidUpdateError`, a routing `KeyError`) or a wrong/empty result. This is the
   only outcome that counts toward the differentiator claim. Confirm it by asking
   "would raw LangGraph, running this same dataflow, error or produce the wrong
   answer?" — **not** "does it violate a pttai rule?"
2. **pttai DSL-strictness (report SEPARATELY, NOT a differentiator)** — pttai
   rejects it but **LangGraph runs it fine**. The canonical case is `dead-end-node`:
   LangGraph treats a childless node as a legal implicit terminal and runs to a
   normal halt. These are correct pttai rejections but they are strictness choices,
   not bugs LangGraph would hit — they must be tallied on their own and excluded
   from the differentiator count.
3. **False positive** — pttai rejects a pipeline that would have **run correctly
   and produced the right result** under pttai's own intended semantics; the
   validator was simply wrong to reject it.

A fourth bucket, **pttai-DSL-internal**: `prompt-placeholder-mismatch` is a pttai
templating error — a `{name}` in a `node_prompt` with no matching scalar `reads=`.
It **cannot occur in raw LangGraph** (LangGraph has no `node_prompt` templating),
so it is neither a shared-state dataflow catch nor a LangGraph-runtime bug. Report
it **separately** from the shared-state dataflow catches (`read-before-write`,
`concurrent-write-no-reducer`, `read-undeclared`, `dangling-choice`) and do not
fold it into the differentiator dataflow total.

Most flags are decidable by inspection (no model run). Per class, judge against
**LangGraph behavior**:

| bug class | LangGraph would… | outcome |
|---|---|---|
| `read-before-write` | raise `KeyError` (or read a stale/empty value) when the reader runs before any producer | differentiator true-bug **if** raw LangGraph errors/misbehaves; trace `.child` order to confirm the producer is not upstream |
| `read-undeclared` | raise `KeyError` / read nothing — no node writes the key and it isn't an input | differentiator true-bug if LangGraph would error/misbehave; grep for a `writes=`/`output_field=` of that key |
| `dangling-choice` | route to a destination not in the edge map → routing `KeyError` | differentiator true-bug if the unwired choice is reachable; check `decision["<choice>"] > x` for every choice |
| `concurrent-write-no-reducer` | raise `InvalidUpdateError` — parallel branches write one reducer-less key in a step | differentiator true-bug if both writers are under one `fanout(...)`/`[a, b]` writing the same plain key |
| `prompt-placeholder-mismatch` | **N/A** — cannot occur in raw LangGraph | pttai-DSL-internal; report separately, not a differentiator |
| `dead-end-node` | run fine (childless node = legal implicit terminal) | pttai DSL-strictness; report separately, not a differentiator |
| `duplicate-node-names` | **also** raise `ValueError('… already present')` at build | caught by both; not a differentiator |

Label `true-bug` only when the pipeline genuinely fails/misbehaves under LangGraph
semantics (outcome 1). Label `false-positive` only when you can show it would have
run fine and produced the right result (outcome 3). DSL-strictness and
DSL-internal flags are recorded in the `note` (and their bug class) so they are
visibly excluded from the differentiator tally — do **not** mark them `true-bug`.

**Second rater recommended.** For the judgment-bearing flags (anything where the
LangGraph verdict is not mechanical), a **second, non-author rater** should label
independently; disagreements are the honest signal about how decidable the flags
really are.

## How to record labels

Create `eval/llmgen/adjudication.csv` with one row per flagged id (ids are in
`results.csv` / the `flagged/` filenames):

```csv
id,verdict,note
rag_qa_03,true-bug,reads `passages` but no node writes it
triage_01,true-bug,`other` choice never wired
multi_hop_02,false-positive,key IS produced by the map node — validator missed it
```

`verdict` must be `true-bug` or `false-positive`. Re-run `score.py`; it picks up
`adjudication.csv` automatically and reports the human-adjudicated
false-positive rate (the summary flips `adjudicated: true`). Without this file,
`score.py` falls back to a heuristic that labels every flag a true bug and marks
the summary `adjudicated: false` — so an unadjudicated 0% FP rate is **not** the
headline number; the adjudicated one is.
