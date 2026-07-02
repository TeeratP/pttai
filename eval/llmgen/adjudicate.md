# Adjudicating the flags — true bug vs. false positive

`score.py` copies every flagged pipeline (plus its error) into `flagged/`. The
headline claim of this study is that the validator's flags are **real bugs, not
noise** — i.e. the false-positive rate among flags is ~0. That claim only holds
if each flag is adjudicated. This file is the labeling protocol.

## The rule

A flag is a **true bug** if the pipeline, run as written, would genuinely fail
or misbehave because of the dataflow problem the validator names. It is a
**false positive** if the pipeline would actually run correctly and the
validator was wrong to reject it.

The validator's hard errors are *may-availability* facts (does any path produce
the key this node reads?), so most flags are decidable by inspection — you do
not need to run the model. Per class:

| bug class | true bug when… | how to confirm |
|---|---|---|
| `read-before-write` | the read key is produced only by a node that runs **later** (or never) | trace `.child` order: is a producer upstream of the reader? |
| `read-undeclared` | the key is read but no node writes it and it isn't an input | grep the pipeline for a `writes=`/`output_field=` of that key |
| `dangling-choice` | a `DecisionNode`/`ConditionNode` choice has no wired handler | is `decision["<choice>"] > x` present for every choice? |
| `dead-end-node` | a reachable non-end node has no outgoing edge | is the node in `end_nodes`, or does it have a `.child`? |
| `concurrent-write-no-reducer` | two parallel branches write the same reducer-less key | are both writers under one `fanout(...)`/`[a, b]`, writing the same key? |
| `prompt-placeholder-mismatch` | a `{name}` in a `node_prompt` has no matching scalar read | does `reads=[...]` contain every `{name}` in the prompt? |
| `duplicate-node-names` | two nodes share a `name=` | grep the `name=` values |

If the answer is "yes, the bug is real", label `true-bug`. Only label
`false-positive` if you can show the pipeline would have run fine.

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
