#!/usr/bin/env bash
# Reproduce the LLM-generated-pipeline study end to end.
#
#   OPENAI_API_KEY=sk-...  bash eval/llmgen/run_study.sh [PER_SPEC]
#
# Step 1 (gen.py)   needs OPENAI_API_KEY — a frontier model writes N pttai
#                   pipelines from short NLP task specs (given only the docs).
# Step 2 (score.py) is offline — builds each pipeline and records the validator's
#                   verdict into results.{csv,json} + flagged/.
#
# PER_SPEC defaults to 5 (10 specs -> ~50 pipelines). Bump it for N≈100.
set -euo pipefail

PER_SPEC="${1:-5}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set. Export your key first, e.g.:" >&2
  echo "    export OPENAI_API_KEY=sk-..." >&2
  echo "(To verify only the scoring path offline, run:" >&2
  echo "    PYTHONPATH=$ROOT $PY eval/llmgen/score.py --gen-dir samples )" >&2
  exit 1
fi

cd "$ROOT"
echo "==> [1/2] Generating pipelines (per-spec=$PER_SPEC)…"
PYTHONPATH="$ROOT" "$PY" eval/llmgen/gen.py --per-spec "$PER_SPEC"

echo "==> [2/2] Scoring generated pipelines…"
PYTHONPATH="$ROOT" "$PY" eval/llmgen/score.py --gen-dir generated

echo
echo "Done. See eval/llmgen/results.{csv,json} and flagged/."
echo "Next: adjudicate the flags per eval/llmgen/adjudicate.md (write"
echo "adjudication.csv), then re-run score.py for the human-adjudicated FP rate."
