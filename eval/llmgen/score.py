"""Build every generated pttai pipeline and record the validator's verdict.

For each ``build_graph(llm)`` module in the target dir we CONSTRUCT the graph
(pttai's validator runs inside ``AgenticGraph.__init__``, so construction alone
triggers it -- NO API key and NO model call is needed to score) and bucket the
outcome:

  * ``clean``    -- built with no error (validator let it through);
  * ``flagged``  -- a validator/structural build error rejected it. We classify
                    the bug class from the error message and map it to the phase
                    raw LangGraph would surface it in (``runtime`` / ``silent`` /
                    ``build``), using the measured mapping from ``eval/bugbench``;
  * ``malformed``-- the module could not even be built for a NON-validator reason
                    (syntax error, bad import, wrong API name). These are
                    generation noise and are EXCLUDED from the false-positive
                    math -- counting them as "flags" would inflate the result.

Emits ``results.csv`` (per pipeline) and ``results.json`` (per pipeline +
summary), copies each flagged snippet + its error into ``flagged/`` for
adjudication, and prints the headline. If an ``adjudication.csv`` (columns:
``id,verdict,note`` where verdict is ``true-bug`` or ``false-positive``) sits in
this dir, its human labels drive the false-positive rate; otherwise a heuristic
labels every flag a true bug and the summary marks ``adjudicated: false``.

    PYTHONPATH=. .venv/bin/python eval/llmgen/score.py                 # scores generated/
    PYTHONPATH=. .venv/bin/python eval/llmgen/score.py --gen-dir samples   # offline proof
"""

import argparse
import csv
import importlib.util
import json
import os
import shutil
import sys
import traceback
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)                       # -> import pttai
sys.path.insert(0, os.path.join(ROOT, "examples"))  # -> from _llm import get_llm

from _llm import get_llm  # noqa: E402  (offline fake unless OPENAI_API_KEY is set)

# ---------------------------------------------------------------------------
# Flag classifier.
#
# Each entry: (bug_class, langgraph_phase, [message substrings]). ``langgraph_phase``
# is where raw LangGraph surfaces the SAME bug, per the measured findings in
# eval/bugbench/README.md:
#   runtime -> LangGraph raises only at invoke() (KeyError / InvalidUpdateError);
#   silent  -> LangGraph never errors (wrong/empty output, or a dropped write);
#   build   -> LangGraph ALSO catches it at construction (not a pttai differentiator).
#   dsl-strictness -> a pttai DSL-strictness rejection, NOT a LangGraph bug:
#             LangGraph runs the pipeline fine (e.g. a childless node is a legal
#             implicit terminal). Excluded from the pttai-only differentiator count.
# ---------------------------------------------------------------------------
_VALIDATION_CLASSES = [
    ("read-before-write", "runtime", ["reads computed key"]),
    ("read-undeclared", "runtime",
     ["is not declared in the state schema", "but no node produces it and it is not an input"]),
    ("concurrent-write-no-reducer", "runtime", ["concurrently writes"]),
    ("dangling-choice", "runtime", ["has no connected node"]),
    ("prompt-placeholder-mismatch", "runtime", ["references placeholder"]),
    ("write-undeclared", "silent", ["writes key", "silently drops unknown-key"]),
]
_STRUCTURAL_CLASSES = [  # non-GraphValidationError build failures pttai still rejects
    # dead-end is DSL-strictness, NOT a LangGraph bug: LangGraph treats a childless
    # node as a legal implicit terminal and runs fine, so this is not a pttai-only
    # differentiator (mirrors the eval/bugbench reclassification).
    ("dead-end-node", "dsl-strictness", ["has no children and is not an end node"]),
    ("duplicate-node-names", "build", ["already present"]),
]
# Exceptions that mean "the model wrote broken/non-pttai code", not a validator flag.
_MALFORMED_TYPES = {"SyntaxError", "ImportError", "ModuleNotFoundError",
                    "AttributeError", "NameError", "IndentationError"}


def _classify(exc_type: str, message: str):
    """-> (bucket, bug_class, langgraph_phase). bucket in {flagged, malformed}."""
    low = message.lower()
    if exc_type == "GraphValidationError":
        for cls, phase, needles in _VALIDATION_CLASSES:
            if any(n.lower() in low for n in needles):
                return "flagged", cls, phase
        return "flagged", "other-validation-error", "runtime"
    for cls, phase, needles in _STRUCTURAL_CLASSES:
        if any(n.lower() in low for n in needles):
            return "flagged", cls, phase
    if exc_type in _MALFORMED_TYPES:
        return "malformed", "generation-error", None
    # A TypeError/ValueError we don't recognize: treat as malformed (unknown API
    # misuse), not a validator flag -- honest default that never inflates flags.
    return "malformed", f"unrecognized/{exc_type}", None


def _load_build_fn(path):
    """Import a pipeline module and return its build_graph. May raise (import-time
    build failures are real outcomes we want to capture)."""
    name = "genpipe_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)          # top-level build (if any) runs here
    fn = getattr(module, "build_graph", None)
    if fn is None:
        raise AttributeError("module defines no build_graph(llm) function")
    return fn


def _score_one(path):
    row = {"id": os.path.splitext(os.path.basename(path))[0],
           "file": os.path.basename(path), "bucket": None, "bug_class": None,
           "langgraph_phase": None, "error_type": None, "message": ""}
    try:
        build = _load_build_fn(path)
        build(get_llm())                      # construct -> validator runs
        row["bucket"] = "clean"
        return row, None
    except Exception as e:                    # noqa: BLE001 -- any failure is an outcome
        etype = type(e).__name__
        full = "".join(traceback.format_exception_only(type(e), e)).strip()
        msg = str(e)
        # GraphValidationError renders a multi-line report; grab the first [error] line.
        line = next((l.strip() for l in msg.splitlines() if "[error]" in l),
                    msg.splitlines()[0] if msg.splitlines() else msg)
        bucket, cls, phase = _classify(etype, msg)
        row.update(bucket=bucket, bug_class=cls, langgraph_phase=phase,
                   error_type=etype, message=line)
        return row, full


def _save_flagged(path, row, full_error, flagged_dir):
    os.makedirs(flagged_dir, exist_ok=True)
    shutil.copy(path, os.path.join(flagged_dir, row["file"]))
    with open(os.path.join(flagged_dir, row["id"] + ".error.txt"), "w") as f:
        f.write(f"id: {row['id']}\nbug_class: {row['bug_class']}\n"
                f"langgraph_phase: {row['langgraph_phase']}\n"
                f"error_type: {row['error_type']}\n\n{full_error}\n")


def _load_adjudication():
    """Return {id: verdict} from adjudication.csv if present, else {}."""
    path = os.path.join(HERE, "adjudication.csv")
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            v = (r.get("verdict") or "").strip().lower()
            if v in ("true-bug", "false-positive"):
                out[r["id"].strip()] = v
    return out


def _rate(num, den):
    return (num / den) if den else 0.0


def summarize(rows, adjudication):
    total = len(rows)
    clean = [r for r in rows if r["bucket"] == "clean"]
    flagged = [r for r in rows if r["bucket"] == "flagged"]
    malformed = [r for r in rows if r["bucket"] == "malformed"]
    buildable = clean + flagged                      # excludes generation noise

    by_class = Counter(r["bug_class"] for r in flagged)
    by_phase = Counter(r["langgraph_phase"] for r in flagged)

    # False positives among flags: a flag is a FALSE POSITIVE if adjudicated so.
    # Absent human labels, the heuristic labels every flag a true bug (the
    # validator's hard errors are may-availability facts: it flags a read of a
    # key nothing produces -> a genuine bug). Summary flags whether it's human-adjudicated.
    adjudicated = bool(adjudication)
    fps = [r for r in flagged if adjudication.get(r["id"]) == "false-positive"]
    true_bugs = [r for r in flagged if r["id"] not in {x["id"] for x in fps}]

    # LangGraph would only surface these at runtime / silently (not at build).
    lg_runtime = sum(r["langgraph_phase"] == "runtime" for r in flagged)
    lg_silent = sum(r["langgraph_phase"] == "silent" for r in flagged)
    lg_build = sum(r["langgraph_phase"] == "build" for r in flagged)
    lg_dsl = sum(r["langgraph_phase"] == "dsl-strictness" for r in flagged)

    return {
        "backend": "REAL OpenAI" if os.environ.get("OPENAI_API_KEY") else "offline fake",
        "adjudicated": adjudicated,
        "counts": {
            "total": total, "buildable": len(buildable), "clean": len(clean),
            "flagged": len(flagged), "malformed_excluded": len(malformed),
        },
        "flag_rate_over_buildable": _rate(len(flagged), len(buildable)),
        "false_positive_rate_among_flags": _rate(len(fps), len(flagged)),
        "false_positives": [r["id"] for r in fps],
        "true_bugs_among_flags": len(true_bugs),
        "langgraph_would": {
            "runtime_only": lg_runtime, "silent": lg_silent, "build_too": lg_build,
            "dsl_strictness": lg_dsl,
            "runtime_or_silent": lg_runtime + lg_silent,
        },
        "flags_by_class": dict(by_class),
        "flags_by_langgraph_phase": dict(by_phase),
    }


def _emit(rows, summary):
    csv_path = os.path.join(HERE, "results.csv")
    fields = ["id", "file", "bucket", "bug_class", "langgraph_phase",
              "error_type", "message"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r.get(k) for k in fields} for r in rows)
    json_path = os.path.join(HERE, "results.json")
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "items": rows}, f, indent=2)
    return csv_path, json_path


def _print(summary, gen_dir):
    c = summary["counts"]
    print("=" * 84)
    print("LLM-generated-pipeline study -- pttai validator as an AI-code guardrail")
    print(f"source: {os.path.relpath(gen_dir)}/    backend: {summary['backend']}")
    print("=" * 84)
    print(f"pipelines scored:            {c['total']}")
    print(f"  buildable (clean+flagged): {c['buildable']}")
    print(f"  clean (validator passed):  {c['clean']}")
    print(f"  FLAGGED by validator:      {c['flagged']}")
    print(f"  malformed (excluded):      {c['malformed_excluded']}  "
          f"(broken/non-pttai code -- not counted as flags)")
    print(f"\nflag rate over buildable:    {summary['flag_rate_over_buildable']:.0%}")
    adj = "human-adjudicated" if summary["adjudicated"] else "HEURISTIC (run adjudication)"
    print(f"false-positive rate among flags ({adj}): "
          f"{summary['false_positive_rate_among_flags']:.0%}  "
          f"({len(summary['false_positives'])}/{c['flagged']})")
    if summary["false_positives"]:
        print(f"    false positives: {summary['false_positives']}")
    lg = summary["langgraph_would"]
    print(f"\nof the {c['flagged']} flags, raw LangGraph would surface:")
    print(f"  only at runtime:   {lg['runtime_only']}")
    print(f"  silently (never):  {lg['silent']}")
    print(f"  also at build:     {lg['build_too']}  (not a pttai differentiator)")
    print(f"  DSL-strictness:    {lg.get('dsl_strictness', 0)}  "
          f"(LangGraph runs fine -- not a pttai differentiator)")
    print(f"  => runtime-or-silent (pttai-only early catch): {lg['runtime_or_silent']}")
    print(f"\nflags by class:           {summary['flags_by_class']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen-dir", default="generated",
                    help="dir of generated pipelines (relative to eval/llmgen/ or absolute)")
    args = ap.parse_args()

    gen_dir = args.gen_dir if os.path.isabs(args.gen_dir) else os.path.join(HERE, args.gen_dir)
    if not os.path.isdir(gen_dir):
        sys.exit(f"ERROR: no such directory: {gen_dir}\n"
                 f"Generate pipelines first (run gen.py with OPENAI_API_KEY), or "
                 f"point --gen-dir at the committed samples/.")

    files = sorted(f for f in os.listdir(gen_dir)
                   if f.endswith(".py") and not f.startswith("_"))
    if not files:
        sys.exit(f"ERROR: no .py pipelines in {gen_dir} (nothing to score).")

    flagged_dir = os.path.join(HERE, "flagged")
    if os.path.isdir(flagged_dir):
        shutil.rmtree(flagged_dir)

    rows = []
    for fname in files:
        path = os.path.join(gen_dir, fname)
        row, full_error = _score_one(path)
        if row["bucket"] == "flagged":
            _save_flagged(path, row, full_error, flagged_dir)
        rows.append(row)

    adjudication = _load_adjudication()
    summary = summarize(rows, adjudication)
    _print(summary, gen_dir)
    csv_path, json_path = _emit(rows, summary)
    print(f"\nwrote {os.path.relpath(csv_path)} and {os.path.relpath(json_path)}")
    if summary["counts"]["flagged"]:
        print(f"flagged snippets + errors copied to {os.path.relpath(flagged_dir)}/ "
              f"-> adjudicate with adjudicate.md")


if __name__ == "__main__":
    main()
