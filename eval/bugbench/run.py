"""Score build-time dataflow-bug detection: pttai's validator vs. raw LangGraph.

For every corpus item we record whether pttai flags it at BUILD time (its
validator raises at construction). For BUGGY items we also run the equivalent
raw-LangGraph graph and record where the bug surfaces (build / runtime-after-K
model calls / silent-never) and how many model calls it burns first. From this
we compute, per framework:

  * build-time CATCH RATE on the buggy set,
  * FALSE-POSITIVE RATE on the clean/valid set (target ~0),
  * a WASTED-COST table (LLM calls burned before the bug surfaces).

All numbers are measured offline with a counting fake LLM — never invented. If
the validator misses a buggy item or flags a clean one, it is reported as-is.

    PYTHONPATH=. .venv/bin/python eval/bugbench/run.py
"""

import csv
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from corpus import CORPUS, BUGGY, CLEAN  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def _build_pttai(build):
    """Run a pttai constructor; return (flagged, error_type, message, ms)."""
    t0 = time.perf_counter()
    try:
        build()
        return False, None, "", (time.perf_counter() - t0) * 1000
    except Exception as e:  # noqa: BLE001 — any build failure = the validator flagged it
        msg = str(e)
        # GraphValidationError renders a multi-line report; take the first [error] line.
        line = next((l.strip() for l in msg.splitlines() if "[error]" in l),
                    msg.splitlines()[0] if msg.splitlines() else msg)
        return True, type(e).__name__, line, (time.perf_counter() - t0) * 1000


def evaluate():
    rows = []
    for item in CORPUS:
        flagged, etype, msg, ms = _build_pttai(item.build_pttai)
        row = {
            "id": item.id,
            "bug_class": item.bug_class,
            "label": item.label,
            "category": item.category,
            "pttai_only": item.pttai_only,
            "pttai_flagged_at_build": flagged,
            "pttai_error_type": etype,
            "pttai_message": msg,
            "pttai_build_ms": round(ms, 3),
            "pttai_wasted_calls": 0 if flagged else None,
            "lg_phase": None,
            "lg_error_type": None,
            "lg_message": None,
            "lg_wasted_calls": None,
        }
        if item.langgraph is not None:               # buggy items only
            lg = item.langgraph()
            row["lg_phase"] = lg["phase"]
            row["lg_error_type"] = lg["error_type"]
            row["lg_message"] = lg["message"]
            row["lg_wasted_calls"] = lg["wasted_calls"]
        rows.append(row)
    return rows


def _rate(num, den):
    return (num / den) if den else 0.0


def summarize(rows):
    by_id = {r["id"]: r for r in rows}
    buggy = [by_id[i.id] for i in BUGGY]
    clean = [by_id[i.id] for i in CLEAN]

    # ---- catch rate on buggy (per class + overall), split by category ----
    per_class = defaultdict(lambda: {"n": 0, "pttai_caught": 0, "lg_build": 0,
                                     "lg_runtime": 0, "lg_silent": 0,
                                     "lg_wasted_total": 0, "category": None})
    for r in buggy:
        c = per_class[r["bug_class"]]
        c["n"] += 1
        c["category"] = r["category"]
        c["pttai_caught"] += int(r["pttai_flagged_at_build"])
        phase = r["lg_phase"]
        if phase == "build":
            c["lg_build"] += 1
        elif phase == "runtime":
            c["lg_runtime"] += 1
        elif phase == "silent":
            c["lg_silent"] += 1
        c["lg_wasted_total"] += r["lg_wasted_calls"] or 0

    n_buggy = len(buggy)
    pttai_caught = sum(r["pttai_flagged_at_build"] for r in buggy)
    lg_caught_build = sum(r["lg_phase"] == "build" for r in buggy)

    # Three disjoint subsets by category:
    #   differentiator — genuine pttai-only-at-build catches (the headline)
    #   both           — caught at build by BOTH frameworks (duplicate names)
    #   dsl-strictness — dead-end: NOT a LangGraph bug (legal implicit terminal),
    #                    pttai rejects it only per its own end_nodes requirement;
    #                    excluded from the differentiator headline + wasted total.
    diff = [r for r in buggy if r["category"] == "differentiator"]
    both = [r for r in buggy if r["category"] == "both"]
    strict = [r for r in buggy if r["category"] == "dsl-strictness"]

    def _subset(rows):
        return {
            "n": len(rows),
            "pttai_caught": sum(r["pttai_flagged_at_build"] for r in rows),
            "langgraph_build_catch": sum(r["lg_phase"] == "build" for r in rows),
            "langgraph_runtime": sum(r["lg_phase"] == "runtime" for r in rows),
            "langgraph_silent": sum(r["lg_phase"] == "silent" for r in rows),
            "langgraph_wasted_calls_total": sum(r["lg_wasted_calls"] or 0 for r in rows),
        }

    # ---- false positives on clean ----
    n_clean = len(clean)
    clean_flagged = [r for r in clean if r["pttai_flagged_at_build"]]

    return {
        "counts": {
            "buggy": n_buggy,
            "clean": n_clean,
            "bug_classes": len(per_class),
            "differentiator_classes": sorted({r["bug_class"] for r in diff}),
            "caught_by_both_classes": sorted({r["bug_class"] for r in both}),
            "dsl_strictness_classes": sorted({r["bug_class"] for r in strict}),
        },
        "buggy": {
            "pttai_build_catch_rate": _rate(pttai_caught, n_buggy),
            "pttai_caught": pttai_caught,
            "langgraph_build_catch": lg_caught_build,
        },
        # Kept as the primary headline subset. NOTE for downstream consumers: the
        # wasted-call figure is SIMULATED (offline fake LLM), worst-case-ordered.
        "pttai_only_subset": _subset(diff),
        "caught_by_both_subset": _subset(both),
        "dsl_strictness_subset": _subset(strict),
        "clean": {
            "pttai_false_positive_rate": _rate(len(clean_flagged), n_clean),
            "pttai_false_positives": len(clean_flagged),
            "false_positive_ids": [r["id"] for r in clean_flagged],
        },
        "per_class": {k: dict(v) for k, v in per_class.items()},
    }


def _emit(rows, summary):
    csv_path = os.path.join(HERE, "results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json_path = os.path.join(HERE, "results.json")
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "items": rows}, f, indent=2)
    return csv_path, json_path


def _print(rows, summary):
    print("=" * 90)
    print("Dataflow-bug benchmark — pttai validator (build time) vs. raw LangGraph")
    print(f"LLM backend: {'REAL OpenAI' if os.environ.get('OPENAI_API_KEY') else 'offline fake'}")
    print(f"Corpus: {summary['counts']['buggy']} buggy across "
          f"{summary['counts']['bug_classes']} classes, "
          f"{summary['counts']['clean']} clean/valid")
    print("=" * 90)

    # ---- per-item buggy detail ----
    print("\nBUGGY ITEMS")
    hdr = ("id", "bug class", "pttai@build", "LangGraph", "LG wasted calls")
    w = (8, 30, 12, 12, 16)

    def row(cols):
        return "  ".join(str(c).ljust(w[i]) for i, c in enumerate(cols))

    print(row(hdr)); print("-" * (sum(w) + 2 * (len(w) - 1)))
    for r in rows:
        if r["label"] != "buggy":
            continue
        print(row((r["id"], r["bug_class"],
                   "caught" if r["pttai_flagged_at_build"] else "MISSED",
                   r["lg_phase"], r["lg_wasted_calls"])))

    # ---- per-class catch-rate table ----
    print("\nCATCH RATE BY CLASS (buggy)")
    hdr = ("bug class", "category", "n", "pttai caught", "LG build", "LG runtime",
           "LG silent", "LG wasted")
    w = (26, 15, 3, 13, 9, 11, 10, 10)
    print(row(hdr)); print("-" * (sum(w) + 2 * (len(w) - 1)))
    _cat = {"differentiator": "pttai-only", "both": "BOTH", "dsl-strictness": "DSL-strict"}
    for cls, c in summary["per_class"].items():
        print(row((cls, _cat.get(c["category"], c["category"]), c["n"],
                   f'{c["pttai_caught"]}/{c["n"]}', c["lg_build"], c["lg_runtime"],
                   c["lg_silent"], c["lg_wasted_total"])))

    b = summary["buggy"]
    d = summary["pttai_only_subset"]
    st = summary["dsl_strictness_subset"]
    cl = summary["clean"]
    print("\n" + "=" * 90)
    print("HEADLINE")
    print("=" * 90)
    print(f"pttai build-time catch rate (ALL buggy):            "
          f"{b['pttai_caught']}/{summary['counts']['buggy']} = {b['pttai_build_catch_rate']:.0%}")
    print(f"  of which raw LangGraph also catches at build:     {b['langgraph_build_catch']} "
          f"(the duplicate-node-name class — caught by BOTH)")
    print(f"pttai-ONLY-at-build subset (differentiator):        "
          f"{d['pttai_caught']}/{d['n']} caught; LangGraph on the SAME set: "
          f"{d['langgraph_build_catch']} build / {d['langgraph_runtime']} runtime / "
          f"{d['langgraph_silent']} silent")
    print(f"pttai false-positive rate (clean/valid set):        "
          f"{cl['pttai_false_positives']}/{summary['counts']['clean']} = "
          f"{cl['pttai_false_positive_rate']:.0%}")
    if cl["pttai_false_positives"]:
        print(f"    !! false positives on: {cl['false_positive_ids']}")
    print(f"\nWASTED COST — SIMULATED (offline fake LLM), worst-case-ordered; NOT real-model cost.")
    print(f"  differentiator subset (raw LangGraph):  "
          f"{d['langgraph_wasted_calls_total']} model calls burned before the bug surfaces")
    print(f"  pttai on every differentiator item:     0 model calls (rejected at build, no invoke)")
    print(f"\nDSL-STRICTNESS subset (dead-end) — NOT a LangGraph bug, excluded from the headline:")
    print(f"  {st['n']} items; pttai build-rejects (its DSL requires declared end_nodes), "
          f"raw LangGraph compiles + runs to a legal implicit-terminal halt "
          f"({st['langgraph_silent']} silent, {st['langgraph_wasted_calls_total']} calls = normal work, not waste).")
    print(f"\ndifferentiator classes:      {summary['counts']['differentiator_classes']}")
    print(f"caught-by-both classes:      {summary['counts']['caught_by_both_classes']}")
    print(f"dsl-strictness classes:      {summary['counts']['dsl_strictness_classes']}")


def main():
    rows = evaluate()
    summary = summarize(rows)
    _print(rows, summary)
    csv_path, json_path = _emit(rows, summary)
    print(f"\nwrote {os.path.relpath(csv_path)} and {os.path.relpath(json_path)}")


if __name__ == "__main__":
    main()
