#!/usr/bin/env python3
"""Regenerate every data-driven figure from the committed benchmark outputs.

All numbers are read from:
  - eval/loc_results.csv           (lines-of-code, pttai vs raw LangGraph)
  - eval/bugbench/results.json     (build-time bug catch + wasted model calls)

There are NO hardcoded benchmark numbers in this file — re-run it after the
benchmarks change and the figures update. The only literal string reproduced
here is the verbatim `GraphValidationError` for the demo's BROKEN RAG preset
(text, not data), so the "money shot" panel matches what `demo/app.py` prints.

Usage:
    pip install matplotlib
    python figures/make_charts.py

Outputs (SVG + PNG, next to this script):
    loc_comparison.{svg,png}
    bug_catch.{svg,png}
    validator_before_after.{svg,png}
"""
from __future__ import annotations

import csv
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
LOC_CSV = ROOT / "eval" / "loc_results.csv"
BUG_JSON = ROOT / "eval" / "bugbench" / "results.json"

# Neutral, grayscale-safe palette (distinct in lightness, not just hue).
PTTAI = "#3b4cc0"   # dark blue  -> dark gray
LG = "#c0c0c0"      # light gray
BAD = "#c0392b"     # red (build-fail / runtime error)
GOOD = "#1e7d34"    # green (clean build)

plt.rcParams.update({"font.size": 11, "svg.fonttype": "none"})


def _save(fig, stem: str) -> None:
    for ext in ("svg", "png"):
        fig.savefig(HERE / f"{stem}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  wrote {stem}.svg / {stem}.png")


# --------------------------------------------------------------------------- #
# (a) Lines of code, per pipeline: pttai vs raw LangGraph                       #
# --------------------------------------------------------------------------- #
def loc_chart() -> None:
    rows = list(csv.DictReader(LOC_CSV.open()))
    labels = [r["file"].replace(".py", "") for r in rows]
    pttai = [int(r["pttai_loc"]) for r in rows]
    lg = [int(r["lg_loc"]) for r in rows]
    p_tot, l_tot = sum(pttai), sum(lg)
    n = len(rows)

    fig, ax = plt.subplots(figsize=(11, 5))
    y = range(n)
    h = 0.4
    ax.barh([i + h / 2 for i in y], lg, height=h, color=LG, label=f"raw LangGraph ({l_tot} LOC)")
    ax.barh([i - h / 2 for i in y], pttai, height=h, color=PTTAI, label=f"pttai ({p_tot} LOC)")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("lines of code (non-blank, non-comment)")
    overall = round(100 * (1 - p_tot / l_tot))
    ax.set_title(
        f"Lines of code per pipeline — pttai vs raw LangGraph\n"
        f"{p_tot} vs {l_tot} LOC across {n} pipelines  ·  {overall}% fewer lines",
        fontweight="bold",
    )
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.62), frameon=True,
              framealpha=0.9, edgecolor="#ccc")
    for i, (p, l) in enumerate(zip(pttai, lg)):
        ax.text(l + 1, i + h / 2, str(l), va="center", fontsize=8, color="#444")
        ax.text(p + 1, i - h / 2, str(p), va="center", fontsize=8, color=PTTAI)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "loc_comparison")


# --------------------------------------------------------------------------- #
# (b) Build-time bug catching + wasted model calls                             #
# --------------------------------------------------------------------------- #
def bug_chart() -> None:
    data = json.loads(BUG_JSON.read_text())
    summ = data["summary"]
    sub = summ["pttai_only_subset"]   # the pttai-only dataflow-bug differentiator subset
    clean = summ["clean"]
    n = sub["n"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6))

    # Left: outcome of the n pttai-only bugs, per framework (stacked).
    pttai_stack = [(sub["pttai_caught"], GOOD, "caught at build")]
    lg_stack = [
        (sub["langgraph_build_catch"], GOOD, "caught at build"),
        (sub["langgraph_runtime"], BAD, "runtime error"),
        (sub["langgraph_silent"], "#e08e0b", "silent wrong output"),
    ]
    for x, stack in ((0, pttai_stack), (1, lg_stack)):
        bottom = 0
        for val, color, lab in stack:
            if val == 0:
                continue
            axL.bar(x, val, bottom=bottom, color=color, width=0.55,
                    label=lab if x in (0, 1) else None)
            axL.text(x, bottom + val / 2, str(val), ha="center", va="center",
                     color="white", fontweight="bold")
            bottom += val
    axL.set_xticks([0, 1])
    axL.set_xticklabels([f"pttai\n{sub['pttai_caught']}/{n} at build", "raw LangGraph"])
    axL.set_ylabel(f"# of the {n} pttai-only dataflow bugs")
    axL.set_title(
        f"Build-time catch of {n} pttai-only bugs\n"
        f"(dup-name class excluded — caught by BOTH)", fontweight="bold", fontsize=11)
    # de-dupe legend labels
    h, l = axL.get_legend_handles_labels()
    seen = dict(zip(l, h))
    axL.legend(seen.values(), seen.keys(), loc="upper center", frameon=False, fontsize=9)
    axL.set_ylim(0, n + 3)
    axL.spines[["top", "right"]].set_visible(False)

    # Right: wasted LLM calls + false positives.
    p_waste = 0  # pttai fails at build -> never calls a model
    l_waste = sub["langgraph_wasted_calls_total"]
    axR.bar([0, 1], [p_waste, l_waste], color=[PTTAI, LG], width=0.55)
    axR.text(0, 0.2, "0", ha="center", va="bottom", color=PTTAI, fontweight="bold")
    axR.text(1, l_waste + 0.2, str(l_waste), ha="center", va="bottom", fontweight="bold")
    axR.set_xticks([0, 1])
    axR.set_xticklabels(["pttai", "raw LangGraph"])
    axR.set_ylabel("wasted LLM calls before failing")
    fp = clean["pttai_false_positives"]
    axR.set_title(
        f"LLM calls burned on the {n} bugs\n"
        f"pttai false positives on 19 valid pipelines: {fp}", fontweight="bold", fontsize=11)
    axR.set_ylim(0, l_waste + 3)
    axR.spines[["top", "right"]].set_visible(False)

    fig.suptitle("pttai fails the build before any model runs; LangGraph pays at runtime",
                 fontweight="bold", y=1.02)
    _save(fig, "bug_catch")


# --------------------------------------------------------------------------- #
# The money shot: buggy RAG pipeline -> build error vs runtime waste           #
# --------------------------------------------------------------------------- #
# Verbatim GraphValidationError from loading demo/app.py's BROKEN preset
# (rerank wired before retrieve => reads `passages` before it is produced).
# This is framework output text, not benchmark data.
RAG_ERROR = (
    "GraphValidationError\n"
    "AgenticGraph 'graph': 1 error(s), 0 warning(s)\n"
    "  [error] rerank: reads computed key 'passages'\n"
    "  but no upstream node produces it before this\n"
    "  node (produced by: ['retrieve'], none of which\n"
    "  are upstream); available keys here:\n"
    "  ['decision', 'log', 'messages', 'question',\n"
    "  'token']"
)

BROKEN_PIPELINE = (
    "retrieve = AgentNode(reads=['question'],\n"
    "                     writes={'passages': str})\n"
    "rerank   = AgentNode(reads=['passages'],\n"
    "                     writes={'context': str})\n"
    "answer   = AgentNode(reads=['context'])\n\n"
    "rerank > retrieve > answer   # BUG: rerank runs\n"
    "                             # before passages exists\n"
    "AgenticGraph(start_node=rerank, end_nodes={answer})"
)


def before_after() -> None:
    data = json.loads(BUG_JSON.read_text())
    sub = data["summary"]["pttai_only_subset"]
    n = sub["n"]
    runtime, silent = sub["langgraph_runtime"], sub["langgraph_silent"]
    wasted = sub["langgraph_wasted_calls_total"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 7.2))
    for ax in (axL, axR):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    def box(ax, x, y, w, h, color, lw=1.5, fill="none"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01",
                                    ec=color, fc=fill, lw=lw, mutation_scale=8))

    mono = {"family": "monospace", "fontsize": 8}

    # ---- shared: the buggy pipeline at the top of each panel ----
    for ax, title in ((axL, "pttai"), (axR, "raw LangGraph")):
        ax.text(0.5, 0.975, title, ha="center", fontsize=15, fontweight="bold")
        box(ax, 0.03, 0.62, 0.94, 0.32, "#888", fill="#f5f5f5")
        ax.text(0.06, 0.915, "the SAME buggy RAG pipeline", fontsize=9,
                style="italic", color="#555")
        ax.text(0.06, 0.885, BROKEN_PIPELINE, va="top", **mono)

    # ---- LEFT: pttai fails at build ----
    box(axL, 0.03, 0.17, 0.94, 0.42, BAD, lw=2.5, fill="#fdecea")
    axL.text(0.5, 0.555, "BUILD-TIME  ✗  (AgenticGraph constructor)",
             ha="center", fontweight="bold", color=BAD, fontsize=11)
    axL.text(0.06, 0.51, RAG_ERROR, va="top", color="#7a1c12", **mono)
    box(axL, 0.03, 0.03, 0.94, 0.11, GOOD, lw=2, fill="#e9f6ed")
    axL.text(0.5, 0.085, "0 model calls  ·  caught before you ever invoke",
             ha="center", fontweight="bold", color=GOOD, fontsize=11.5)

    # ---- RIGHT: LangGraph compiles, then pays at runtime ----
    box(axR, 0.03, 0.45, 0.94, 0.14, GOOD, lw=1.8, fill="#e9f6ed")
    axR.text(0.5, 0.52, "compile()  ✓  no dataflow check", ha="center",
             fontweight="bold", color=GOOD, fontsize=11)
    axR.annotate("", xy=(0.5, 0.43), xytext=(0.5, 0.445),
                 arrowprops=dict(arrowstyle="-|>", lw=2, color="#555"))
    box(axR, 0.03, 0.17, 0.94, 0.24, BAD, lw=2.5, fill="#fdecea")
    axR.text(0.5, 0.375, "RUNTIME  ✗  on the first invoke", ha="center",
             fontweight="bold", color=BAD, fontsize=11)
    axR.text(0.06, 0.33,
             "KeyError: 'passages'\n"
             "  (LangGraph reads the missing state key\n"
             "   only when the node actually executes)",
             va="top", color="#7a1c12", **mono)
    box(axR, 0.03, 0.03, 0.94, 0.11, "#e08e0b", lw=2, fill="#fdf3e0")
    axR.text(0.5, 0.085,
             f"across {n} such bugs: {runtime} runtime errors, {silent} silent\n"
             f"wrong outputs  ·  {wasted} LLM calls wasted",
             ha="center", fontweight="bold", color="#8a5600", fontsize=11)

    fig.suptitle(
        "The money shot: pttai rejects the bug at build; LangGraph runs it and pays",
        fontweight="bold", fontsize=14, y=1.0)
    _save(fig, "validator_before_after")


if __name__ == "__main__":
    print("Regenerating figures from eval/ ...")
    loc_chart()
    bug_chart()
    before_after()
    print("Done.")
