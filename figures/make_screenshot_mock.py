#!/usr/bin/env python3
"""Compose a FAITHFUL mock of the demo/app.py Gradio playground.

This is a *mock*, not a live headless capture (a real Gradio + browser capture
wasn't feasible in the offline build environment). Everything shown is REAL
output, not invented:

  - the DSL is demo/app.py's WORKING_EXAMPLE verbatim;
  - the "Compiled LangGraph" panel embeds `_compiled_graph.png`, which is
    `graph.compiled_graph.get_graph().draw_mermaid()` rendered by mermaid-cli
    (the exact string the demo renders client-side);
  - the "Validator" panel is the verbatim `graph.summary()` for that graph.

To drop in a real screenshot later: run `python demo/app.py`, load the working
example, click Build + Validate, and screenshot the page as
`figures/demo_screenshot.png`.

Usage:  pip install matplotlib ; python figures/make_screenshot_mock.py
Output: figures/demo_screenshot_mock.png
"""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = pathlib.Path(__file__).resolve().parent

# demo/app.py WORKING_EXAMPLE (verbatim)
DSL = '''\
# RAG QA pipeline: retrieve passages, rerank them,
# then answer grounded in them. Each node's scalar
# reads are produced by the node before it, so the
# validator confirms the dataflow and compiles clean.
retrieve = AgentNode(
    name="retrieve", llm=get_llm(),
    node_prompt="Retrieve passages relevant to: {question}",
    reads=["question"], writes={"passages": str},
)
rerank = AgentNode(
    name="rerank", llm=get_llm(),
    node_prompt="Rerank these passages: {passages}",
    reads=["passages"], writes={"context": str},
)
answer = AgentNode(
    name="answer", llm=get_llm(),
    node_prompt="Answer grounded ONLY in: {context}",
    reads=["context"], writes=["messages"],
)

retrieve > rerank > answer
graph = AgenticGraph(start_node=retrieve,
                     end_nodes={answer})'''

# verbatim graph.summary() for the working preset
SUMMARY = '''\
AgenticGraph 'graph'   state=AgenticStatePlus
initial: context, decision, log, messages,
         passages, question, token
------------------------------------------------
node      type       reads     writes
retrieve  AgentNode  question  log,passages
rerank    AgentNode  passages  context,log
answer    AgentNode  context   log,messages
------------------------------------------------
3 nodes · 0 errors · 0 warning(s)'''

PURPLE = "#f2f0ff"
INK = "#1f2430"


def box(ax, x, y, w, h, ec, fc, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.005",
                                ec=ec, fc=fc, lw=lw, mutation_scale=8,
                                transform=ax.transAxes, clip_on=False))


def main() -> None:
    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # window chrome
    box(ax, 0.01, 0.01, 0.98, 0.98, "#c9c6e0", "white", lw=1.6)
    # header
    box(ax, 0.01, 0.90, 0.98, 0.09, "#c9c6e0", PURPLE, lw=1.6)
    ax.text(0.03, 0.955, "pttai playground", fontsize=20, fontweight="bold", color=INK)
    ax.text(0.03, 0.918,
            "Paste a pttai >-DSL snippet, then Build + Validate — get the compiled "
            "LangGraph diagram and the build-time validator output. No API key needed.",
            fontsize=9.5, color="#555")

    # ---- LEFT: code editor ----
    ax.text(0.04, 0.878, "pttai DSL", fontsize=11, fontweight="bold", color=INK)
    box(ax, 0.03, 0.145, 0.44, 0.70, "#2b2b3a", "#282a36", lw=1.2)
    ax.text(0.045, 0.845, DSL, va="top", ha="left", family="monospace",
            fontsize=8.2, color="#f8f8f2")
    # buttons
    box(ax, 0.03, 0.075, 0.15, 0.05, "#8a83c9", "#efeefb", lw=1.2)
    ax.text(0.105, 0.10, "Load working example", ha="center", va="center", fontsize=8.2, color=INK)
    box(ax, 0.19, 0.075, 0.14, 0.05, "#e06666", "#fdeaea", lw=1.2)
    ax.text(0.26, 0.10, "Load BROKEN example", ha="center", va="center", fontsize=8.2, color="#a12")
    box(ax, 0.34, 0.075, 0.13, 0.05, "#6b5bd6", "#6b5bd6", lw=1.2)
    ax.text(0.405, 0.10, "Build + Validate", ha="center", va="center",
            fontsize=8.6, fontweight="bold", color="white")

    # ---- RIGHT top: compiled graph ----
    ax.text(0.52, 0.865, "Compiled LangGraph", fontsize=11, fontweight="bold", color=INK)
    box(ax, 0.51, 0.47, 0.47, 0.385, "#c9c6e0", "white", lw=1.2)
    img = mpimg.imread(HERE / "_compiled_graph.png")
    h, w = img.shape[0], img.shape[1]
    # fit image inside the panel while preserving aspect ratio
    panel = [0.535, 0.485, 0.42, 0.35]  # x, y, w, h in fig coords
    ar = w / h
    pw, ph = panel[2], panel[3] * (13 / 8)  # convert h to same units via fig aspect
    # simpler: place via inset axes and let imshow keep aspect
    iax = fig.add_axes([panel[0], panel[1], panel[2], panel[3]])
    iax.imshow(img)
    iax.axis("off")

    # ---- RIGHT bottom: validator ----
    ax.text(0.52, 0.44,
            "Validator: OK — 0 error(s), 0 warning(s)",
            fontsize=11, fontweight="bold", color="#1e7d34")
    box(ax, 0.51, 0.145, 0.47, 0.285, "#bfe6c9", "#f2fbf4", lw=1.3)
    ax.text(0.525, 0.415, SUMMARY, va="top", ha="left", family="monospace",
            fontsize=8.0, color="#14421f")

    # footer note
    ax.text(0.5, 0.045,
            "MOCK of demo/app.py — real compiled-graph render + real summary(); "
            "swap in a live screenshot when captured.",
            ha="center", fontsize=8.5, style="italic", color="#888")

    out = HERE / "demo_screenshot_mock.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out.name}")


if __name__ == "__main__":
    main()
