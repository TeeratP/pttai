---
title: pttai playground
emoji: 🕸️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: mit
short_description: Build a pttai >-DSL agent graph; see it compiled + validated.
---

# pttai playground (interactive demo)

A one-command playground for the pttai `>`-DSL. Paste a snippet, click
**Build + Validate**, and see:

1. **The graph** — rendered as a mermaid node/edge diagram. On a clean build
   this is the compiled LangGraph (`graph.compiled_graph.get_graph().draw_mermaid()`).
   When the build **fails**, the diagram is instead built from the pre-compile
   `>`-wiring with the **offending node painted red** and the error attached —
   the *picture of the bug*, before the graph is ever invoked. (mermaid.js is
   loaded from a CDN and renders client-side; offline, the raw diagram source is
   shown instead.)
2. **The build-time dataflow validator** — the `graph.summary()` table (each
   node's reads / writes / available keys) plus every `ValidationReport` issue.

**No `OPENAI_API_KEY` needed** — building, validating, and visualizing a graph
never call a model, and `get_llm()` returns an offline fake so the nodes
construct.

## Run locally

From a clone of the repo:

```bash
pip install -r demo/requirements.txt
python demo/app.py
```

Then open the printed local URL. (Run from the repo so `import pttai` and the
offline `get_llm()` resolve from source.)

## Preset gallery

- **Working presets** (one per `examples/nlp/` pipeline): **RAG QA**
  (`retrieve > rerank > answer`), **Extract → Summarize** (typed structured
  output), **Doc triage** (decision routing). Each compiles clean with a green
  validator.
- **Broken presets** (one per bug class in `eval/bugbench/corpus.py`):
  read-before-write, dangling decision choice, dead-end node, duplicate node
  names, concurrent write to a reducer-less key, and prompt-placeholder
  mismatch. Each **fails the build** and paints the offending node red — raw
  LangGraph would compile these and only fail at runtime.

## Sandbox (safe for public hosting)

The demo `exec`s the code you paste, so the pasted snippet is hardened before it
runs:

- **Input size cap** — snippets over 20 000 chars are rejected.
- **AST allow-list** — imports are restricted to `pttai` / `typing` /
  stdlib-safe roots; dunder attribute access (`__class__`, `__globals__`, …) and
  dangerous builtins (`eval`, `exec`, `open`, `__import__(...)`, `getattr`, …)
  are rejected statically.
- **Restricted builtins** — the exec namespace exposes only a safe subset of
  builtins.
- **Per-request timeout** — a snippet that never returns is aborted (building a
  graph never calls a model, so it is instant).

A malicious snippet such as `__import__('os').system(...)` is refused before it
executes.

## Deploy to Hugging Face Spaces

This folder is a self-contained Gradio Space (`app_file: app.py`,
`requirements.txt` installs `pttai` from git). To deploy:

```bash
# 1. create a Gradio Space (once), then clone it
huggingface-cli repo create pttai-playground --type space --space_sdk gradio
git clone https://huggingface.co/spaces/<you>/pttai-playground && cd pttai-playground

# 2. copy the demo files in (this README carries the Space metadata header)
cp /path/to/agentic-framework/demo/{app.py,requirements.txt,README.md} .

# 3. push
git add -A && git commit -m "pttai playground" && git push
```

The Space installs `pttai` from git per `requirements.txt`; `app.py` falls back
to a vendored offline `get_llm()` when `examples/_llm.py` isn't present, so the
Space needs nothing from the rest of the repo. mermaid.js loads from its CDN at
runtime (Spaces have network).
