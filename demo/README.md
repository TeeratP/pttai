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
- **Broken presets** (one example per validator bug class):
  read-before-write, dangling decision choice, dead-end node, duplicate node
  names, concurrent write to a reducer-less key, and prompt-placeholder
  mismatch. Each **fails the build** and paints the offending node red — raw
  LangGraph would compile these and only fail at runtime.

## Sandbox

The demo `exec`s the code you paste, so the pasted snippet is hardened before it
runs. This is a **hardened best-effort sandbox** — good enough for a public
playground, but *not* a substitute for OS-level isolation (containers, seccomp,
a locked-down user). Host accordingly.

**What the sandbox does:**

- **Input size cap** — snippets over 20 000 chars are rejected.
- **AST allow-list** — imports are restricted to `pttai` / `typing` /
  stdlib-safe roots. Escape-prone modules are deliberately kept *off* the
  list — notably `operator` and `functools`, because
  `operator.attrgetter('__globals__')` reaches a dunder attribute through a
  *string* the AST check can't see, chaining to `__globals__ → __builtins__ →
  __import__ → os`.
- **Static rejection** of literal dunder attribute access (`__class__`,
  `__globals__`, …) and of dangerous builtins (`eval`, `exec`, `open`,
  `__import__(...)`, `getattr`, `vars`, `globals`, `locals`,
  `attrgetter`/`methodcaller`, …).
- **Restricted builtins** — the exec namespace exposes only a safe subset of
  builtins.
- **Terminable timeout** — the code runs in a **separate process** that is
  force-terminated once it exceeds the per-request timeout. A runaway
  `while True:` is killed and its CPU core reclaimed (a plain thread could not be
  interrupted). Building + validating a graph never calls a model, so the normal
  case returns instantly.

A malicious snippet such as `__import__('os').system(...)` or the `operator`
reach above is refused *before* it executes (verified by `_selftest()`, which
runs at startup).

**What it does not guarantee:** it is not a defense against every possible
CPython escape or resource-exhaustion vector (memory, file descriptors, fork
bombs). For untrusted traffic at scale, run it behind real OS-level isolation.

## Deploy to Hugging Face Spaces

This folder is a self-contained Gradio Space (`app_file: app.py`). To deploy:

```bash
# 1. create a Gradio Space (once), then clone it
huggingface-cli repo create pttai-playground --type space --space_sdk gradio
git clone https://huggingface.co/spaces/<you>/pttai-playground && cd pttai-playground

# 2. copy the demo files in (this README carries the Space metadata header)
cp /path/to/pttai/demo/{app.py,requirements.txt,README.md} .

# 3. uncomment the `pttai @ git+https://...` line in requirements.txt so the
#    Space installs pttai (it isn't on the Space's Python path otherwise)

# 4. push
git add -A && git commit -m "pttai playground" && git push
```

`requirements.txt` ships with the `pttai` install line commented out (so a
local clone resolves `pttai` from source instead); uncomment it before pushing
to a Space. `app.py` falls back to a vendored offline `get_llm()` when
`examples/_llm.py` isn't present, so the Space needs nothing else from the rest
of the repo. mermaid.js loads from its CDN at runtime (Spaces have network).
