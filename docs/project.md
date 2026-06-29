# Project status & working notes

Working notes for the project — setup, dependencies, how to verify work, current status, rough edges, and the roadmap. (The stable architecture and conventions live in the README's *Design decisions* and *How the `>` DSL compiles* sections.)

The **Status** and **Recent changes** sections at the bottom are auto-maintained by the post-commit hook (`.githooks/post-commit`) via headless Claude. You can also edit them by hand — the hook only refreshes those two sections.

## Setup & running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # editable install + pytest  (or: uv pip install -e ".[dev]")
```

- Python **3.10–3.12** (the pinned `requirements.txt` lock targets 3.12; 3.14 can't build the pinned numpy/pandas).
- Real LLM calls need `OPENAI_API_KEY` in `.env` — see `.env.example`.
- Optional extras: `.[openai]` (langchain-openai + dotenv for the notebook), `.[rag]` (langchain-chroma for `ChromaRAG`).

## Dependencies & gotchas

- `pyproject.toml` declares the **direct** deps (langgraph≥1.0, langchain-core≥1.0, pydantic) + extras — **no `langchain` meta-dep** (tool classes come from `langchain_core.tools`). `requirements.txt` is the **frozen lock** (langgraph 1.2.5, langchain-core 1.4.7, langchain-openai 1.3.2, pytest).
- `test.ipynb` imports `langchain_ollama` (`ChatOllama`), which is **not** declared. Either `pip install langchain-ollama` or stick to the `ChatOpenAI` line.
- `ChromaRAG` needs the `rag` extra; the import of `langchain_chroma` is lazy so `pttai.tools` imports fine without it.

## Verifying changes

- **Automated:** `python -m pytest tests/` — 69 tests, no API calls (a scripted `FakeLLM` stands in). Covers state reducers, graph construction, routing, the tool-call loop, interrupt/resume, RAG tool wiring, streaming/async, configurable fields, parallel fan-out/join, map-reduce, multi-key IO, static validation, and node caching/retry/`reasoning_effort`/`durability`.
- **Offline feature tour:** `python examples/parallel_usage.py` — no API key; demonstrates parallel fan-out/join, map-reduce, typed multi-key IO, `summary()`, and a caught build-time validation error on an inline fake LLM.
- **Manual / live model:** `python examples/sample_usage.py` (or `test.ipynb`) with `OPENAI_API_KEY` set — exercises **gpt-5.4-nano** end-to-end (tool call → routing → reasoning handler); inspect `state['log']` for the per-node trace.

## Current status & rough edges

The package is now **`pttai`** (renamed from the old `agentic-framework` import; PyPI name `pttai`). Phases 1–5 are complete: reducer state, delta nodes, routing off messages, interrupt/resume, RAG tooling, packaging, streaming/async, configurable fields, the **LangGraph 1.0 stack**, **gpt-5.4-nano** + node `reasoning_effort`, node `cache_ttl`/`retry`, a `durability` passthrough, and the Phase 5 surface: **parallel fan-out/join** (`a > fanout(b, c) > d` and `a > [b, c] > d`, multi-node branch chains), **map-reduce** (`worker.map("field")`, Send-based, deferred collector), **typed multi-key state IO** (`reads=[...]`/`writes=[...]`), and **compile-time static validation** (`validate=`/`validate()`/`summary()`/`inputs=`). Verified live on gpt-5.4-nano. Remaining rough edges / deferred items:

- **Structured multi-write fields are `str`-only in v1.** A typed/nested escape hatch (`output_model=`) is planned, not built.
- **Map workers don't echo their source item** — the worker receives each item but returns only its reply, and must output `messages`.
- **`must` (all-paths) validation is imprecise for decision→handler→merge** — warning-only, never a wrong hard error (hard errors come only from the precise `may` analysis).
- **Async is graph-level only.** `ainvoke`/`astream` run the sync nodes in LangGraph's threadpool; true per-node async LLM calls aren't implemented — which is also why **node `timeout` is unavailable** (LangGraph only times out async nodes).
- **`reasoning_effort` is AgentNode-only** — it conflicts with `DecisionNode`'s structured output on current OpenAI models.
- **ChromaRAG is untested end-to-end.** Only `make_retriever_tool` is covered (fake retriever); the Chroma path needs live embeddings.

## Roadmap

Done:
- [x] Resumable Human/Input node — `checkpointer` + `Command(resume=...)`.
- [x] RAG tools — `make_retriever_tool` + `ChromaRAG`.
- [x] Configurable input/output fields per node — `AgentNode`/`DecisionNode`.
- [x] LangSmith integration — automatic via env vars (see `.env.example`).
- [x] LangGraph 1.0 upgrade; gpt-5.4-nano + node `reasoning_effort`; node `cache_ttl`/`retry`; `durability` passthrough (Phase 4 #1–4).
- [x] Parallel fan-out/join (`fanout(...)` / `[a, b]`, multi-node branch chains) and map-reduce (`worker.map("field")`).
- [x] Typed multi-key state IO (`reads=[...]`/`writes=[...]`: type-based reads, 3 write modes, tools-XOR-structured).
- [x] Compile-time static validation (`validate=`/`validate()`/`summary()`/`inputs=`).
- [x] Rename to **`pttai`** + public import surface (`from pttai import AgenticGraph, AgentNode, …`).

Remaining:
- [ ] Typed/nested structured output via `output_model=` (today multi-write fields are `str`-only).
- [ ] Map workers that echo their source item alongside the reply.
- [ ] Tighten `must` (all-paths) validation precision for decision→handler→merge.
- [ ] Conversation memory (`ConversationSummaryMemory`-style summarization node).
- [ ] TTS and STT nodes.
- [ ] True per-node async LLM execution (also unlocks node `timeout` and token-level streaming).

## Status

<!-- AUTO-MAINTAINED by .githooks/post-commit — keep this a 1-3 sentence summary -->
Phases 1–5 are complete and the package is the public-ready `pttai` (LangGraph 1.0 / LangChain 1.0 stack): reducer state, delta nodes, decision routing, interrupt/resume, RAG tooling, packaging, streaming/async, configurable I/O, `reasoning_effort`/`cache_ttl`/`retry`/`durability`, plus the Phase 5 surface — parallel fan-out/join, map-reduce, schema-free typed multi-key IO, and compile-time static validation (`summary()`/`validate()`). 69 tests pass with no API calls; offline tours live in `examples/`, and a runnable live-model sandbox showcasing the schema-free API and all seven shipped features is at `example.py` (gitignored).

## Recent changes

<!-- AUTO-MAINTAINED by .githooks/post-commit — newest first, max 15 bullets -->
- 52ddd24 2026-06-29 — Add runnable `example.py` live-model sandbox (gitignored): schema-free API tour of all 7 shipped features — parallel fan-out/join, per-model token usage, opt-in prompt caching, map-reduce, typed multi-key IO, `summary()`, build-time validation
- 63b6e53 2026-06-29 — Stop tracking `.gitignore` (kept local per request); also ignore the personal `example.ipynb` scratch notebook
- e5ac44c 2026-06-20 — Rewrite README as a portfolio-grade showcase: tagline + badges, problem→solution value prop, a Mermaid diagram of the example graph, and a "Design decisions" section (all claims verified against source)
- 842f24e 2026-06-20 — Expand README with a Features list, surface the 33-test suite count, and add a Limitations section (docs only, grounded in existing code)
- 0822000 2026-06-20 — Rewrite README as a finished project (accurate LangGraph 1.0 feature set, `>` DSL/InputNode/RAG/caching) + add MIT LICENSE for the now-public repo
- d3f52d1 2026-06-17 — Stop tracking build artifacts (`pttai.egg-info`); add `__pycache__/`/`*.egg-info/`/`.venv/` to .gitignore
- a561424 2026-06-17 — Phase 4 #2-4: AgentNode `reasoning_effort`, per-node `cache_ttl`/`retry` knobs (auto `InMemoryCache`), and `durability` passthrough on invoke/stream; 33 tests pass
- 666ef08 2026-06-17 — Phase 4 #1: upgrade to LangGraph 1.0 / LangChain 1.0 stack (tool imports → `langchain_core.tools`, `MemorySaver`→`InMemorySaver`, relocked deps); 28 tests pass unchanged
- 86e8d91 2026-06-17 — Stop tracking `.env` (committed before the ignore rule); leaked key remains in history and should be rotated
- 9b7a9a6 2026-06-17 — Add examples/sample_usage.py: runnable tool→decision→branch demo wired with `>`
- 7f1735b 2026-06-17 — Phase 3: streaming/async, configurable I/O fields, LangSmith docs
- 9bfebfa 2026-06-17 — Phase 2: interrupt/resume, tool-loop cap, RAG tool, packaging
- bcd7d8f 2026-06-17 — Phase 1: pytest suite + clear stale notebook outputs
- a97590e 2026-06-17 — Phase 1: nodes return deltas; routing off messages; dup-name guard
- c86c91a 2026-06-17 — Phase 1: reducer-based AgenticState + decision routing field
- d7fe0a1 2026-06-17 — Add static/living docs split and post-commit docs hook
- ecc40bb 2026-06-17 — Add InputNode for human-in-the-loop and clean up state/imports
