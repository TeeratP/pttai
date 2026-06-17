# Project status & working notes

Living companion to [CLAUDE.md](../CLAUDE.md). CLAUDE.md holds the stable architecture; this file holds everything that changes — setup, dependencies, how to verify work, current status, rough edges, and the roadmap.

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

- `pyproject.toml` declares the **direct** deps (langgraph, langchain, langchain-core, pydantic) + extras. `requirements.txt` remains the **frozen lock** for exact reproduction (now includes pytest).
- `test.ipynb` imports `langchain_ollama` (`ChatOllama`), which is **not** declared. Either `pip install langchain-ollama` or stick to the `ChatOpenAI` line.
- `ChromaRAG` needs the `rag` extra; the import of `langchain_chroma` is lazy so `agentic_framework.tools` imports fine without it.

## Verifying changes

- **Automated:** `python -m pytest tests/` — 28 tests, no API calls (a scripted `FakeLLM` stands in for the model). Covers state reducers, graph construction, routing, the tool-call loop, interrupt/resume, RAG tool wiring, streaming/async, and configurable fields.
- **Manual / live model:** run `test.ipynb` with `OPENAI_API_KEY` set; inspect `state['log']` for the per-node trace. (Notebook outputs are cleared in git — re-run locally.)

## Current status & rough edges

The 3-phase LangGraph-native refactor (branch `roadmap/phases-1-3`) is complete: reducer-based state, delta-returning nodes, routing off the message stream, interrupt/resume, RAG tooling, packaging, streaming/async, and configurable I/O fields. Remaining rough edges:

- **Async is graph-level only.** `ainvoke`/`astream` run the sync nodes in LangGraph's threadpool; true per-node async LLM calls (a node-level `ainvoke` using `llm.ainvoke`) are not implemented yet.
- **ChromaRAG is untested end-to-end.** Only `make_retriever_tool` is covered (fake retriever); the Chroma path needs live embeddings and isn't exercised by the suite.
- `test.ipynb` hasn't been re-run against a live model since the refactor (outputs cleared).

## Roadmap

Done in the refactor:
- [x] Resumable Human/Input node — `checkpointer` + `Command(resume=...)`.
- [x] RAG tools — `make_retriever_tool` + `ChromaRAG`.
- [x] Configurable input/output fields per node — `AgentNode`/`DecisionNode`.
- [x] LangSmith integration — automatic via env vars (see `.env.example`).

Remaining:
- [ ] Conversation memory (`ConversationSummaryMemory`-style summarization node).
- [ ] TTS and STT nodes.
- [ ] True per-node async LLM execution.

## Status

<!-- AUTO-MAINTAINED by .githooks/post-commit — keep this a 1-3 sentence summary -->
3-phase LangGraph-native refactor complete on branch `roadmap/phases-1-3`: reducer state, delta nodes, decision routing field, interrupt/resume, RAG tool, packaging, streaming/async, configurable I/O fields. 28 tests passing. A runnable end-to-end demo now lives at `examples/sample_usage.py` (tool-using agent → decision routing → branched handlers, wired with `>`).

## Recent changes

<!-- AUTO-MAINTAINED by .githooks/post-commit — newest first, max 15 bullets -->
- 86e8d91 2026-06-17 — Stop tracking `.env` (committed before the ignore rule); leaked key remains in history and should be rotated
- 9b7a9a6 2026-06-17 — Add examples/sample_usage.py: runnable tool→decision→branch demo wired with `>`
- 7f1735b 2026-06-17 — Phase 3: streaming/async, configurable I/O fields, LangSmith docs
- 9bfebfa 2026-06-17 — Phase 2: interrupt/resume, tool-loop cap, RAG tool, packaging
- bcd7d8f 2026-06-17 — Phase 1: pytest suite + clear stale notebook outputs
- a97590e 2026-06-17 — Phase 1: nodes return deltas; routing off messages; dup-name guard
- c86c91a 2026-06-17 — Phase 1: reducer-based AgenticState + decision routing field
- d7fe0a1 2026-06-17 — Add static/living docs split and post-commit docs hook
- ecc40bb 2026-06-17 — Add InputNode for human-in-the-loop and clean up state/imports
