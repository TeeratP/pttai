# Contributing to pttai

Thanks for your interest in contributing! pttai is a thin declarative layer over
[LangGraph](https://langchain-ai.github.io/langgraph/). This guide covers how to
get set up, run the tests, and submit a change.

## Setup

Requires **Python ≥ 3.10**.

```bash
git clone https://github.com/TeeratP/pttai && cd pttai
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # core + pytest
```

Optional extras: `[openai]` (live model calls via `langchain-openai`, needs
`OPENAI_API_KEY`), `[rag]` (Chroma retriever), `[docs]` (the MkDocs site).

## Running the tests

```bash
python -m pytest tests/
```

The full suite runs **offline with no API calls** — a scripted `FakeLLM` stands
in for a real model, so tests are fast and deterministic. CI runs this same
command across Python 3.10 / 3.11 / 3.12 on every push and pull request (see
`.github/workflows/ci.yml`). **Please make sure `pytest tests/` is green before
opening a PR**, and add tests for any new behavior.

## Building the docs

```bash
pip install -e ".[docs]"
mkdocs serve            # live preview at http://127.0.0.1:8000
mkdocs build --strict   # what CI/the docs deploy runs — must pass with no warnings
```

New pages must be added to the `nav:` in `mkdocs.yml`, and `mkdocs build
--strict` must pass with no warnings (broken links and orphaned pages fail the
build).

## Submitting a pull request

1. Fork and create a topic branch off `main`.
2. Keep changes **surgical** — touch only what the change requires; match the
   existing style rather than reformatting unrelated code.
3. Add or update tests; run `python -m pytest tests/`.
4. If you touched docs, run `mkdocs build --strict`.
5. Open a PR against `main` with a clear description of *what* changed and
   *why*. Link any related issue.

## Code style

- Python, targeting 3.10+. Match the surrounding code.
- New node types subclass `Node` (`pttai/node.py`), implement
  `__call__(state) -> delta` **returning only the keys they update** (never
  mutate `state` in place), set `self.child = None` in `__init__`, and register
  in `pttai/nodes/__init__.py`. Add an `isinstance` branch in
  `AgenticGraph._build_graph` only if the node needs non-default edge handling.
- Keep the LLM injected per-node (`llm=`), and preserve the compile-time
  validator's guarantees — if you add a node that reads/writes state, teach
  `_node_io` about it so the dataflow analysis stays sound.

See [`CLAUDE.md`](CLAUDE.md) for the architecture overview and
[`docs/`](docs/) for the user-facing documentation.
