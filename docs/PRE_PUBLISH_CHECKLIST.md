# Pre-publish privacy / gitignore review (issue #22)

A one-time review of what becomes public when this repo goes open-source. Safe
mechanical changes are applied; owner-decision items are surfaced with a
recommendation. **Nothing under "OWNER DECISION" has been changed** — those are
the owner's calls.

> Note: several premises in issue #22 were written against an older snapshot of
> the repo (package `agentic_framework`, an `example_full.ipynb`, a gitignored
> `CLAUDE.md` file on disk, a private `.githooks/post-commit`). The repo has since
> evolved. This checklist reflects the **actual current state** on branch
> `chore/22-privacy-review` (identical to `main`).

## Secret sweep — CLEAN

- Command: `git grep -nEi 'api[_-]?key|secret|token|password|sk-[A-Za-z0-9]'`.
- **No real secrets in tracked files.** All hits are benign:
  - `.env.example` — an empty template (`OPENAI_API_KEY=` with no value, commented
    `LANGCHAIN_API_KEY=`). Intended to ship.
  - `README.md` / `examples/*` / `example.ipynb` — `sk-...` are placeholders in
    setup instructions, not real keys.
  - The `token` matches are the framework's legitimate `token` state channel
    (per-model usage), not credentials.
- **No `.env` in the working tree.**
- **No `.env` in reachable git history** (63 commits, clean). The old "leaked key
  remains in history and should be rotated" note in `docs/project.md` refers to
  commit `86e8d91`, which **no longer exists** — the history was rewritten, so the
  leak is not present in what would be published. (No action needed for the leak
  itself; see the `docs/project.md` row about scrubbing the stale narrative.)

## Public vs. private

| Item | Disposition | Notes |
|---|---|---|
| `pttai/` (source), `tests/`, `examples/`, `README.md`, `LICENSE`, `TODO.md`, `pyproject.toml`, `requirements.txt` | **PUBLIC** | Core library + tests + offline example scripts. Intended to ship. |
| `.env.example` | **PUBLIC** | Empty template, no values. Correct to ship. |
| `example.ipynb` | **PUBLIC** (already tracked) | This is the multi-section live-model tour that already ships. See the `example_full.ipynb` row. |
| `CLAUDE.md` | **KEPT PRIVATE** | Gitignored; not present on disk in this checkout. Stays out of the publish. See OWNER DECISION below. |
| `.env`, `.vscode/settings.json`, `__pycache__/`, `*.egg-info/`, `.venv/`, `*.pyc` | **KEPT PRIVATE** | Gitignored build/local artifacts. Correct. |
| `.gitignore` | **KEPT PRIVATE** | The repo deliberately does not track `.gitignore` (see recent-changes note); it stays local. |
| `.githooks/post-commit` | **OWNER DECISION** (currently PUBLIC — it is tracked) | See below. |
| `test.ipynb` | **OWNER DECISION** (currently PUBLIC — it is tracked) | See below. |
| `docs/project.md` | **PUBLIC**, but has scrub candidates | See below. |

## `example_full.ipynb` — intent already satisfied (no change made)

Issue #22 asked to un-gitignore `example_full.ipynb` so the "best 12-section tour"
ships. **That file does not exist in the repo, and there is no `example_full.ipynb`
line in `.gitignore`** — so there is nothing to un-ignore. The multi-section tour
that the issue refers to now ships as **`example.ipynb`**, which is already tracked
and public. The intent (best tour ships publicly) is already met. **No `.gitignore`
edit and no file staging were performed** — the premise is outdated.

## OWNER DECISION items

### 1. `CLAUDE.md` — keep private (recommended) vs. publish scrubbed
- Currently gitignored ("Personal AI-assistant working notes — kept local").
- **Recommendation: keep private.** It documents the owner's private agent
  workflow (task-board protocol, model-selection tiers, session URLs). Publishing
  even a scrubbed version adds maintenance burden for little public value.
  If a public contributor guide is wanted later, write a fresh `CONTRIBUTING.md`
  rather than scrubbing `CLAUDE.md`.

### 2. `.githooks/post-commit` — keep private / remove / document
- **This file is currently TRACKED, so it WILL ship publicly as-is.** (Issue #22
  assumed it was private — it is not.)
- It is the owner's headless-Claude "living docs" automation: on each commit it
  shells out to `claude -p ...` to rewrite the auto-maintained sections of
  `docs/project.md` and (conditionally) `CLAUDE.md`. It contains **no secrets**,
  but it is personal workflow tooling, not part of the library.
- **Recommendation: stop tracking it before publishing** (move it out of the repo,
  or add `.githooks/` to `.gitignore` and `git rm --cached` it). It only works for
  someone with the owner's `claude` CLI set up and references the private
  `CLAUDE.md`, so it has little value to public users and slightly exposes the
  owner's workflow. If the owner instead wants to showcase it, keep it but add a
  short header comment explaining it is optional/owner-specific.

### 3. `test.ipynb` — clean or drop before publishing (recommended: drop)
- Tracked, so it ships. Issues:
  - Imports `from langchain_ollama import ChatOllama`, which is **not declared** in
    `requirements.txt` or `pyproject.toml` — a public user running it hits an
    `ImportError`.
  - Contains **Thai-language commented-out code** (poem-composing agents,
    "การเมืองไทย") — leftover personal scratch content.
- **Recommendation: drop it** (the polished tour is `example.ipynb`), or, if kept,
  remove the `langchain_ollama` import + declare the dep, and delete the
  commented-out Thai scratch cells. Owner's call.

### 4. `docs/project.md` — scrub the auto-maintained sections before publishing
- The bottom **Status** and **Recent changes** sections are `<!-- AUTO-MAINTAINED
  by .githooks/post-commit -->` and are tied to the private hook (item 2).
- They currently: reference the private `.githooks/post-commit`; describe a
  gitignored `example.py` sandbox **that does not exist** on disk; list **dead
  commit SHAs** (e.g. `86e8d91`, `d7fe0a1`) from a rewritten history; and include a
  "leaked key remains in history and should be rotated" bullet that no longer
  applies (that commit is gone).
- **Recommendation: scrub/regenerate these two sections** (accurate current-state
  summary; drop dead-SHA and leaked-key narrative) or remove the auto-maintained
  markers before going public. If the hook is removed (item 2), also drop the
  line at the top of `docs/project.md` that documents the hook.

## Verification
- `git status`: only the new `docs/PRE_PUBLISH_CHECKLIST.md` is added. No
  `.gitignore` change (nothing to change) and no notebook content modified.
