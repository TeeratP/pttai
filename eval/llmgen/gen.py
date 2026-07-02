"""Generate pttai pipelines with a frontier LLM from short NLP task specs.

The study's premise: give a frontier model ONLY the pttai docs plus a short,
natural task spec (RAG QA, multi-hop, extract->summarize, triage, rerank,
tool-use...) and ask it to write a runnable pttai pipeline. The model is NEVER
told to introduce bugs. Whatever dataflow bugs appear are model-produced, not
author-planted -- which is exactly what kills the "you graded your own bugs"
circularity objection against the self-authored corpus in ``eval/bugbench/``.

Each generated pipeline is written to ``generated/<slug>.py`` following one
contract so ``score.py`` can build every one uniformly::

    def build_graph(llm):
        ...
        return AgenticGraph(start_node=..., end_nodes={...})

``build_graph`` must CONSTRUCT (not invoke) the graph -- pttai's validator runs
inside ``AgenticGraph.__init__``, so construction alone is enough to score.

Requires ``OPENAI_API_KEY`` (this is the only step that calls the real model).
Run it via ``run_study.sh`` or directly:

    OPENAI_API_KEY=sk-... PYTHONPATH=. .venv/bin/python eval/llmgen/gen.py --per-spec 5
"""

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))  # -> from _llm import get_llm

# Docs handed to the model as its ONLY knowledge of pttai (relative to repo root).
DOC_FILES = [
    "docs/getting-started.md",
    "docs/node-types.md",
    "docs/validator.md",
    "docs/api-notes.md",
    "docs/coming-from-langgraph.md",
]

# Short, natural NLP task specs. Deliberately terse and bug-agnostic -- the model
# gets a goal, not a graph. Repeated ``--per-spec`` times (the model's own
# variation across samples supplies the bug distribution).
SPECS = [
    ("rag_qa",
     "Answer a user question grounded in a small document corpus: retrieve "
     "relevant passages with a search tool, then answer using only those passages."),
    ("multi_hop",
     "Answer a multi-hop question that needs two sequential lookups: first find "
     "an intermediate fact, then use it to look up and produce the final answer."),
    ("extract_summarize",
     "Extract the key entities and claims from a document, then write a short "
     "summary that uses the extracted information."),
    ("triage",
     "Triage an incoming support message: classify it as billing, technical, or "
     "other, then route it to a handler that drafts the appropriate reply."),
    ("rerank",
     "Given a query, retrieve candidate passages, rerank them by relevance, then "
     "answer the query using the top reranked passage."),
    ("tool_use_math",
     "Build a calculator agent that answers arithmetic word problems by calling "
     "add and multiply tools in a loop until it has the final number."),
    ("reflect_revise",
     "Draft an answer, then critique the draft, then revise the answer using the "
     "critique before returning it."),
    ("plan_execute",
     "Make a short plan for a research question, then execute the plan step by "
     "step and synthesize the findings into one answer."),
    ("sentiment_route",
     "Classify the sentiment of a product review as positive or negative, then "
     "route to a handler that writes an appropriate response."),
    ("map_reduce_summ",
     "Summarize each of several documents independently in parallel, then reduce "
     "the per-document summaries into one combined summary."),
]

CONTRACT = '''\
You are writing a Python module that defines ONE pttai pipeline for the task below.

Output requirements (STRICT):
- Output ONLY a single Python code block, no prose before or after.
- Define exactly one function: `def build_graph(llm):` that CONSTRUCTS and RETURNS
  an `AgenticGraph`. Do NOT call `.invoke(...)`, `.stream(...)`, or run the graph.
- Use the `llm` argument for every node's `llm=` (do not construct your own model).
- Import only from `pttai` (e.g. `from pttai import AgentNode, DecisionNode,
  ConditionNode, AgenticGraph, fanout`) and, if you need a retriever tool,
  `from pttai.tools import make_retriever_tool`.
- Prefer a real, working pipeline for the task. Write it the way the docs show.

Skeleton:

```python
from pttai import AgentNode, AgenticGraph

def build_graph(llm):
    # ... define and wire nodes ...
    return AgenticGraph(start_node=..., end_nodes={...})
```
'''


def _read_docs() -> str:
    parts = []
    for rel in DOC_FILES:
        path = os.path.join(ROOT, rel)
        if os.path.exists(path):
            with open(path) as f:
                parts.append(f"===== {rel} =====\n{f.read()}")
    return "\n\n".join(parts)


def _extract_code(text: str) -> str:
    """Pull the Python out of a model reply: prefer a fenced block, else the raw text."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip() + "\n"


def _require_key():
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit(
            "ERROR: OPENAI_API_KEY is not set.\n"
            "gen.py calls a real frontier model to generate pipelines and cannot "
            "run offline.\n"
            "Set your key and retry, e.g.:\n"
            "    OPENAI_API_KEY=sk-... PYTHONPATH=. .venv/bin/python eval/llmgen/gen.py\n"
            "(To verify the SCORING path without a key, run score.py over the "
            "committed samples: PYTHONPATH=. .venv/bin/python eval/llmgen/score.py "
            "--gen-dir samples)"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-spec", type=int, default=5,
                    help="pipelines to generate per task spec (default 5 -> ~50 total)")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "generated"),
                    help="where to write generated pipelines")
    ap.add_argument("--model", default=None,
                    help="override the model id (default: whatever get_llm/env selects)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="only these spec ids (default: all)")
    args = ap.parse_args()

    _require_key()  # fail fast with a clear message before importing model libs

    from langchain_core.messages import HumanMessage, SystemMessage
    from _llm import get_llm

    llm = get_llm()
    if args.model:
        # get_llm() returns ChatOpenAI when a key is set; swap the model id.
        try:
            llm.model_name = args.model
        except Exception:  # noqa: BLE001
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=args.model)

    docs = _read_docs()
    system = SystemMessage(
        "You write pttai pipelines. pttai is a declarative DSL over LangGraph; "
        "wire nodes with `>` and compile with AgenticGraph. Here are the docs, "
        "your ONLY reference:\n\n" + docs)

    specs = [(sid, txt) for sid, txt in SPECS
             if args.only is None or sid in args.only]

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = []
    n = 0
    for sid, task in specs:
        for i in range(args.per_spec):
            prompt = f"{CONTRACT}\n\nTASK: {task}\n"
            reply = llm.invoke([system, HumanMessage(prompt)])
            code = _extract_code(getattr(reply, "content", "") or "")
            slug = f"{sid}_{i:02d}"
            fname = f"{slug}.py"
            with open(os.path.join(args.out_dir, fname), "w") as f:
                f.write(f'"""Generated pipeline for task {sid!r} (sample {i}).\n\n'
                        f'TASK: {task}\n"""\n\n{code}')
            ok = True
            try:
                compile(code, fname, "exec")
            except SyntaxError as e:
                ok = False
                print(f"  ! {fname}: syntax error at generation time: {e}")
            manifest.append({"file": fname, "spec": sid, "sample": i,
                             "task": task, "syntax_ok": ok})
            n += 1
            print(f"  [{n}] wrote {fname}")

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump({"model": args.model or "get_llm-default",
                   "per_spec": args.per_spec, "count": n,
                   "specs": [s for s, _ in specs], "items": manifest}, f, indent=2)
    print(f"\nGenerated {n} pipelines into {os.path.relpath(args.out_dir)}/ "
          f"(+ manifest.json). Now run score.py.")


if __name__ == "__main__":
    main()
