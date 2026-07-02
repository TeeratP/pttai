"""Generate pttai pipelines with a small OpenAI model from short NLP task specs.

The study's premise: give the model (``gpt-5.4-nano`` by default, overridable via
``--model``) ONLY the pttai docs plus a short, natural task spec (RAG QA,
multi-hop, extract->summarize, triage, rerank, tool-use...) and ask it to write a
runnable pttai pipeline. The model is NEVER told to introduce bugs. Whatever
dataflow bugs appear are model-produced, not author-planted -- which is what
answers the "you graded your own bugs" circularity objection against the
self-authored corpus in ``eval/bugbench/``.

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
# gets a goal, NEVER a graph, keys, or any mention of bugs/the validator. The set
# is spread across structural shapes (chains, multi-branch routing, parallel
# fan-out/join, map-reduce, reflection loops, tool use, human-in-the-loop) so the
# model's own dataflow mistakes can land in many bug classes, not just one.
# Repeated ``--per-spec`` times (the model's variation supplies the distribution).
SPECS = [
    # --- retrieval / multi-stage chains -----------------------------------
    ("rag_qa",
     "Answer a user question grounded in a small document corpus: retrieve "
     "relevant passages with a search tool, then answer using only those passages."),
    ("multi_hop",
     "Answer a multi-hop question that needs two sequential lookups: first find "
     "an intermediate fact, then use it to look up and produce the final answer."),
    ("extract_summarize",
     "Extract the key entities and claims from a document, then write a short "
     "summary that uses the extracted information."),
    ("rerank",
     "Given a query, retrieve candidate passages, rerank them by relevance, then "
     "answer the query using the top reranked passage."),
    ("plan_execute",
     "Make a short plan for a research question, then execute the plan step by "
     "step and synthesize the findings into one answer."),
    ("translate_glossary",
     "Detect the language of a passage, translate it into English, then build a "
     "short glossary of the key terms that appear in the translation."),
    ("fact_check",
     "Given a factual claim, gather supporting and contradicting evidence, then "
     "issue a verdict with a short rationale."),
    ("dialogue_next_turn",
     "From a customer conversation, work out the requested action and the details "
     "still needed, then write the next agent turn."),
    # --- multi-branch routing / triage ------------------------------------
    ("triage",
     "Triage an incoming support message: classify it as billing, technical, or "
     "other, then route it to a handler that drafts the appropriate reply."),
    ("sentiment_route",
     "Classify the sentiment of a product review as positive or negative, then "
     "route to a handler that writes an appropriate response."),
    ("intent_router",
     "Classify a chat message into one of several intents -- weather, news, "
     "reminder, small talk, or unknown -- and route it to a matching responder."),
    ("change_review_route",
     "Read a description of a code change, decide whether it is a bug fix, a new "
     "feature, or a refactor, and route to a reviewer that writes fitting feedback."),
    ("escalation",
     "Read a support thread, judge how severe it is, then either auto-resolve it, "
     "ask the customer a clarifying question, or escalate it to a human agent."),
    # --- parallel fan-out / join ------------------------------------------
    ("map_reduce_summ",
     "Summarize each of several documents independently in parallel, then reduce "
     "the per-document summaries into one combined summary."),
    ("aspect_review",
     "Review a short piece of writing along several aspects at once -- clarity, "
     "grammar, and tone -- then merge the notes into one consolidated critique."),
    ("entity_dedup",
     "Extract the named entities from several documents in parallel, then "
     "reconcile and deduplicate them into one canonical list."),
    # --- reflection / evaluator-optimizer loops ---------------------------
    ("reflect_revise",
     "Draft an answer, then critique the draft, then revise the answer using the "
     "critique before returning it."),
    ("refine_until_good",
     "Draft a summary, score its quality, and keep refining it until the score is "
     "good enough or a few rounds have passed, then return the best version."),
    # --- tool use / ReAct -------------------------------------------------
    ("tool_use_math",
     "Build a calculator agent that answers arithmetic word problems by calling "
     "add and multiply tools in a loop until it has the final number."),
    # --- structured scoring -----------------------------------------------
    ("interview_grader",
     "Grade a candidate's interview answer on several rubric dimensions and give "
     "an overall hire / no-hire recommendation."),
]

# A COMPACT, ACCURATE API surface handed to the model alongside the prose docs so
# it stops hallucinating pttai's API (wrong kwargs, non-existent imports, misused
# internals). Every signature/kwarg/rule below is verified against the pttai
# source. This is API SURFACE + 1-2 line snippets ONLY -- deliberately NOT
# complete, copyable, bug-free pipelines, so the model's own dataflow mistakes are
# preserved for the study.
#
# HONEST DISCLOSURE: the cheatsheet DOES coach correct usage. It states the core
# API and wiring rules -- including several that map directly onto validator error
# classes: "a scalar key you read must be produced by an upstream node or seeded
# at invoke" (read-before-write), "wire EACH choice by name" (dangling-choice),
# and "tools=[...] CANNOT be combined with multi-field structured writes" (a build
# error). Telling the model the real API is legitimate, but we do NOT claim the
# cheatsheet is non-circular or bug-neutral: it teaches valid pttai. Whatever bugs
# remain in the generated pipelines are the model's, not steered by us toward any
# particular class.
API_CHEATSHEET = '''\
===== pttai API cheatsheet (authoritative -- prefer this over guesses) =====

Imports (these are the ONLY public names):
    from pttai import (AgentNode, DecisionNode, ConditionNode, HumanNode,
                       AgenticGraph, fanout, AgenticState)
    from pttai.tools import make_retriever_tool   # (optional; wraps a retriever)
There is no `pttai.tools.add`/`multiply` etc. -- define your own plain functions
and pass them as `tools=[...]`.

Node constructors (only these kwargs exist):
  AgentNode(name=None, llm=None, node_prompt="...", tools=None,
            max_tool_iterations=25, input_field="messages",
            output_field="messages", reads=None, writes=None,
            reasoning_effort=None, cache_ttl=None, retry=False)
  DecisionNode(name=None, llm=None, node_prompt="<required, non-empty>",
               choices=["a","b",...] <required>, tools=None,
               input_field="messages", reads=None, cache_ttl=None, retry=False)
      # DecisionNode does NOT accept writes/output_field/reasoning_effort.
      # It needs an llm; it writes its pick to the built-in `decision` field.
  ConditionNode(name=None, condition=<fn(state)->str>, choices=[...],
                reads=None, cache_ttl=None, retry=False)   # no llm, no prompt
  HumanNode(name=None, node_prompt="...", n=1, show=None, into="messages",
            cache_ttl=None, retry=False)                   # no llm, no tools
  AgenticGraph(start_node=<node>, end_nodes={<terminal nodes>}, ...)

LLM: use the injected `llm` argument for every node's `llm=` -- do NOT construct
your own model (no `ChatOpenAI(...)`, no api keys).

Wiring with `>` (deferred until AgenticGraph builds it):
  a > b > c                      # sequential chain
  a > fanout(b, c) > d           # parallel fan-out then join at d (also: a > [b,c] > d)
  worker.map("field") > collect  # map worker over state["field"], join into collect
  router["choice"] > handler     # DecisionNode/ConditionNode: wire EACH choice by name
      # `router > x` raises; a choice cannot route straight into fanout/.map.

reads / writes (AgentNode multi-key IO; use reads OR input_field, writes OR output_field):
  reads=["topic"]                # scalar -> interpolated into node_prompt via {topic}
  reads=["messages"]             # a message list -> used as conversation history
  writes=["messages"]            # (default) append the reply to the conversation
  writes=["answer"]              # one scalar key -> writes the reply's text there
  writes={"score": int}          # typed structured output -> a real int in state["score"]
      # tools=[...] CANNOT be combined with multi-field structured writes.
  A scalar key you read must be produced by an upstream node or seeded at invoke.

DecisionNode routing snippet:
    d = DecisionNode(llm=llm, node_prompt="Classify.", choices=["x","y"])
    d["x"] > handler_x;  d["y"] > handler_y
'''


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
            "gen.py calls a real OpenAI model (gpt-5.4-nano by default) to "
            "generate pipelines and cannot run offline.\n"
            "Set your key and retry, e.g.:\n"
            "    OPENAI_API_KEY=sk-... PYTHONPATH=. .venv/bin/python eval/llmgen/gen.py\n"
            "(To verify the SCORING path without a key, run score.py over the "
            "committed samples: PYTHONPATH=. .venv/bin/python eval/llmgen/score.py "
            "--gen-dir samples)"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-spec", type=int, default=5,
                    help="pipelines to generate per task spec (default 5; 20 specs -> ~100 total)")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "generated"),
                    help="where to write generated pipelines")
    ap.add_argument("--model", default=None,
                    help="override the model id (default: gpt-5.4-nano, whatever get_llm selects)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="only these spec ids (default: all)")
    ap.add_argument("--force", action="store_true",
                    help="wipe a non-empty --out-dir before generating (default: refuse, "
                         "so a run cannot silently mix generations from different harness versions)")
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

    # Resolve the ACTUAL model id used, so the run is attributable in the manifest.
    model_id = (getattr(llm, "model_name", None) or getattr(llm, "model", None)
                or args.model or "unknown")

    docs = _read_docs()
    system = SystemMessage(
        "You write pttai pipelines. pttai is a declarative DSL over LangGraph; "
        "wire nodes with `>` and compile with AgenticGraph. Below is a compact "
        "API cheatsheet (authoritative) followed by the full docs, your reference:"
        "\n\n" + API_CHEATSHEET + "\n\n" + docs)

    specs = [(sid, txt) for sid, txt in SPECS
             if args.only is None or sid in args.only]

    os.makedirs(args.out_dir, exist_ok=True)
    # Guard: never silently mix generations from different harness versions into
    # one dir (this once produced a contaminated 150-item run). Refuse a non-empty
    # out-dir unless --force, which wipes it first with a printed warning.
    existing = [f for f in os.listdir(args.out_dir)
                if f.endswith(".py") or f == "manifest.json"]
    if existing:
        if not args.force:
            sys.exit(f"ERROR: out-dir not empty ({len(existing)} generation file(s) in "
                     f"{args.out_dir}); pass --force to overwrite.")
        print(f"WARNING: --force given; wiping {len(existing)} existing "
              f"generation file(s) in {args.out_dir} before generating.")
        for f in existing:
            os.remove(os.path.join(args.out_dir, f))

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
        json.dump({"model": model_id,
                   "per_spec": args.per_spec, "count": n,
                   "specs": [s for s, _ in specs], "items": manifest}, f, indent=2)
    print(f"\nGenerated {n} pipelines into {os.path.relpath(args.out_dir)}/ "
          f"(+ manifest.json). Now run score.py.")


if __name__ == "__main__":
    main()
