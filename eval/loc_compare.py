"""LOC + node-count comparison: pttai vs. raw LangGraph.

Every file under ``examples/architectures/`` (and a curated few of
``examples/basics/``) ships a ``pttai_version()`` and a ``langgraph_version()``
with identical runtime behavior. This script parses each with ``ast`` and counts,
per function body:

  * LOC   — non-blank, non-comment-only *physical* source lines in the function
            body (a leading docstring, if any, is skipped; the def line is not
            counted). This is the exact rule ``docs/COMPARISON.md`` states, and
            it reproduces COMPARISON's hand-cited numbers verbatim.
  * nodes — graph nodes declared: pttai counts node constructors
            (AgentNode/DecisionNode/ConditionNode/HumanNode); LangGraph counts
            ``builder.add_node(...)`` calls.

Emits a Markdown table to stdout and writes ``eval/loc_results.csv``. Reconciles
the three architectures COMPARISON hardcodes; the harness is source of truth, so
any delta is reported.

    PYTHONPATH=. .venv/bin/python eval/loc_compare.py
"""

import ast
import csv
import glob
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loc_results.csv")

# All architectures, plus a curated few basics (representative topologies).
ARCH_GLOB = os.path.join(ROOT, "examples", "architectures", "*.py")
BASICS = ["01_single_agent.py", "03_sequential.py",
          "04_decision_routing.py", "06_parallel_fanout.py"]

# COMPARISON.md's hand-cited numbers (pttai LOC, raw LangGraph LOC).
COMPARISON_REF = {
    "react_agent.py": (5, 14),
    "prompt_chaining.py": (11, 20),
    "routing.py": (15, 35),
}

PTTAI_NODE_CLASSES = {"AgentNode", "DecisionNode", "ConditionNode", "HumanNode"}


def _funcs(tree):
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def _body_loc(func, lines):
    """Non-blank, non-comment physical source lines in a function body."""
    body = func.body
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]                       # drop a leading docstring
    if not body:
        return 0
    start, end = body[0].lineno, func.end_lineno
    n = 0
    for i in range(start, end + 1):
        s = lines[i - 1].strip()
        if s and not s.startswith("#"):
            n += 1
    return n


def _count_pttai_nodes(func):
    return sum(1 for n in ast.walk(func)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id in PTTAI_NODE_CLASSES)


def _count_lg_nodes(func):
    return sum(1 for n in ast.walk(func)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == "add_node")


def analyze(path):
    src = open(path).read()
    lines = src.splitlines()
    fs = _funcs(ast.parse(src))
    if "pttai_version" not in fs or "langgraph_version" not in fs:
        return None
    p, l = fs["pttai_version"], fs["langgraph_version"]
    p_loc, l_loc = _body_loc(p, lines), _body_loc(l, lines)
    return {
        "file": os.path.basename(path),
        "pttai_loc": p_loc,
        "lg_loc": l_loc,
        "reduction_pct": round(100 * (l_loc - p_loc) / l_loc) if l_loc else 0,
        "pttai_nodes": _count_pttai_nodes(p),
        "lg_nodes": _count_lg_nodes(l),
    }


def _table(title, rows):
    print(f"\n### {title}\n")
    print("| Example | pttai LOC | LangGraph LOC | reduction | pttai nodes | LangGraph nodes |")
    print("|---|:--:|:--:|:--:|:--:|:--:|")
    for r in rows:
        print(f"| `{r['file']}` | {r['pttai_loc']} | {r['lg_loc']} | "
              f"{r['reduction_pct']}% | {r['pttai_nodes']} | {r['lg_nodes']} |")


def main():
    arch = [analyze(p) for p in sorted(glob.glob(ARCH_GLOB))]
    arch = [r for r in arch if r]
    basics = []
    for name in BASICS:
        r = analyze(os.path.join(ROOT, "examples", "basics", name))
        if r:
            basics.append(r)

    print("# LOC + node-count: pttai vs. raw LangGraph")
    print("\nLOC = non-blank, non-comment physical source lines in the "
          "`pttai_version()` / `langgraph_version()` body (counted via `ast`).")
    _table("Architectures", arch)
    _table("Basics (selected)", basics)

    all_rows = arch + basics
    tp = sum(r["pttai_loc"] for r in all_rows)
    tl = sum(r["lg_loc"] for r in all_rows)
    print(f"\n**Totals across {len(all_rows)} examples:** "
          f"pttai {tp} LOC vs LangGraph {tl} LOC "
          f"(~{round(100 * (tl - tp) / tl)}% fewer lines).")

    # Reconcile with COMPARISON.md — harness is source of truth.
    print("\n### Reconciliation with `docs/COMPARISON.md`\n")
    print("| Example | metric | COMPARISON | harness | delta |")
    print("|---|---|:--:|:--:|:--:|")
    by_file = {r["file"]: r for r in all_rows}
    any_delta = False
    for fname, (ref_p, ref_l) in COMPARISON_REF.items():
        r = by_file.get(fname)
        if not r:
            continue
        for metric, ref, got in (("pttai LOC", ref_p, r["pttai_loc"]),
                                 ("LangGraph LOC", ref_l, r["lg_loc"])):
            d = got - ref
            if d:
                any_delta = True
            print(f"| `{fname}` | {metric} | {ref} | {got} | "
                  f"{'+' if d > 0 else ''}{d} |")
    print("\n" + ("All COMPARISON numbers reproduced exactly (delta 0)."
                  if not any_delta else
                  "Deltas found above — the harness numbers supersede COMPARISON.md."))

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "file", "category", "pttai_loc", "lg_loc",
            "reduction_pct", "pttai_nodes", "lg_nodes"])
        w.writeheader()
        for r in arch:
            w.writerow({**r, "category": "architecture"})
        for r in basics:
            w.writerow({**r, "category": "basic"})
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
