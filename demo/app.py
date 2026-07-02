"""pttai interactive demo — paste DSL, see the compiled LangGraph + validator.

A Gradio playground for the pttai `>` DSL. You paste/edit a small pttai snippet;
on submit it is ``exec``'d in a HARDENED namespace (see the "Sandbox" section
below) that exposes the pttai public API plus ``get_llm()`` (an offline fake
model, so no API key is needed), and two things are rendered:

  1. the **graph** as a mermaid node/edge diagram (Gradio ships mermaid.js, so
     the fenced ```mermaid block renders client-side, offline). On a clean build
     this is the *compiled LangGraph*
     (``graph.compiled_graph.get_graph().draw_mermaid()``). When the build FAILS
     — ``AgenticGraph(...)`` raises a ``GraphValidationError`` (or a structural
     build error) *at construction, before any invoke* — the diagram is instead
     built from the pre-compile ``>``-wiring node/edge structure with the
     **offending node(s) painted red** and the error attached. That red-node
     picture is the whole point: the *picture of the bug*, before you ever run
     the graph. Raw LangGraph would have compiled the same graph and only blown
     up at runtime.
  2. the **build-time dataflow validator** output — the ``summary()`` table and
     every issue.

Sandbox (safe for public hosting): the pasted code is size-capped, AST-checked
(imports are allow-listed to pttai/typing/stdlib-safe roots; dunder access and
dangerous builtins like ``eval``/``open``/``__import__(...)``/``getattr`` are
rejected — and escape-prone modules like ``operator`` are kept off the import
allow-list, since ``operator.attrgetter('__globals__')`` would otherwise reach
dunder attributes through a *string* the AST check can't see), run with a
restricted ``__builtins__``, and executed in a **separate process that is
force-terminated past a per-request timeout** (so a runaway ``while True:``
cannot keep burning a CPU core). A malicious snippet (e.g.
``__import__('os').system(...)`` or the ``operator`` reach above) is refused
before it runs. This is hardened best-effort — not a substitute for OS-level
isolation.

    pip install -r demo/requirements.txt
    python demo/app.py
"""

import ast
import io
import multiprocessing as mp
import os
import queue
import re
import sys
import traceback
import builtins as _builtins

# Make `import pttai` (repo root) and `from _llm import get_llm` (examples/) work
# regardless of the cwd the demo is launched from. Harmless when pttai is instead
# pip-installed (e.g. on a Hugging Face Space): the inserted paths simply don't
# resolve and the installed package is used.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "examples"))


# --- offline LLM -----------------------------------------------------------

def _get_llm():
    """The offline fake chat model. Uses examples/_llm.py when present (local
    repo run); falls back to a tiny self-contained fake so the demo is a
    self-deployable Space with no dependency on examples/."""
    try:
        from _llm import get_llm  # examples/_llm.py, present in the repo
        return get_llm()
    except Exception:
        return _FallbackLLM()


class _FallbackLLM:
    """Minimal offline stand-in — enough for nodes to construct/validate. Never
    calls a network. The demo only builds+validates graphs, so a scripted reply
    is plenty; structured output returns a valid instance of the model."""

    def invoke(self, messages, **kwargs):
        from langchain_core.messages import AIMessage
        return AIMessage(content="(offline fake)")

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, model):
        from typing import Literal, get_args, get_origin

        def value_for(ann):
            if get_origin(ann) is Literal:
                return get_args(ann)[0]
            return {int: 0, float: 0.0, bool: True}.get(ann, "...")

        class _S:
            def invoke(_self, messages, **kw):
                return model(**{n: value_for(f.annotation)
                                for n, f in model.model_fields.items()})

            def bind_tools(_self, tools):
                return _self

        return _S()


# --- sandbox ---------------------------------------------------------------

MAX_CODE_CHARS = 20_000
EXEC_TIMEOUT_S = 8.0

# Import roots the AST allow-list permits. Everything else (os, sys,
# subprocess, socket, importlib, ...) is rejected before exec runs.
#
# NOTE: ``operator`` and ``functools`` are deliberately EXCLUDED. The AST check
# only blocks dunder attributes written literally (``x.__globals__``); it cannot
# see a dunder passed as a *string*. ``operator.attrgetter('__globals__')`` (or
# ``methodcaller``/``functools`` equivalents) would reach ``__globals__`` ->
# ``__builtins__`` -> ``__import__`` -> ``os`` through that string, bypassing the
# check entirely. Building a pttai graph never needs them, so they stay out.
_ALLOWED_IMPORT_ROOTS = {
    "pttai", "typing", "typing_extensions", "dataclasses",
    "enum", "collections", "itertools", "math",
    "langchain_core", "pydantic", "_llm",
}

# Names that must never be called from a pasted snippet (escape / IO vectors).
# ``getattr``/``vars``/``globals``/``locals`` and the ``operator`` string-based
# attribute reachers (``attrgetter``/``methodcaller``) are included so that even
# if such a name became reachable it is rejected before exec.
_FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "open", "__import__", "input", "breakpoint",
    "globals", "locals", "vars", "getattr", "setattr", "delattr", "memoryview",
    "attrgetter", "methodcaller", "help", "exit", "quit", "license", "credits",
}

# The only builtins exposed inside the exec namespace. ``__import__`` and
# ``__build_class__`` are required by the ``import`` / ``class`` statements
# themselves (not user-referenceable — a literal ``__import__(...)`` call is
# rejected by the AST check), so they are included but nothing dangerous is.
_SAFE_BUILTIN_NAMES = [
    "True", "False", "None", "bool", "int", "float", "complex", "str", "bytes",
    "list", "dict", "set", "frozenset", "tuple", "len", "range", "enumerate",
    "zip", "map", "filter", "sorted", "reversed", "sum", "min", "max", "abs",
    "round", "print", "isinstance", "issubclass", "type", "repr", "format",
    "hasattr", "iter", "next", "any", "all", "object", "super", "property",
    "staticmethod", "classmethod", "slice", "hash", "id", "ord", "chr",
    "Exception", "ValueError", "KeyError", "TypeError", "RuntimeError",
    "IndexError", "AttributeError", "StopIteration",
    "__import__", "__build_class__", "__name__",
]


class SafetyError(Exception):
    """Raised when a pasted snippet violates the sandbox policy."""


def _safe_builtins() -> dict:
    d = {}
    for name in _SAFE_BUILTIN_NAMES:
        if name == "__name__":
            d[name] = "__pttai_demo__"
        elif hasattr(_builtins, name):
            d[name] = getattr(_builtins, name)
    return d


def _validate_ast(code: str) -> None:
    """Reject dangerous constructs statically, before anything is executed."""
    if len(code) > MAX_CODE_CHARS:
        raise SafetyError(
            f"snippet is {len(code)} chars; the demo caps input at "
            f"{MAX_CODE_CHARS} chars")
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SafetyError(f"syntax error: {e}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in _ALLOWED_IMPORT_ROOTS:
                    raise SafetyError(
                        f"import of {alias.name!r} is not allowed; permitted "
                        f"roots: {sorted(_ALLOWED_IMPORT_ROOTS)}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in _ALLOWED_IMPORT_ROOTS:
                raise SafetyError(
                    f"import from {node.module!r} is not allowed; permitted "
                    f"roots: {sorted(_ALLOWED_IMPORT_ROOTS)}")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                raise SafetyError(
                    f"access to dunder attribute {node.attr!r} is not allowed")
        elif isinstance(node, ast.Name):
            if node.id in _FORBIDDEN_NAMES:
                raise SafetyError(f"use of {node.id!r} is not allowed")
            if (node.id.startswith("__") and node.id.endswith("__")
                    and node.id != "__name__"):
                raise SafetyError(f"use of dunder name {node.id!r} is not allowed")


def _safe_exec(code: str, ns: dict) -> None:
    """AST-check, then exec ``code`` in ``ns``.

    The runaway-code timeout is NOT enforced here: a daemon thread cannot be
    interrupted, so a ``while True:`` would keep burning a core after ``join()``
    returned. Instead ``build_and_report`` runs this whole path in a separate
    process it can ``terminate()``. This function only vets + executes."""
    _validate_ast(code)
    ns["__builtins__"] = _safe_builtins()
    compiled = compile(code, "<pttai-demo>", "exec")
    exec(compiled, ns)


# --- exec namespace with a tracing AgenticGraph ----------------------------

def _namespace():
    """A fresh exec namespace exposing the pttai public API + ``get_llm()``.

    ``AgenticGraph`` is wrapped so we capture ``start_node`` / ``end_nodes`` even
    when construction raises — that is what lets us draw the pre-compile wiring
    of a graph that never compiled."""
    import inspect
    import pttai
    from pttai.validation import GraphValidationError

    captured: dict = {}
    real_graph = pttai.AgenticGraph
    sig = inspect.signature(real_graph)

    def _tracing_graph(*args, **kwargs):
        try:
            bound = sig.bind_partial(*args, **kwargs)
            captured["start_node"] = bound.arguments.get("start_node")
            captured["end_nodes"] = bound.arguments.get("end_nodes")
        except TypeError:
            pass
        captured["args"] = args
        captured["kwargs"] = kwargs
        return real_graph(*args, **kwargs)

    ns = {"get_llm": _get_llm, "GraphValidationError": GraphValidationError}
    for name in pttai.__all__:
        ns[name] = getattr(pttai, name)
    ns["AgenticGraph"] = _tracing_graph  # override with the tracing wrapper
    return ns, captured


# --- red-node rendering ----------------------------------------------------

def _esc(text) -> str:
    return str(text).replace('"', "'").replace("\n", " ")


def _slug(text) -> str:
    return re.sub(r"[^0-9A-Za-z_]", "_", str(text))


def _html_escape(text) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _graph_html(heading: str, subtitle: str, mermaid_src: str) -> str:
    """Wrap a mermaid diagram for the ``gr.HTML`` graph pane. mermaid.js (loaded
    in the Blocks ``head``) renders every ``div.mermaid`` client-side. The
    source is HTML-escaped so the browser hands mermaid the literal diagram text
    (``<br/>``/``<i>`` in node labels survive as label markup, not stray tags)."""
    sub = f'<p style="color:#555;margin:.2em 0 .6em">{subtitle}</p>' if subtitle else ""
    return (
        f'<h3 style="margin:.2em 0">{heading}</h3>{sub}'
        f'<div class="mermaid" style="background:#fff">{_html_escape(mermaid_src)}</div>'
    )


def _wiring_mermaid(start, ends, offending) -> str:
    """Build a mermaid flowchart from the pre-compile ``>``-wiring structure
    (``.children`` / ``RouterNode.choices`` / ``Spread``), painting every node
    whose name is in ``offending`` red. Works even when the graph never
    compiled — it reads the node objects directly, not a compiled graph."""
    from pttai.node import Spread
    from pttai.nodes.decision_node import RouterNode

    offending = offending or set()
    end_seq = ends if isinstance(ends, (set, list, tuple)) else ([ends] if ends else [])
    end_ids = {id(n) for n in end_seq if n is not None}

    ids: dict = {}
    node_lines: dict = {}
    edges: list = []
    bad: list = []
    ctr = [0]

    def nid(n):
        if id(n) not in ids:
            ctr[0] += 1
            ids[id(n)] = f"n{ctr[0]}"
        return ids[id(n)]

    def register(n):
        i = nid(n)
        if i not in node_lines:
            nm = getattr(n, "name", None) or type(n).__name__
            node_lines[i] = f'{i}["{_esc(nm)}<br/><i>{type(n).__name__}</i>"]'
            if getattr(n, "name", None) in offending:
                bad.append(i)
        return i

    visited: set = set()

    def walk(n):
        i = register(n)
        if id(n) in visited:
            return i
        visited.add(id(n))
        if isinstance(n, RouterNode):
            for choice in getattr(n, "choices", []):
                if choice.child is None:  # dangling choice: show the missing branch
                    ph = f"{i}_x_{_slug(choice.name)}"
                    node_lines[ph] = f'{ph}(["unwired: {_esc(choice.name)}"])'
                    bad.append(ph)
                    edges.append(f'{i} -. "{_esc(choice.name)}" .-> {ph}')
                else:
                    edges.append(f'{i} -. "{_esc(choice.name)}" .-> {walk(choice.child)}')
        else:
            for child in (getattr(n, "children", None) or []):
                if isinstance(child, Spread):
                    wj = walk(child.worker)
                    edges.append(f'{i} == "map({_esc(child.field)})" ==> {wj}')
                    if child.collector is not None:
                        edges.append(f"{wj} --> {walk(child.collector)}")
                else:
                    edges.append(f"{i} --> {walk(child)}")
        if id(n) in end_ids:
            edges.append(f"{i} --> __end__")
        return i

    start_id = walk(start) if start is not None else None
    out = ["flowchart TD", "__start__([START])", "__end__([END])"]
    out += list(node_lines.values())
    if start_id is not None:
        out.append(f"__start__ --> {start_id}")
    seen: set = set()
    for e in edges:
        if e not in seen:
            seen.add(e)
            out.append(e)
    out.append("classDef bad fill:#ffcdd2,stroke:#c62828,stroke-width:3px,color:#111;")
    if bad:
        out.append("class " + ",".join(dict.fromkeys(bad)) + " bad;")
    return "\n".join(out)


def _diagnose_failure(captured):
    """Return ``(offending_node_names, issue_lines)`` for a failed build.

    Reconstructs the graph with ``validate=False`` to recover a structured
    ``ValidationReport`` (read-before-write, dangling choice, concurrent write,
    prompt-placeholder mismatch, ...). When even the unvalidated build raises a
    structural error (dead-end node, duplicate names), the offending name is
    parsed from the error message."""
    import pttai

    args = captured.get("args", ())
    kwargs = dict(captured.get("kwargs", {}))
    kwargs["validate"] = False
    try:
        g = pttai.AgenticGraph(*args, **kwargs)
        report = g.validate()
        offending = {i.node for i in report.errors}
        lines = [str(i) for i in report.issues] or ["(build failed with no structured issues)"]
        return offending, lines
    except Exception as e:  # structural build error (dead-end / duplicate name)
        msg = str(e)
        names = set(re.findall(r"'([^']+)'", msg))
        return names, [msg]


def _render_failed(captured):
    """Render the red-node diagram + error panel for a failed build."""
    start = captured.get("start_node")
    ends = captured.get("end_nodes")
    offending, lines = _diagnose_failure(captured)
    mermaid = _wiring_mermaid(start, ends, offending)
    mer = _graph_html(
        "Graph — build FAILED (the validator caught it before invoke)",
        "The <b>red</b> node(s) are where the build broke. This diagram is built "
        "from the <code>&gt;</code>-wiring, not a compiled graph — because the "
        "graph never compiled. Raw LangGraph would have compiled this and only "
        "failed at runtime.",
        mermaid,
    )
    val = (
        "### Build FAILED — errors caught at construction\n\n"
        "```\n" + "\n".join(lines) + "\n```\n\n"
        "*The build raised before any model call — no wasted tokens.*"
    )
    return mer, val


# --- build + report --------------------------------------------------------

def _build_and_report_impl(code: str):
    """AST-check + exec the snippet, then return ``(mermaid_md, validator_md)``.

    Runs inside the sandbox subprocess spawned by ``build_and_report``. Only its
    two return strings cross the process boundary (they are picklable; the live
    graph/LLM objects it builds are not)."""
    from pttai import AgenticGraph as _RealAgenticGraph
    from pttai.validation import GraphValidationError

    ns, captured = _namespace()
    try:
        _safe_exec(code, ns)
    except SafetyError as e:
        return (
            "<p><i>Snippet rejected by the sandbox.</i></p>",
            "### Rejected by the sandbox\n\n```\n" + str(e) + "\n```",
        )
    except GraphValidationError:
        return _render_failed(captured)
    except Exception:
        # A structural build error (dead-end / duplicate name) reaches here as a
        # ValueError AFTER AgenticGraph was called — captured has the wiring, so
        # still draw the red-node diagram. Anything raised before the graph call
        # is a plain snippet error -> show the traceback.
        if captured.get("start_node") is not None:
            return _render_failed(captured)
        tb = traceback.format_exc()
        return (
            "<p><i>No diagram: the snippet raised before building a graph.</i></p>",
            "### Error running snippet\n\n```\n" + tb + "\n```",
        )

    graphs = [v for v in ns.values() if isinstance(v, _RealAgenticGraph)]
    if not graphs:
        return (
            "<p><i>No <code>AgenticGraph</code> found.</i></p>",
            "Bind your graph to a variable, e.g. "
            "`graph = AgenticGraph(start_node=..., end_nodes={...})`.",
        )
    graph = graphs[-1]

    # (i) compiled LangGraph -> mermaid
    try:
        mermaid = graph.compiled_graph.get_graph().draw_mermaid()
        mer = _graph_html("Compiled LangGraph",
                          "Built clean — the validator confirmed every node's "
                          "reads are produced upstream.", mermaid)
    except Exception as e:
        mer = "<p><i>Could not render mermaid: " + _html_escape(repr(e)) + "</i></p>"

    # (ii) validator: summary() table + report issues
    report = graph.validate()
    buf = io.StringIO()
    graph.summary(file=buf)  # explicit file: summary()'s default binds the
                             # original sys.stdout, so redirect_stdout misses it
    status = "OK" if report.ok else "ERRORS"
    val = (
        f"### Validator: {status} — {len(report.errors)} error(s), "
        f"{len(report.warnings)} warning(s)\n\n"
        "```\n" + buf.getvalue().rstrip() + "\n```"
    )
    if report.issues:
        val += "\n\n**Issues:**\n\n```\n" + "\n".join(str(i) for i in report.issues) + "\n```"
    return mer, val


def _report_worker(code: str, q) -> None:
    """Subprocess entrypoint: run the build and hand the two strings back."""
    q.put(_build_and_report_impl(code))


def build_and_report(code: str):
    """Vet the snippet, then run the build in a terminable subprocess.

    Returns ``(mermaid_md, validator_md)``. The AST vetting runs in-parent so
    refusals are instant and the child only ever executes already-vetted code;
    the child is force-terminated if it runs past ``EXEC_TIMEOUT_S`` (a runaway
    ``while True:`` is killed, reclaiming the core, instead of leaking a thread)."""
    try:
        _validate_ast(code)
    except SafetyError as e:
        return (
            "<p><i>Snippet rejected by the sandbox.</i></p>",
            "### Rejected by the sandbox\n\n```\n" + str(e) + "\n```",
        )

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_report_worker, args=(code, q), daemon=True)
    p.start()
    p.join(EXEC_TIMEOUT_S)
    if p.is_alive():
        p.terminate()
        p.join()
        return (
            "<p><i>Snippet timed out and was terminated.</i></p>",
            "### Rejected by the sandbox\n\n```\n"
            f"snippet ran past {EXEC_TIMEOUT_S:.0f}s and was force-terminated "
            "(building + validating a graph never calls a model, so it should be "
            "instant)\n```",
        )
    try:
        return q.get_nowait()
    except queue.Empty:
        return (
            "<p><i>The sandbox process exited without a result.</i></p>",
            "### Error running snippet\n\n```\n"
            "the sandbox process exited unexpectedly\n```",
        )


# --- preset gallery --------------------------------------------------------

# One WORKING preset per examples/nlp pipeline.
WORKING_PRESETS = {
    "RAG QA  (retrieve → rerank → answer)": '''\
# RAG QA pipeline: retrieve passages, rerank them, then answer grounded in them.
# Each node's scalar reads are produced by the node before it, so the build-time
# validator confirms the dataflow and compiles clean.
retrieve = AgentNode(
    name="retrieve", llm=get_llm(),
    node_prompt="Retrieve passages relevant to the question: {question}",
    reads=["question"], writes={"passages": str},
)
rerank = AgentNode(
    name="rerank", llm=get_llm(),
    node_prompt="Rerank these passages by relevance: {passages}",
    reads=["passages"], writes={"context": str},
)
answer = AgentNode(
    name="answer", llm=get_llm(),
    node_prompt="Answer the question grounded ONLY in this context: {context}",
    reads=["context"], writes=["messages"],
)

retrieve > rerank > answer            # retrieve -> rerank -> answer
graph = AgenticGraph(start_node=retrieve, end_nodes={answer})
''',
    "Extract → Summarize  (typed structured output)": '''\
# Typed extraction feeding a summary chain. `extract` uses a dict-form
# writes={...} to switch to structured output and return NATIVE-typed fields
# (severity comes back an int); `summarize` reads those scalars and writes a
# one-line summary to its own channel. The validator confirms every scalar
# `summarize` reads is produced upstream.
extract = AgentNode(
    name="extract", llm=get_llm(),
    node_prompt="Extract structured fields from this support ticket: {ticket}",
    reads=["ticket"],
    writes={"product": str, "issue_type": str, "severity": int},
)
summarize = AgentNode(
    name="summarize", llm=get_llm(),
    node_prompt="Write a one-sentence triage summary: a severity-{severity} "
                "{issue_type} affecting {product}.",
    reads=["product", "issue_type", "severity"],
    writes=["summary"], output_field="summary",
)

extract > summarize
graph = AgenticGraph(start_node=extract, end_nodes={summarize})
''',
    "Doc triage  (decision routing)": '''\
# A DecisionNode classifies the document with constrained structured output
# (the model MUST return one of its choices), then routes to the matching
# handler. The chosen label goes to the dedicated `decision` channel, never
# into the conversation.
triage = DecisionNode(
    name="triage", llm=get_llm(),
    node_prompt="Classify this incoming document by its intent.",
    choices=["bug_report", "feature_request", "question"],
)
bug = AgentNode(name="bug", llm=get_llm(),
                node_prompt="Acknowledge the bug and ask for reproduction steps.")
feature = AgentNode(name="feature", llm=get_llm(),
                    node_prompt="Thank the user and log the feature request.")
question = AgentNode(name="question", llm=get_llm(),
                     node_prompt="Answer the user's question helpfully.")

triage["bug_report"] > bug
triage["feature_request"] > feature
triage["question"] > question
graph = AgenticGraph(start_node=triage, end_nodes={bug, feature, question})
''',
}

# One BROKEN preset per validator bug class. Each fails the
# build and paints the offending node red.
BROKEN_PRESETS = {
    "read-before-write": '''\
# BROKEN on purpose: the same RAG pipeline, but `rerank` is wired BEFORE
# `retrieve`. So `rerank` reads `passages` when nothing upstream has produced it
# yet — a read-before-write, i.e. a guaranteed runtime KeyError in raw LangGraph.
# pttai's validator FAILS the build here, before you ever invoke the graph.
retrieve = AgentNode(
    name="retrieve", llm=get_llm(),
    node_prompt="Retrieve passages relevant to the question: {question}",
    reads=["question"], writes={"passages": str},
)
rerank = AgentNode(
    name="rerank", llm=get_llm(),
    node_prompt="Rerank these passages by relevance: {passages}",
    reads=["passages"], writes={"context": str},
)
answer = AgentNode(
    name="answer", llm=get_llm(),
    node_prompt="Answer the question grounded ONLY in this context: {context}",
    reads=["context"], writes=["messages"],
)

rerank > retrieve > answer            # BUG: rerank runs before retrieve writes `passages`
graph = AgenticGraph(start_node=rerank, end_nodes={answer})
''',
    "dangling decision choice": '''\
# BROKEN: the decision has a choice ("feature") that is never wired to a
# handler. At runtime the model could pick it and the graph would have nowhere
# to go. The validator flags the dangling choice at build time.
classify = DecisionNode(
    name="classify", llm=get_llm(),
    node_prompt="Classify the incoming request.",
    choices=["bug", "feature"],
)
handle_bug = AgentNode(name="handle_bug", llm=get_llm(),
                       node_prompt="Acknowledge the bug.")

classify["bug"] > handle_bug          # BUG: choice "feature" is never wired
graph = AgenticGraph(start_node=classify, end_nodes={handle_bug})
''',
    "dead-end node": '''\
# BROKEN: `second` has no outgoing edge and is not declared an end node, so the
# graph dead-ends there. The build refuses to compile a node that goes nowhere.
first = AgentNode(name="first", llm=get_llm(), node_prompt="Do step one.")
second = AgentNode(name="second", llm=get_llm(), node_prompt="Do step two.")

first > second                        # BUG: `second` is never made an end node
graph = AgenticGraph(start_node=first, end_nodes=set())
''',
    "duplicate node names": '''\
# BROKEN: two distinct nodes share the name "node". Node names must be unique
# within a graph (they key the compiled StateGraph), so the build rejects it.
draft = AgentNode(name="node", llm=get_llm(), node_prompt="Draft a reply.")
review = AgentNode(name="node", llm=get_llm(), node_prompt="Review the reply.")

draft > review                        # BUG: both nodes are named "node"
graph = AgenticGraph(start_node=draft, end_nodes={review})
''',
    "concurrent write, no reducer": '''\
# BROKEN: `score_b` and `score_c` run in parallel and BOTH write `result`, a
# plain (reducer-less) channel. Two concurrent writers to a plain key raise
# InvalidUpdateError at runtime in LangGraph; the validator catches it at build.
a = AgentNode(name="a", llm=get_llm(), node_prompt="Start.")
score_b = AgentNode(name="score_b", llm=get_llm(),
                    node_prompt="Score option B.", output_field="result")
score_c = AgentNode(name="score_c", llm=get_llm(),
                    node_prompt="Score option C.", output_field="result")
join = AgentNode(name="join", llm=get_llm(),
                 node_prompt="Combine the two scores into a verdict: {result}",
                 reads=["result"])

(a > [score_b, score_c]) > join       # BUG: score_b and score_c both write `result`
graph = AgenticGraph(start_node=a, end_nodes={join})
''',
    "prompt placeholder mismatch": '''\
# BROKEN: `write`'s node_prompt interpolates {subject}, but the node only
# declares `topic` as a scalar read. `{subject}` has no value at run time —
# a guaranteed runtime KeyError. The validator catches the mismatch at build.
produce = AgentNode(name="produce", llm=get_llm(),
                    node_prompt="Pick a topic.", writes={"topic": str})
write = AgentNode(name="write", llm=get_llm(),
                  node_prompt="Write an essay about {subject}.", reads=["topic"])

produce > write                       # BUG: prompt uses {subject}, only `topic` is read
graph = AgenticGraph(start_node=produce, end_nodes={write})
''',
}

WORKING_EXAMPLE = WORKING_PRESETS["RAG QA  (retrieve → rerank → answer)"]


# mermaid.js, loaded once in the page <head>. A small polling/observer loop
# renders every un-rendered `div.mermaid` the graph pane emits (client-side,
# no server round-trip). Needs network for the CDN; when offline the raw
# diagram source is shown instead (graceful degradation).
MERMAID_HEAD = """
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
  mermaid.initialize({ startOnLoad: false, securityLevel: 'loose' });
  const render = () => {
    try { mermaid.run({ querySelector: 'div.mermaid:not([data-processed])' }); }
    catch (e) {}
  };
  window.addEventListener('load', render);
  new MutationObserver(render).observe(document.documentElement,
                                       { childList: true, subtree: true });
  setInterval(render, 400);
</script>
"""


def build_ui() -> "gr.Blocks":
    import gradio as gr

    with gr.Blocks(title="pttai playground", head=MERMAID_HEAD) as demo:
        gr.Markdown(
            "# pttai playground\n"
            "Paste a pttai `>`-DSL snippet, then **Build + Validate**. You get the "
            "**graph** diagram and the **build-time validator** output — "
            "read-before-write, dangling choices, concurrent writes, and duplicate "
            "names are caught *before* you ever invoke. When a build fails, the "
            "diagram still renders with the **offending node in red**. `get_llm()` "
            "is an offline fake, so **no API key** is needed."
        )
        with gr.Row():
            with gr.Column(scale=1):
                code = gr.Code(value=WORKING_EXAMPLE, language="python",
                               label="pttai DSL", lines=20)
                with gr.Row():
                    working_dd = gr.Dropdown(
                        choices=list(WORKING_PRESETS), label="Load a working preset",
                        value=None)
                    broken_dd = gr.Dropdown(
                        choices=list(BROKEN_PRESETS),
                        label="Load a broken preset (validator catches it)",
                        value=None)
                run = gr.Button("Build + Validate", variant="primary")
            with gr.Column(scale=1):
                mermaid_out = gr.HTML(label="Graph")
                validator_out = gr.Markdown(label="Validator")

        run.click(build_and_report, inputs=code, outputs=[mermaid_out, validator_out])
        working_dd.change(lambda k: WORKING_PRESETS[k] if k else gr.update(),
                          inputs=working_dd, outputs=code)
        broken_dd.change(lambda k: BROKEN_PRESETS[k] if k else gr.update(),
                         inputs=broken_dd, outputs=code)
    return demo


def _selftest() -> None:
    """Assert the sandbox statically refuses known escape vectors. Fast (AST
    only, no subprocess) — a guard that fails loudly if the policy regresses."""
    must_reject = [
        # the operator string-reach escape: blocked at the import, since operator
        # is no longer on the allow-list
        "import operator\noperator.attrgetter('__globals__')(print)",
        "from operator import attrgetter",
        "import functools",
        "__import__('os').system('echo x')",
        "import os",
        "getattr(object, '__globals__')",
        "().__class__.__bases__",
    ]
    for src in must_reject:
        try:
            _validate_ast(src)
        except SafetyError:
            continue
        raise AssertionError(f"sandbox failed to reject: {src!r}")


if __name__ == "__main__":
    _selftest()  # fail fast if the sandbox policy has regressed
    build_ui().launch()
