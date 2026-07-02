"""Compile-time state-availability validation for AgenticGraph.

The core promise: when a graph is built we statically verify that
every state key a node READS is produced upstream (or present in the initial
schema), and that every key it WRITES is declared in the schema. The build
FAILS on a hard error.

The core is a forward dataflow over a tagged edge list (recorded during the
build): `may` (any-path availability, a monotone-increasing union fixpoint)
and `must` (guaranteed-on-all-paths, a monotone-decreasing fixpoint that
UNIONs across AND-parallel joins and INTERSECTs within an exclusive
DecisionNode choice-group). Hard errors come from `may`; `must` only drives
warnings. Loop-back edges are excluded when computing incoming availability, so
loop-carried reads (a key produced only by a downstream node that cycles back)
are caught as read-before-write on the first iteration rather than silently
credited.

This is a may/must dataflow analysis, not a soundness proof. It reasons over
DECLARED node `reads`/`writes` and the graph's edges — it does NOT read the
code inside `ConditionNode` predicates (arbitrary lambdas), tool bodies, or
other custom callables, so a state key touched only inside such a callable is
invisible to it unless the node declares it. Within those limits it flags
read-before-write and undeclared reads/writes; no false positives have been
observed on our example corpus, but "compiles" is not a guarantee of dataflow
soundness.
"""

import sys
from typing import get_type_hints

# LangGraph's sentinel node names (kept as literals to avoid importing here).
START = "__start__"
END = "__end__"


class GraphValidationError(Exception):
    """Raised at build time when validation finds a hard error."""


class Issue:
    """A single validation finding."""

    def __init__(self, level: str, node: str, message: str):
        self.level = level  # "error" | "warning"
        self.node = node
        self.message = message

    def __str__(self):
        return f"[{self.level}] {self.node}: {self.message}"


class ValidationReport:
    """Collected issues for one graph; `.ok` is True when there are no errors."""

    def __init__(self, issues, graph_name: str = "graph"):
        self.issues = list(issues)
        self.graph_name = graph_name

    @property
    def errors(self):
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self):
        return [i for i in self.issues if i.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def __str__(self):
        head = (f"AgenticGraph {self.graph_name!r}: "
                f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)")
        lines = [head] + [f"  {i}" for i in self.issues]
        return "\n".join(lines)


def schema_keys(state) -> set:
    """Every key the state schema declares (including inherited TypedDict keys)."""
    return set(get_type_hints(state, include_extras=True))


def reduced_keys(state) -> set:
    """Keys whose annotation is `Annotated[..., <reducer>]` (i.e. have a reducer)."""
    hints = get_type_hints(state, include_extras=True)
    return {k for k, v in hints.items() if hasattr(v, "__metadata__")}


def _back_edges(edges):
    """The set of loop-back edges `(src, dst)` — edges that close a cycle.

    Detected by an iterative three-colour DFS from START: an edge into a node
    still on the current DFS path (GRAY / on-stack) closes a cycle, so it is a
    back-edge. Removing every back-edge yields the graph's acyclic condensation.

    Why this matters for availability: a back-edge carries LOOP-CARRIED keys —
    values that only exist after the loop body has run at least once. On the
    FIRST entry into a cycle they are absent, so a node that reads a key written
    ONLY by a downstream node that loops back would KeyError on iteration one.
    Excluding back-edges when computing incoming availability makes the analysis
    reflect that first-iteration reality (loop-carried keys are never counted as
    guaranteed, and not even as some-path available for the first read)."""
    adj = {}
    for s, d, _ in edges:
        if d == END:
            continue
        adj.setdefault(s, []).append(d)
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {}
    back = set()
    # Root the DFS at START, then sweep any remaining nodes so a cycle sitting
    # in an unreachable-from-START component is still recognised (its nodes are
    # reported unreachable elsewhere; here we just avoid leaking loop-carried
    # availability into them).
    roots = [START] + [n for n in adj if n != START]
    for root in roots:
        if color.get(root, WHITE) != WHITE:
            continue
        color[root] = GRAY
        stack = [(root, iter(adj.get(root, ())))]
        while stack:
            node, it = stack[-1]
            pushed = False
            for nxt in it:
                c = color.get(nxt, WHITE)
                if c == GRAY:                 # edge into an on-stack node: back-edge
                    back.add((node, nxt))
                elif c == WHITE:              # tree edge: descend
                    color[nxt] = GRAY
                    stack.append((nxt, iter(adj.get(nxt, ()))))
                    pushed = True
                    break
                # BLACK (already finished): forward/cross edge — ignore
            if not pushed:
                color[node] = BLACK
                stack.pop()
    return back


def compute_availability(initial: set, edges: list, writes: dict, send_workers: "dict | None" = None, spread_collectors: "dict | None" = None) -> tuple:
    """Forward dataflow fixpoint over the tagged edge list.

    Args:
        initial: the INPUT keys — keys guaranteed available at invoke (reduced
            channels, schema keys no node writes, and user-declared `inputs`).
            Seeding with these (not all schema keys) is what makes the may/must
            analysis catch read-before-written on COMPUTED keys.
        edges: list of `(src, dst, kind)` where kind is "seq" (AND,
            unconditional) or "cond" (OR, exclusive — DecisionNode choices and
            the Send fan-out).
        writes: `{node_name: set(keys_written)}`.
        send_workers: `{worker_name: forwarded_keys}` — a Send worker sees ONLY
            the forwarded payload, not global state.
        spread_collectors: `{collector_name: (spread_pred_name, worker_name)}` —
            a collector's input is the state just before the spread plus the
            worker's writes.

    Returns:
        `(may, must)` dicts: `{node_name: set(available_keys)}`.
    """
    send_workers = send_workers or {}
    spread_collectors = spread_collectors or {}

    names = set()
    for s, d, _ in edges:
        names.add(s)
        names.add(d)
    names.discard(END)

    # Loop-back edges carry loop-carried keys that do NOT exist on the first
    # iteration; excluding them means a node's incoming availability is computed
    # over the acyclic condensation, so a read of a key produced ONLY by a
    # downstream node that loops back is correctly flagged (not credited as
    # available on first entry). Keys produced upstream-before-the-cycle, or
    # earlier in the same loop body via a forward edge, are unaffected.
    back = _back_edges(edges)
    seq_preds = {n: set() for n in names}
    cond_groups = {n: {} for n in names}  # dst -> {src: set(src)}  (one group per source)
    for s, d, kind in edges:
        if d == END or (s, d) in back:
            continue
        if kind == "seq":
            seq_preds[d].add(s)
        else:
            cond_groups[d].setdefault(s, set()).add(s)

    all_keys = set(initial)
    for w in writes.values():
        all_keys |= w

    def w_of(p):
        return writes.get(p, set())

    real = [n for n in names if n != START]
    may = {n: set() for n in names}
    must = {n: set(all_keys) for n in names}  # greatest-fixpoint init (top)
    may[START] = set(initial)
    must[START] = set(initial)

    # Iterate to a TRUE fixpoint: keep sweeping until a full pass changes
    # nothing. The safety cap is len(real_nodes) — the longest dependency chain
    # a key can travel is bounded by the node count, so that many sweeps always
    # suffice. (The old fixed bound of len(all_keys)+2 under-converged when a
    # graph had more nodes than distinct state keys — the common case, since
    # many nodes share `messages` — and set-hash iteration order is not
    # topological, so a key produced early but read far downstream needs up to
    # path-length sweeps.)
    for _ in range(max(len(real), 1)):
        changed = False
        for n in real:
            if n in send_workers:                       # Send worker: payload only
                nm = set(send_workers[n])
                nmust = set(send_workers[n])
            elif n in spread_collectors:                # collector: pre-spread state + worker writes
                prev, worker = spread_collectors[n]
                base_may = (may.get(prev, set(initial)) | w_of(prev))
                base_must = (must.get(prev, set(initial)) | w_of(prev))
                nm = set(initial) | base_may | w_of(worker)
                nmust = set(initial) | base_must | w_of(worker)
            else:
                nm = set(initial)
                nmust = set(initial)
                # ponytail: a decision->handler->merge handler reaches the merge
                # by a `seq` edge, so its writes union into must[merge] as if
                # unconditional — must over-approximates exclusivity here. Effect
                # is WARNING-only (a some-paths read may under-warn); it never
                # produces a wrong hard error (those come from `may`). Tracking
                # exclusivity by ancestry would tighten it; left out by design.
                for p in seq_preds[n]:                   # AND preds: union
                    nm |= may[p] | w_of(p)
                    nmust |= must[p] | w_of(p)
                for grp in cond_groups[n].values():      # OR group: union (may) / intersect (must)
                    gm = set()
                    inter = None
                    for p in grp:
                        nm_p = may[p] | w_of(p)
                        gm |= nm_p
                        s2 = must[p] | w_of(p)
                        inter = s2 if inter is None else (inter & s2)
                    nm |= gm
                    if inter:
                        nmust |= inter
            if nm != may[n]:
                may[n] = nm
                changed = True
            if nmust != must[n]:
                must[n] = nmust
                changed = True
        if not changed:
            break
    return may, must


def check_placeholders(name, node_prompt, scalar_reads: set, placeholders: set):
    """Cross-check a node's scalar reads against its `node_prompt` placeholders.

    A scalar read exists to be interpolated into the prompt. With no scalar
    reads the node skips `.format_map` entirely (brace-heavy prompts pass
    untouched), so this check is skipped. Otherwise:
      - Check A (hard error): every `{name}` placeholder must be a declared
        scalar read — an undeclared one is a guaranteed runtime `KeyError`.
      - Check B (warning): every scalar read should appear as a placeholder —
        an uninterpolated scalar read is a dead read.

    Args:
        scalar_reads: the node's reads that are NOT message-history channels.
        placeholders: `{name}` field names parsed from `node_prompt`.
    """
    if not scalar_reads:
        return []
    issues = []
    for ph in sorted(placeholders):
        if ph not in scalar_reads:
            issues.append(Issue("error", name,
                f"node_prompt references placeholder {{{ph}}} but {ph!r} is not a "
                f"declared scalar read {sorted(scalar_reads)}; this raises KeyError "
                f"when the prompt is interpolated"))
    for sr in sorted(scalar_reads):
        if sr not in placeholders:
            issues.append(Issue("warning", name,
                f"declares scalar read {sr!r} but node_prompt never interpolates "
                f"{{{sr}}}; the value is fetched and silently ignored (dead read)"))
    return issues


def collect_issues(graph_name, schema, reduced, node_io: dict, may, must,
                   concurrency_pairs: set, decision_choices: dict, end_with_children: set,
                   unreachable: set):
    """Run all checks and return a list of Issues. See module docstring / spec.

    Args:
        node_io: `{name: (reads, writes)}` for every real node.
        concurrency_pairs: set of frozenset({x, y}) node-name pairs that run in
            parallel (AND-parallel branches).
        decision_choices: `{decision_name: [(choice_name, has_child_bool)]}`.
        end_with_children: names of declared end nodes that still have children.
        unreachable: names of nodes not reachable from START.
    """
    issues = []

    # Which node(s) write each key — used to explain ordering errors (the key
    # exists, but its producer runs downstream / on a sibling branch).
    writers_by_key = {}
    for nm, (_, w) in node_io.items():
        for k in w:
            writers_by_key.setdefault(k, set()).add(nm)

    for name, (reads, writes) in node_io.items():
        avail_may = may.get(name, set())
        avail_must = must.get(name, set())
        # (a) reads must be available BEFORE this node runs.
        for r in reads:
            optional = r.endswith("?")
            key = r[:-1] if optional else r

            if optional:
                # Explicit opt-out: a missing/some-path optional read is at most a
                # warning (never fails the build), even if the key is undeclared.
                if key not in avail_may:
                    issues.append(Issue("warning", name,
                        f"reads optional key {key!r} which is never produced "
                        f"upstream or supplied as an input; available: {sorted(avail_may)}"))
                elif key not in avail_must:
                    issues.append(Issue("warning", name,
                        f"reads optional key {key!r} which is only produced on "
                        f"SOME paths; guaranteed on all paths: {sorted(avail_must)}"))
                continue

            if key not in schema:
                # Undeclared key (typo) — LangGraph would silently read nothing.
                issues.append(Issue("error", name,
                    f"reads {key!r} which is not declared in the state schema "
                    f"{sorted(schema)}; available keys here: {sorted(avail_may)}"))
            elif key in avail_may:
                if key not in avail_must:
                    issues.append(Issue("warning", name,
                        f"reads {key!r} which is only produced on SOME paths; "
                        f"guaranteed on all paths: {sorted(avail_must)}"))
                # else: produced on every path — ok.
            else:
                # Declared, but it is a COMPUTED key (some node writes it) that is
                # not available here: the producer runs downstream or on a sibling
                # branch -> a runtime KeyError waiting to happen.
                writers = sorted(writers_by_key.get(key, set()) - {name})
                if writers:
                    issues.append(Issue("error", name,
                        f"reads computed key {key!r} but no upstream node produces "
                        f"it before this node (produced by: {writers}, none of which "
                        f"are upstream); available keys here: {sorted(avail_may)}"))
                else:
                    issues.append(Issue("error", name,
                        f"reads {key!r} but no node produces it and it is not an "
                        f"input key; produce it upstream or declare it in inputs=... "
                        f"if you supply it at invoke(); available keys here: "
                        f"{sorted(avail_may)}"))
        # (b) writes must be declared in the schema
        for wkey in writes:
            if wkey not in schema:
                issues.append(Issue("error", name,
                    f"writes key {wkey!r} which is not declared in the state "
                    f"schema {sorted(schema)}; LangGraph silently drops "
                    f"unknown-key writes"))

    # (c) concurrent writes to a key with no reducer
    for pair in concurrency_pairs:
        x, y = sorted(pair)
        wx = node_io.get(x, (set(), set()))[1]
        wy = node_io.get(y, (set(), set()))[1]
        for key in (wx & wy):
            if key in schema and key not in reduced:
                issues.append(Issue("error", x,
                    f"concurrently writes {key!r} with node {y!r}, but {key!r} "
                    f"has no reducer in the schema; concurrent writes to a plain "
                    f"key raise InvalidUpdateError at runtime"))

    # (e) a decision choice with no connected node
    for dname, choices in decision_choices.items():
        for cname, has_child in choices:
            if not has_child:
                issues.append(Issue("error", dname,
                    f"choice {cname!r} has no connected node; "
                    f"wire it, e.g. `decision[{cname!r}] > some_node`"))

    # (f) declared end node that still has children, or an unreachable node
    for name in end_with_children:
        issues.append(Issue("warning", name,
            "is an end node but still has children; the children are unreachable "
            "from it (an end node terminates the graph)"))
    for name in unreachable:
        issues.append(Issue("warning", name, "is unreachable from the start node"))

    return issues
