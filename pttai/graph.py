"""
Graph implementation for the Agentic Framework.
"""

import sys
import uuid
from typing import Any, Literal, Union, TypedDict, get_type_hints
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import CachePolicy, RetryPolicy, Send
from langgraph.cache.memory import InMemoryCache
from langchain_core.messages import HumanMessage
from pttai.node import Branch, Spread
from pttai.nodes import AgentNode, InputNode, DecisionNode
from pttai.nodes._fields import prompt_placeholders, is_history_annotation
from pttai.state import AgenticState
from pttai.validation import (
    GraphValidationError, ValidationReport, Issue, schema_keys, reduced_keys,
    compute_availability, collect_issues, check_placeholders,
)


def _augment_schema(base, extra_keys):
    """Return a TypedDict like ``base`` plus a PLAIN (last-writer-wins) channel
    for each key in ``extra_keys``. Existing reduced channels keep their
    reducers (their resolved ``Annotated[..., reducer]`` annotations are copied
    verbatim). Returns ``base`` unchanged when there is nothing to add."""
    if not extra_keys:
        return base
    hints = dict(get_type_hints(base, include_extras=True))
    for k in extra_keys:
        hints[k] = Any  # untyped plain channel (no reducer metadata)
    return TypedDict(f"{base.__name__}Plus", hints)


def _make_send_fn(worker, field):
    """Conditional-edge function: fan one Send per item of state[field] to the
    worker. Each Send carries a COMPLETE 1-item input state for that invocation."""
    def send(state):
        return [Send(worker.name, {"messages": [HumanMessage(content=str(item))], "log": []})
                for item in state[field]]
    return send

class AgenticGraph(StateGraph):
    """
    A graph implementation for managing agent and decision nodes in a workflow.
    
    This class extends the base Graph class to provide specialized handling of
    AgentNode and DecisionNode types, allowing for the construction of complex
    agent-based workflows.
    """
    
    def __init__(self, state=None, start_node=None, end_nodes=None, name: str = 'graph',
                 checkpointer=None, cache=None, validate: bool = True,
                 inputs=None) -> None:
        """
        Initialize the AgenticGraph.

        Args:
            state: The state schema (a TypedDict). OPTIONAL — defaults to the
                standard ``AgenticState`` (messages/log/decision). Any scalar key
                a node declares in ``writes`` that the schema doesn't already
                declare is auto-registered as a PLAIN (last-writer-wins) channel,
                so you rarely need a hand-written schema. Pass an explicit
                ``state=`` (with reducers) when you want to type/reduce a key
                precisely.
            start_node: The initial node in the graph
            end_nodes: Set of nodes that represent terminal states
            checkpointer: Optional LangGraph checkpointer (e.g. InMemorySaver).
                Required for InputNode interrupt/resume. When set, every invoke
                must pass a config with a ``thread_id``.
            cache: Optional LangGraph cache backend. Defaults to an InMemoryCache
                when any node sets ``cache_ttl``; pass explicitly to override.
            inputs: Keys you pass at ``invoke()`` that are PLAIN (non-reduced) AND
                written by some node — declare them here so the validator treats
                them as provided at entry, not produced. (Reduced channels and
                keys no node writes are already inferred as inputs; you only need
                this for a plain key you seed at invoke that a node also writes.)
            validate: When True (default), statically check that every key each
                node reads is produced upstream/in the schema and every key it
                writes is declared, FAILING the build (GraphValidationError) on a
                hard error. Set False to skip (escape hatch). See ``validate()``.
        """
        if state is None:
            state = AgenticState
        self.start_node = start_node
        self.end_nodes = end_nodes if isinstance(end_nodes, (list, tuple, set)) else [end_nodes]
        # Auto-register every undeclared key a node WRITES as a plain channel, so
        # users don't have to predeclare scalars in a TypedDict. (Reads are NOT
        # auto-registered: a read of a key no node writes and that the schema
        # doesn't declare stays a hard error — the dangling-read guardrail.)
        state = _augment_schema(state, self._auto_register_keys(state))

        super().__init__(state)

        self.name = name
        self.state_schema = state
        self.children: list = []
        self._head = self  # head of the chain this graph is currently the tail of
        self.checkpointer = checkpointer
        self.inputs = set(inputs) if inputs else set()

        self._seen_nodes: dict = {}
        self._edges_added: set = set()
        # validation bookkeeping, populated during the build (see _build_graph):
        self._edges: list = []            # (src, dst, kind) tagged edge list
        self._edges_seen: set = set()     # dedup for _edges
        self._send_workers: dict = {}     # worker_name -> forwarded keys
        self._spread_collectors: dict = {}  # collector_name -> (spread_pred, worker)
        self._spread_fields: list = []    # (spread_pred_name, field) — field is read here
        self.build_graph()
        # Compute this graph's free reads / schema-bound writes so a PARENT graph
        # can treat it as a single node (subgraph composition).
        self._compute_subgraph_io()

        if validate:
            report = self.validate()
            if not report.ok:
                raise GraphValidationError(str(report))

        if cache is None and any(getattr(n, "cache_ttl", None) for n in self._seen_nodes.values()):
            cache = InMemoryCache()
        self.cache = cache
        self.compiled_graph = self.compile(checkpointer=checkpointer, cache=cache)
        
    def build_graph(self) -> None:
        """Build the graph structure starting from the start node."""
        self._mark_parallel_joins()
        self._build_graph(self.start_node, START)
        
    def __call__(self, state, config=None, durability=None):
        """
        Process the current state through the graph.

        Args:
            state: Initial state, or a ``Command`` (e.g. ``Command(resume=...)``)
                to resume an interrupted run.
            config: Optional LangGraph config, e.g.
                ``{"configurable": {"thread_id": "abc"}}`` (required when a
                checkpointer is set).

        Returns:
            Updated state after processing through the graph
        """
        return self.compiled_graph.invoke(state, config=config, durability=durability)

    def invoke(self, state, config=None, durability=None):
        """
        Process the current state through the graph.

        Args:
            state: Initial state, or a ``Command`` (e.g. ``Command(resume=...)``)
                to resume an interrupted run.
            config: Optional LangGraph config, e.g.
                ``{"configurable": {"thread_id": "abc"}}`` (required when a
                checkpointer is set).

        Returns:
            Updated state after processing through the graph
        """
        return self.compiled_graph.invoke(state, config=config, durability=durability)

    def stream(self, state, config=None, durability=None, **kwargs):
        """
        Stream graph execution, yielding updates as nodes complete.

        Passes through to the compiled graph's stream (default mode='updates').
        """
        return self.compiled_graph.stream(state, config=config, durability=durability, **kwargs)

    async def ainvoke(self, state, config=None, durability=None):
        """Async variant of invoke. Sync nodes run in LangGraph's threadpool."""
        return await self.compiled_graph.ainvoke(state, config=config, durability=durability)

    async def astream(self, state, config=None, durability=None, **kwargs):
        """Async streaming variant of stream."""
        async for chunk in self.compiled_graph.astream(state, config=config, durability=durability, **kwargs):
            yield chunk

    def compile(self, checkpointer = None, *, cache = None, store = None, interrupt_before = None, interrupt_after = None, debug = False):
        return super().compile(checkpointer, cache=cache, store=store, interrupt_before=interrupt_before, interrupt_after=interrupt_after, debug=debug)
            
    def _repr_mimebundle_(self, **kwargs):
        """
        Provide a MIME bundle representation of the compiled graph for Jupyter.
        """
        if not hasattr(self, 'compiled_graph') or self.compiled_graph is None:
            return {
                "text/plain": "The graph has not been compiled yet."
            }

        # Check if the compiled graph has its own `_repr_mimebundle_` method
        if hasattr(self.compiled_graph, '_repr_mimebundle_'):
            return self.compiled_graph._repr_mimebundle_(**kwargs)

        # Fallback if no visualization is available
        return {
            "text/plain": "The compiled graph does not support visualization."
        }
        
    def _node_policies(self, node) -> dict:
        """Build add_node policy kwargs (cache/retry) from node attrs."""
        kw = {}
        if getattr(node, "cache_ttl", None):
            kw["cache_policy"] = CachePolicy(ttl=node.cache_ttl)
        if getattr(node, "retry", False):
            kw["retry_policy"] = RetryPolicy()
        return kw

    def _add_edge(self, src, tgt) -> None:
        """Add a static edge once; a fan-in join is reached from several parents."""
        if (src, tgt) not in self._edges_added:
            self._edges_added.add((src, tgt))
            self.add_edge(src, tgt)
            self._record_edge(src, tgt, "seq")

    def _record_edge(self, src, tgt, kind) -> None:
        """Append a tagged edge for the validation dataflow (deduped)."""
        if (src, tgt, kind) not in self._edges_seen:
            self._edges_seen.add((src, tgt, kind))
            self._edges.append((src, tgt, kind))

    def _mark_parallel_joins(self) -> None:
        """Flag fan-in joins so they defer to the end of the Pregel super-step.

        Counts static (non-decision) in-edges per node over the ``.children``
        DAG; a node with in-degree >= 2 is a fan-in join and gets
        ``_defer = True`` (so an unbalanced diamond fires it once, not once per
        arriving branch). A DecisionNode's out-edges are conditional, so they
        are excluded from the count — a node reached only as a choice target is
        not a static join. ``Branch.__gt__`` may also flag a join explicitly
        via ``_is_join``. A visited set keeps cycles/diamonds finite.
        """
        indeg: dict = {}
        nodes: dict = {}

        def walk(node):
            if node.name in nodes:
                return
            nodes[node.name] = node
            if isinstance(node, DecisionNode):
                for choice in node.choices:  # conditional out-edges: not counted
                    if choice.child is not None:
                        walk(choice.child)
            else:
                for child in node.children:
                    # A Spread (`worker.map(...)`) has no graph identity: its
                    # Send fan-out and the worker->collector edge are not static
                    # in-edges, so don't count them. The collector already
                    # defers via _is_join. Recurse straight into the collector.
                    if isinstance(child, Spread):
                        if child.collector is not None:
                            walk(child.collector)
                        continue
                    indeg[child.name] = indeg.get(child.name, 0) + 1
                    walk(child)

        walk(self.start_node)
        for name, node in nodes.items():
            node._defer = indeg.get(name, 0) >= 2 or getattr(node, "_is_join", False)

    def _build_graph(self, node, prev_node) -> None:
        """
        Recursively build the graph structure.
        
        Args:
            node: Current node being processed
            prev_node: Previous node in the graph
            
        Raises:
            TypeError: If node is neither AgentNode nor DecisionNode
        """
        prev_node_name = prev_node if prev_node == START else prev_node.name

        if isinstance(node, Spread):
            worker, field, collector = node.worker, node.field, node.collector
            if collector is None:
                raise ValueError("worker.map(...) must be followed by `> collector`.")
            # guard: the Spread must be the SOLE child of its predecessor
            if prev_node is not START and getattr(prev_node, "children", None) != [node]:
                raise ValueError(
                    f"A `.map(...)` spread must be the only child of {prev_node_name!r}; "
                    "it cannot share a source with other branches.")
            # register the worker as a real node (idempotent via _seen_nodes)
            if worker.name not in self._seen_nodes:
                self._seen_nodes[worker.name] = worker
                self.add_node(worker.name, worker, **self._node_policies(worker))
            # dynamic fan-out: prev --Send--> worker (conditional edge; Send needs it)
            self.add_conditional_edges(prev_node_name, _make_send_fn(worker, field), [worker.name])
            self._record_edge(prev_node_name, worker.name, "cond")
            # A Send worker sees ONLY the forwarded payload, not global state; its
            # collector's input is the pre-spread state plus the worker's writes.
            self._send_workers[worker.name] = {"messages", "log"}
            self._spread_collectors[collector.name] = (prev_node_name, worker.name)
            # `field` is read from state where the spread fires (at prev's output).
            self._spread_fields.append((prev_node_name, field))
            # worker --> collector (static); recurse so collector + downstream build normally
            self._build_graph(collector, worker)
            return

        curr_node_name = node.name

        if curr_node_name in self._seen_nodes:
            if self._seen_nodes[curr_node_name] is not node:
                raise ValueError(
                    f"Duplicate node name {curr_node_name!r}: two distinct nodes share "
                    "this name. Node names must be unique within a graph."
                )
            # Fan-in revisit: the node is already built, but this parent's join
            # edge is not yet wired. Add it (decision edges are conditional).
            if prev_node is not START and not isinstance(prev_node, DecisionNode):
                self._add_edge(prev_node_name, curr_node_name)
            return

        self._seen_nodes[curr_node_name] = node

        if isinstance(node, AgentNode) or isinstance(node, InputNode):

            self.add_node(curr_node_name, node, defer=getattr(node, '_defer', False), **self._node_policies(node))
            if not isinstance(prev_node, DecisionNode):  # normal case
                self._add_edge(prev_node_name, curr_node_name)
            # else: edge from decision node is already added in DecisionNode

            if node in self.end_nodes:
                self._add_edge(curr_node_name, END)  # only case that creates forward edge is when node is end node
            elif not node.children:
                raise ValueError(f"Node {curr_node_name!r} has no children and is not an end node.")
            else:
                for child in node.children:  # recursively build graph
                    self._build_graph(child, node)

        elif isinstance(node, DecisionNode):

            self.add_node(curr_node_name, node, **self._node_policies(node))
            self._add_edge(prev_node_name, curr_node_name)
            # Only wire choices that are connected; a None child is reported as a
            # hard error by validate() (instead of crashing the build here).
            wired = [x for x in node.choices if x.child is not None]
            self.add_conditional_edges(curr_node_name, node.route, [x.child.name for x in wired])  # node performs as function that returns name of next node
            for choice in wired:
                self._record_edge(curr_node_name, choice.child.name, "cond")
                self._build_graph(choice.child, node)

        elif isinstance(node, AgenticGraph):

            self.add_node(curr_node_name, node.compiled_graph, defer=getattr(node, '_defer', False))
            if not isinstance(prev_node, DecisionNode):  # normal case
                self._add_edge(prev_node_name, curr_node_name)
            # else: edge from decision node is already added in DecisionNode

            if node in self.end_nodes:
                self._add_edge(curr_node_name, END)  # only case that creates forward edge is when node is end node
            elif not node.children:
                raise ValueError(f"Node {curr_node_name!r} has no children and is not an end node.")
            else:
                for child in node.children:  # recursively build graph
                    self._build_graph(child, node)

        else:
            raise TypeError(f"Unsupported node type: {type(node)}. Must be either AgentNode or DecisionNode")

    # --- validation / introspection -------------------------------------

    def _collect_nodes(self, start) -> dict:
        """Walk the ``.children`` / choice / spread graph from ``start`` and
        return ``{name: node}`` for every node, BEFORE the StateGraph is built
        (so key auto-registration can run before ``super().__init__``)."""
        seen: dict = {}

        def children_of(node):
            if isinstance(node, DecisionNode):
                return [c.child for c in node.choices if c.child is not None]
            return list(getattr(node, "children", None) or [])

        def walk(node):
            if isinstance(node, Spread):
                worker = node.worker
                if worker.name not in seen:
                    seen[worker.name] = worker
                    for ch in children_of(worker):
                        walk(ch)
                if node.collector is not None:
                    walk(node.collector)
                return
            name = node.name
            if name in seen:
                return
            seen[name] = node
            for ch in children_of(node):
                walk(ch)

        if start is not None:
            walk(start)
        return seen

    def _auto_register_keys(self, base_state) -> set:
        """Keys WRITTEN by some node that ``base_state`` does not declare. These
        become plain channels so the build doesn't reject undeclared writes."""
        base = schema_keys(base_state)
        extra = set()
        for node in self._collect_nodes(self.start_node).values():
            _, writes = self._node_io(node)
            extra |= writes
        return extra - base

    def _node_io(self, node):
        """Per-node (reads, writes) key sets for the dataflow analysis."""
        if isinstance(node, AgenticGraph):
            return set(node._io_reads), set(node._io_writes)
        if isinstance(node, DecisionNode):
            return set(node.reads), {"decision", "log"}
        if isinstance(node, InputNode):
            return {"messages"}, {"messages"}
        if isinstance(node, AgentNode):
            return set(node.reads), set(node.writes) | {"log"}
        return set(), set()  # ponytail: unknown node type — no IO claims

    def _compute_subgraph_io(self) -> None:
        """Free reads / schema-bound writes, so a PARENT can treat this graph as
        one node. Free reads = internal reads not produced by any internal node."""
        all_reads, all_writes = set(), set()
        for n in self._seen_nodes.values():
            reads, writes = self._node_io(n)
            all_reads |= {r[:-1] if r.endswith("?") else r for r in reads}
            all_writes |= writes
        schema = schema_keys(self.state_schema)
        self._io_writes = all_writes & schema
        self._io_reads = all_reads - all_writes

    def _input_keys(self, schema, reduced, writes):
        """Keys guaranteed available at invoke (the dataflow seed).

        ``input_keys = reduced ∪ (schema − written) ∪ inputs``:
          - a reduced channel (messages/log/…) is seeded/accumulated at invoke;
          - a schema key NO node writes must be supplied at invoke;
          - ``inputs`` lets the user declare extra plain entry keys they seed.
        Everything else (plain keys written by ≥1 node) is COMPUTED and must be
        produced upstream of any reader.
        """
        written = set()
        for w in writes.values():
            written |= w
        return reduced | (schema - written) | (self.inputs & schema)

    def _seq_adjacency(self):
        """Children adjacency over the static (AND) edges, excluding START/END."""
        adj = {}
        for s, d, kind in self._edges:
            if kind != "seq" or s == START or d == END:
                continue
            adj.setdefault(s, set()).add(d)
        return adj

    def _descendants(self, start, adj):
        seen = set()
        stack = [start]
        while stack:
            cur = stack.pop()
            for nxt in adj.get(cur, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    def _concurrency_pairs(self):
        """Node-name pairs that run in parallel (different branches of a static
        fan-out, neither downstream of the other). The shared join and post-join
        region are excluded by intersecting out keys reachable from sibling
        branches."""
        adj = self._seq_adjacency()
        pairs = set()
        for f, kids in adj.items():
            if len(kids) < 2:
                continue
            reach = {c: ({c} | self._descendants(c, adj)) for c in kids}
            for c in kids:
                others = set().union(*(reach[o] for o in kids if o is not c))
                exclusive = reach[c] - others
                reach[c] = exclusive
            kids_list = list(kids)
            for i in range(len(kids_list)):
                for j in range(i + 1, len(kids_list)):
                    for x in reach[kids_list[i]]:
                        for y in reach[kids_list[j]]:
                            if x != y:
                                pairs.add(frozenset((x, y)))
        return pairs

    def _reachable_from_start(self):
        adj = {}
        for s, d, _ in self._edges:
            adj.setdefault(s, set()).add(d)
        return {START} | self._descendants(START, adj)

    def validate(self, strict: bool = False, *, interrupt=None) -> ValidationReport:
        """Statically check state availability; returns a ValidationReport.

        Raises nothing itself (the build path raises on errors). With
        ``strict=True`` the returned report has its warnings promoted to errors.
        """
        # LangGraph's StateGraph.compile() calls self.validate(interrupt=...) for
        # its own structural check; delegate that to the base class.
        if interrupt is not None:
            return super().validate(interrupt=interrupt)

        schema = schema_keys(self.state_schema)
        reduced = reduced_keys(self.state_schema)
        node_io = {name: self._node_io(node) for name, node in self._seen_nodes.items()}
        writes = {name: w for name, (_, w) in node_io.items()}
        input_keys = self._input_keys(schema, reduced, writes)

        may, must = compute_availability(
            input_keys, self._edges, writes,
            send_workers=self._send_workers,
            spread_collectors=self._spread_collectors,
        )

        decision_choices = {
            name: [(c.name, c.child is not None) for c in node.choices]
            for name, node in self._seen_nodes.items()
            if isinstance(node, DecisionNode)
        }
        end_names = {n.name for n in self.end_nodes}
        end_with_children = [
            name for name, node in self._seen_nodes.items()
            if name in end_names and getattr(node, "children", None)
            and not isinstance(node, DecisionNode)
        ]
        reachable = self._reachable_from_start()
        unreachable = [name for name in self._seen_nodes if name not in reachable]

        issues = collect_issues(
            self.name, schema, reduced, node_io, may, must,
            self._concurrency_pairs(), decision_choices, end_with_children, unreachable,
        )

        # A `.map(...)` spread reads `field` from state where it fires (at the
        # predecessor's output). A declared field that is a COMPUTED key produced
        # downstream / on a sibling branch is a hard error, exactly like a node
        # read. (An undeclared field is out of scope here — see ponytail note.)
        for pred, field in self._spread_fields:
            if field not in schema:
                continue  # ponytail: undeclared spread fields aren't flagged
            avail = may.get(pred, set(input_keys)) | writes.get(pred, set())
            if field not in avail:
                producers = sorted(
                    n for n, (_, w) in node_io.items() if field in w)
                issues.append(Issue("error", pred,
                    f"map(...) spreads over {field!r} but it is not available "
                    f"here (produced by: {producers}, none of which are upstream); "
                    f"available keys here: {sorted(avail)}"))

        # Scalar reads must line up with node_prompt {placeholders} (Issue #4):
        # a placeholder with no matching scalar read is a runtime KeyError (hard
        # error); a scalar read never interpolated is a dead read (warning). A
        # node with no scalar reads skips .format_map entirely, so it is skipped.
        hints = get_type_hints(self.state_schema, include_extras=True)
        message_keys = {k for k, ann in hints.items() if is_history_annotation(ann)}
        for nm, node in self._seen_nodes.items():
            if not isinstance(node, (AgentNode, DecisionNode)):
                continue
            scalar_reads = {r[:-1] if r.endswith("?") else r
                            for r in node.reads} - message_keys
            issues.extend(check_placeholders(
                nm, node.node_prompt, scalar_reads, prompt_placeholders(node.node_prompt)))

        if strict:
            for i in issues:
                if i.level == "warning":
                    i.level = "error"
        return ValidationReport(issues, self.name)

    def summary(self, file=sys.stdout) -> None:
        """Print a Keras-``model.summary()``-style table of every node's reads,
        writes, and available (``must``) keys (``may``\\``must`` keys suffixed ``?``)."""
        schema = schema_keys(self.state_schema)
        reduced = reduced_keys(self.state_schema)
        node_io = {name: self._node_io(node) for name, node in self._seen_nodes.items()}
        writes = {name: w for name, (_, w) in node_io.items()}
        input_keys = self._input_keys(schema, reduced, writes)
        may, must = compute_availability(
            input_keys, self._edges, writes,
            send_workers=self._send_workers,
            spread_collectors=self._spread_collectors,
        )
        report = self.validate()

        def fmt(keys):
            return ",".join(sorted(keys)) or "-"

        def avail(name):
            m, mu = may.get(name, set()), must.get(name, set())
            return ",".join(
                k + ("?" if k not in mu else "") for k in sorted(m)
            ) or "-"

        rows = []
        for name, (reads, w) in node_io.items():
            rows.append((name, type(self._seen_nodes[name]).__name__,
                         fmt(reads), fmt(w), avail(name)))

        headers = ("node", "type", "reads", "writes", "available")
        widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
                  for i, h in enumerate(headers)]

        def line(cols):
            return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cols))

        print(f"AgenticGraph {self.name!r}   state={self.state_schema.__name__}", file=file)
        print(f"initial: {', '.join(sorted(schema))}", file=file)
        print("-" * (sum(widths) + 2 * (len(widths) - 1)), file=file)
        print(line(headers), file=file)
        for r in rows:
            print(line(r), file=file)
        print("-" * (sum(widths) + 2 * (len(widths) - 1)), file=file)
        print(f"{len(rows)} nodes · {len(report.errors)} errors · "
              f"{len(report.warnings)} warning(s)", file=file)
        for wline in report.warnings:
            print(f"  ! {wline}", file=file)

    def __gt__(self, other):
        """
        Wire this graph forward to one child, or fan out to several.

        Args:
            other: A single node (sequential edge), or a list of nodes
                (parallel fan-out — each becomes a child).

        Returns:
            The single child (chain building), or a ``Branch`` wrapping the
            fan-out heads whose own ``>`` wires the fan-in/join.
        """
        if isinstance(other, Branch):
            self.children = [t._head for t in other.tails]
            return other
        if isinstance(other, list):
            self.children = [m._head for m in other]
            return Branch(other)
        other._head = self._head
        self.children = [other]
        return other