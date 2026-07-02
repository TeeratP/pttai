"""Node base class and the `>`-DSL wiring primitives.

Home of `Node` — the abstract base every node type subclasses — plus the objects
the wiring operators return: `Branch` (from `a > [b, c]` and `fanout(...)`, whose
own `>` wires the fan-in/join) and `Spread` (from `worker.map(field)`, the
map-reduce wrapper). Wiring is deferred: `a > b` just records `.children`
pointers in memory; `AgenticGraph` walks them at construction and emits the
actual LangGraph edges. This module also holds `_infer_name`, the best-effort
inference of a node's name from its assignment target, and the `fanout` helper.
"""

import ast
import linecache
import sys
from abc import ABC, abstractmethod
from typing import Any, Optional


# Memoized ast trees keyed by SOURCE CONTENT hash (not filename/length: a re-run
# notebook cell / REPL reuses the same filename at the same line count with
# changed content, so a length-based key would return a stale tree).
_AST_CACHE: dict = {}


def _infer_name():
    """Best-effort: the LHS variable of `x = SomeNode(...)` / `self.x = ...`.
    Returns the name str, or None when it can't be determined (caller falls
    back to build-time numbering). Never raises."""
    try:
        frame = sys._getframe(1)
        # climb past pttai's own __init__ frames to the user's call site
        while frame is not None and frame.f_globals.get("__name__", "").startswith("pttai"):
            frame = frame.f_back
        if frame is None:
            return None
        source = "".join(linecache.getlines(frame.f_code.co_filename, frame.f_globals))
        if not source:
            return None
        lineno = frame.f_lineno
        key = hash(source)
        tree = _AST_CACHE.get(key)
        if tree is None:
            tree = ast.parse(source)
            _AST_CACHE[key] = tree
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and getattr(node, "lineno", None) is not None \
                    and node.lineno <= lineno <= (node.end_lineno or node.lineno) \
                    and len(node.targets) == 1:
                tgt = node.targets[0]
                if isinstance(tgt, ast.Name):
                    return tgt.id
                if isinstance(tgt, ast.Attribute):
                    return tgt.attr
            if isinstance(node, ast.AnnAssign) and getattr(node, "lineno", None) is not None \
                    and node.lineno <= lineno <= (node.end_lineno or node.lineno) \
                    and isinstance(node.target, ast.Name):
                return node.target.id
    except Exception:
        return None
    return None


class Node(ABC):
    """
    Abstract base class for all nodes in the agent framework.
    
    This class defines the basic interface that all node types must implement,
    providing core functionality for language model integration and prompt management.
    """
    
    def __init__(self, name: Optional[str] = None, llm: Optional[Any] = None, node_prompt: str = "",
                 cache_ttl: Optional[int] = None, retry: bool = False) -> None:
        """
        Initialize a new Node.

        Args:
            name: Unique identifier for the node. OPTIONAL — when omitted, the
                node's name is inferred from the assignment target at the call
                site (`reviewer = AgentNode()` -> "reviewer"); the graph resolves
                it (with collision suffixing) at build time. An explicit name
                always wins. `_name_base` keeps the unresolved base so a rebuild
                is idempotent; `_auto_name` flags an inferred/numbered name.
            llm: Language model instance to be used by this node
            node_prompt: System prompt/instructions for the language model
            cache_ttl: Seconds to cache this node's result (LangGraph CachePolicy).
            retry: When True, retry this node on exception (LangGraph RetryPolicy).
        """
        if name is not None:
            self._name_base = name
            self._auto_name = False
        else:
            self._name_base = _infer_name()
            self._auto_name = True
        self.name = self._name_base  # may be None until the graph resolves it
        self.llm = llm
        self.node_prompt = node_prompt
        self.cache_ttl = cache_ttl
        self.retry = retry
        self.children: list = []
        self._head = self  # head of the chain this node is currently the tail of

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute the node's primary function.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Node execution result
            
        Raises:
            ValueError: If LLM is not set
        """
        pass
    
    def __gt__(self, other: "Node | list | Branch") -> "Node | Branch":
        """
        Wire this node forward to one child, or fan out to several.

        Args:
            other: A single node (sequential edge), a list of nodes, or a
                `Branch` (parallel fan-out). Branch members may be chain
                *tails* (e.g. `b>c>d`); each fans out from its chain *head*.

        Returns:
            The single child (chain building), or the `Branch` whose own
            `>` wires the fan-in/join.
        """
        if isinstance(other, Branch):       # a > fanout(b>c, d)
            self.children = [t._head for t in other.tails]   # a -> each branch HEAD
            return other
        if isinstance(other, list):
            self.children = [m._head for m in other]          # a -> each branch HEAD
            return Branch(other)          # group whose __gt__ wires the join
        other._head = self._head          # propagate chain head downstream
        self.children = [other]
        return other

    def map(self, field):
        """Run this node once per item of `state[field]` in parallel (map-reduce).

        Use as `a > worker.map("items") > collector`: the worker fans out via
        LangGraph `Send` over the items, then all replies join into the
        collector (which runs once, after).

        A mapped worker's scalar write key becomes a LIST channel — one entry per
        item (a 1-element list for a single item; arrival order, not input order)
        — and `state=` is not required: the channel and its accumulate reducer
        are inferred at build time.

        v1 limitation: workers return their REPLY only, not the source item (the
        `Send` payload is the worker's input, never a state update).
        """
        return Spread(self, field)

    def __lt__(self, other):
        """Fan-in join: wire a list of branch tails INTO this node.

        This is the reflected half of `[tails] > self`. Python evaluates
        `x > [a, b] > self` as a chained comparison `(x > [a,b]) and ([a,b] > self)`;
        the second term is `list.__gt__(self)`, which returns `NotImplemented`,
        so Python falls back to `self.__lt__([a, b])` — landing here. It points
        every tail's child at `self` and marks `self` a join, mirroring
        `Branch.__gt__`. Only `[..] > node` is supported (not `[..] > [..]`).
        """
        # Reflected from `[tails] > self` (Python chained comparison
        # `x > [a,b] > self` calls list.__gt__ -> NotImplemented -> self.__lt__(list)).
        # Wires the branch tails as a fan-in/join into self. Mirrors Branch.__gt__.
        # ponytail: only handles `[..] > node`. `[b,c] > [d,e]` (fan-out straight
        # into fan-out) is a real list<list compare, not our wiring — put a node
        # between the two fan-outs.
        if isinstance(other, list):
            for t in other:
                t.children = [self]
            self._is_join = True
            return self
        return NotImplemented


class Branch:
    """Group returned by `node > [a, b, ...]`; its members are branch *tails*,
    and its `>` wires the fan-in/join."""

    def __init__(self, tails: list) -> None:
        """Hold the fan-out branches (`tails`) so the next `>` can join them.

        Args:
            tails: The branch nodes (each may be a chain tail such as `b > c`).
        """
        self.tails = tails

    def __gt__(self, join):
        """Wire every branch tail into `join` (the fan-in).

        Points each tail's child at `join` and marks `join` a deferred join so it
        runs once, after all branches complete (see `graph.py`). Returns `join`
        so the chain continues.
        """
        for t in self.tails:
            t.children = [join]      # tail -> join
        join._is_join = True   # marks it for defer (see graph.py)
        return join


class Spread:
    """Map-reduce wrapper: `worker.map(field)` fans `worker` across
    `state[field]` via LangGraph `Send`, joining into the collector wired by
    `> collector`. Built specially in `AgenticGraph._build_graph` — a Spread
    has no graph identity of its own; the worker is the real node."""

    def __init__(self, worker: "Node", field: str) -> None:
        """Capture the map-reduce intent from `worker.map(field)`.

        Args:
            worker: The node to run once per item of `state[field]`.
            field: The state key whose items are fanned out (one `Send` each).
        """
        self.worker = worker
        self.field = field
        self.collector = None
        self.children = []      # set to [collector] so the DAG walk reaches it
        self._head = self       # so `a > spread` (single path) can set _head

    def __gt__(self, collector):
        """Wire the map's fan-in: all worker replies join into `collector`.

        Records `collector` and marks it a deferred join so it runs exactly once,
        after every `Send`-fanned worker has finished. Returns `collector`.
        """
        self.collector = collector
        self.children = [collector]
        collector._is_join = True   # collector defers (runs once after all sends)
        return collector


def fanout(*tails):
    """Explicit parallel fan-out: `a > fanout(b, c) > d` is equivalent to
    `a > [b, c] > d`. Returns a Branch (a single object), so it composes in a
    chained `>` without relying on list semantics. Members may be chain tails
    (e.g. `fanout(b>c>d, e>f)`).

    The branches run concurrently; the join node (`d`) is deferred and runs once,
    after every branch finishes.

    Examples:
        ```python
        from pttai import AgentNode, AgenticGraph, fanout

        start = AgentNode(llm=llm, node_prompt="Restate the task.")
        pros = AgentNode(llm=llm, node_prompt="List the pros.")
        cons = AgentNode(llm=llm, node_prompt="List the cons.")
        combine = AgentNode(llm=llm, node_prompt="Weigh the pros and cons.")

        start > fanout(pros, cons) > combine

        graph = AgenticGraph(start_node=start, end_nodes={combine})
        ```
    """
    return Branch(list(tails))