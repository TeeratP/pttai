"""
Base node implementation for the Agentic Framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

class Node(ABC):
    """
    Abstract base class for all nodes in the agent framework.
    
    This class defines the basic interface that all node types must implement,
    providing core functionality for language model integration and prompt management.
    """
    
    def __init__(self, name: str, llm: Optional[Any] = None, node_prompt: str = "",
                 cache_ttl: Optional[int] = None, retry: bool = False) -> None:
        """
        Initialize a new Node.

        Args:
            name: Unique identifier for the node
            llm: Language model instance to be used by this node
            node_prompt: System prompt/instructions for the language model
            cache_ttl: Seconds to cache this node's result (LangGraph CachePolicy).
            retry: When True, retry this node on exception (LangGraph RetryPolicy).
        """
        self.name = name
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
    
    def set_llm(self, llm: Any) -> None:
        """
        Set the language model for this node.
        
        Args:
            llm: Language model instance
        """
        self.llm = llm

    def __gt__(self, other):
        """
        Wire this node forward to one child, or fan out to several.

        Args:
            other: A single node (sequential edge), a list of nodes, or a
                ``Branch`` (parallel fan-out). Branch members may be chain
                *tails* (e.g. ``b>c>d``); each fans out from its chain *head*.

        Returns:
            The single child (chain building), or the ``Branch`` whose own
            ``>`` wires the fan-in/join.
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
        """Run this node once per item of ``state[field]`` in parallel (map-reduce).

        Use as ``a > worker.map("items") > collector``: the worker fans out via
        LangGraph ``Send`` over the items, then all replies join into the
        collector (which runs once, after).

        v1 limitations: (1) workers return their REPLY only, not the source item
        (the ``Send`` payload is the worker's input, never a state update);
        (2) map workers MUST output ``messages`` (the default) — a worker with a
        non-message ``output_field``/scalar write would make N parallel workers
        write the same plain key, raising ``InvalidUpdateError``.
        """
        return Spread(self, field)

    def __lt__(self, other):
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

    def __init__(self, tails):
        self.tails = tails

    def __gt__(self, join):
        for t in self.tails:
            t.children = [join]      # tail -> join
        join._is_join = True   # marks it for defer (see graph.py)
        return join


class Spread:
    """Map-reduce wrapper: ``worker.map(field)`` fans ``worker`` across
    ``state[field]`` via LangGraph ``Send``, joining into the collector wired by
    ``> collector``. Built specially in ``AgenticGraph._build_graph`` — a Spread
    has no graph identity of its own; the worker is the real node."""

    def __init__(self, worker, field):
        self.worker = worker
        self.field = field
        self.collector = None
        self.children = []      # set to [collector] so the DAG walk reaches it
        self._head = self       # so `a > spread` (single path) can set _head

    def __gt__(self, collector):
        self.collector = collector
        self.children = [collector]
        collector._is_join = True   # collector defers (runs once after all sends)
        return collector


def fanout(*tails):
    """Explicit parallel fan-out: `a > fanout(b, c) > d` is equivalent to
    `a > [b, c] > d`. Returns a Branch (a single object), so it composes in a
    chained `>` without relying on list semantics. Members may be chain tails
    (e.g. `fanout(b>c>d, e>f)`)."""
    return Branch(list(tails))