"""
Condition node implementation for the Agentic Framework.

The deterministic, code sibling of `DecisionNode`: routes by a Python
predicate over state instead of an LLM call — free, deterministic, no prompt.
"""

from typing import Callable, List, Optional
from pttai.node import Node
from pttai.nodes.decision_node import RouterNode


class ConditionNode(RouterNode, Node):
    """Routing by a Python predicate over state — no LLM, no prompt, free and
    deterministic. `condition(state)` must return one of `choices`.

    The code sibling of [DecisionNode][pttai.nodes.decision_node.DecisionNode]:
    reach for it whenever the branch is decidable in Python (a length check, a
    loop counter, a flag another node wrote) — you get zero-cost, reproducible
    routing with no model call. Wire it by indexing a choice, exactly like a
    `DecisionNode`: `route["short"] > handler`. The chosen label is written to
    the per-node `decision_{name}` channel, never into `messages`.

    Examples:
        ```python
        from pttai import AgentNode, ConditionNode, AgenticGraph

        def is_short(state):
            return "short" if len(state["messages"][-1].content) < 20 else "long"

        route = ConditionNode(condition=is_short, choices=["short", "long"])
        brief = AgentNode(llm=llm, node_prompt="Give a one-line answer.")
        detailed = AgentNode(llm=llm, node_prompt="Give a thorough answer.")

        route["short"] > brief
        route["long"] > detailed

        graph = AgenticGraph(start_node=route, end_nodes={brief, detailed})
        ```
    """

    def __init__(self,
                 name: Optional[str] = None,
                 condition: Optional[Callable] = None,
                 choices: List[str] = [],
                 reads: Optional[List[str]] = None,
                 cache_ttl: Optional[int] = None,
                 retry: bool = False) -> None:
        """
        Initialize a ConditionNode.

        Args:
            name: Unique identifier for the node
            condition: Callable `condition(state) -> str` returning one of
                `choices` (the routing label).
            choices: List of possible routing labels.
            reads: State keys this node reads — default `[]` (the callable
                reads state directly). Pass `reads=[...]` so the validator can
                check those keys are available where this node runs.
            cache_ttl: see [Node][pttai.node.Node] — node-level result caching.
            retry: see [Node][pttai.node.Node] — node-level retry on exception.

        Raises:
            AssertionError: If condition or choices is empty
        """
        assert condition is not None, "ConditionNode requires a condition callable"
        assert choices, "ConditionNode requires choices to be set."
        Node.__init__(self, name, cache_ttl=cache_ttl, retry=retry)
        self._setup_choices(choices, reads=reads if reads is not None else [])
        self.condition = condition

    def __call__(self, state):
        """Evaluate the predicate and emit the routing label.

        Calls `condition(state)`, validates the result is one of `choices`
        (raising `ValueError` otherwise), and writes it to the per-node
        `decision_{name}` field that `route()` reads. Returns only the delta.
        """
        label = self.condition(state)
        if label not in self.choices_name:
            raise ValueError(
                f"{self.name}: condition returned {label!r}, not in choices {self.choices_name}")
        # Routing label written to the dedicated per-node `decision_{name}` field
        # (read by route()), never into `messages` — same contract as DecisionNode.
        return {f"decision_{self.name}": label, "log": [f"{self.name}:{label}"]}
