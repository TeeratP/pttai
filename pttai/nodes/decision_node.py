"""LLM-driven routing nodes for pttai.

Defines the routing machinery shared by every branching node: `Choice` (a named
branch you wire with `decision["label"] > handler`), the `RouterNode` mixin
(owns the choice list, the `route()` conditional-edge callback, and the wiring
operators), and `DecisionNode` — the LLM router that returns a constrained
`Literal[*choices]` structured output so the model can only pick a valid branch.
`ConditionNode` (the code-only sibling) reuses `RouterNode` from its own module.
"""

from typing import Any, List, Literal, Optional
from pttai.node import Node, Branch, Spread
from pttai.nodes.llm_node import LLMNode
from pttai.nodes._fields import partition_reads
from langchain_core.messages import SystemMessage
from pydantic import BaseModel

class Choice:
    """
    Represents a decision option in a DecisionNode.

    Each Choice has a name and can be connected to another node in the graph.
    """

    def __init__(self, name: str):
        """
        Initialize a Choice.

        Args:
            name: Unique identifier for this choice
        """
        self.name = name
        self.child = None

    def __gt__(self, other):
        """
        Create an edge from this choice to another node.

        Args:
            other: The node to create an edge to

        Returns:
            The other node to allow for chain building
        """
        # ponytail: v1 — a decision choice can't host a Send/fan-out directly
        # (the choice edge is already conditional; nesting a Spread/Branch fan-out
        # under it isn't supported). Wrap the fan-out in an AgenticGraph instead.
        if isinstance(other, (Spread, Branch)):
            raise ValueError(
                "A decision choice cannot route directly to a fan-out/`.map`; "
                "wrap it in an AgenticGraph and route to that.")
        self.child = other
        return other

class RouterNode:
    """
    Routing mixin for nodes that pick the next node instead of appending to the
    conversation. Owns the `Choice` list, `route()`, the `node["choice"] >
    handler` wiring (`__getitem__`), and the `__gt__` override that forbids a
    direct edge off the router.

    It is a MIXIN (not a Node subclass): concrete routers combine it with their
    real base — `ConditionNode(RouterNode, Node)` and
    `DecisionNode(RouterNode, LLMNode)` — and call `_setup_choices` explicitly
    from their own `__init__`. Subclasses decide HOW the label is chosen
    (`DecisionNode` via an LLM, `ConditionNode` via a Python callable) and
    both write it to a dedicated per-node `decision_{name}` state field (read by
    `route()`), never into `messages`. Because `RouterNode` stays in the MRO of both,
    `graph.py`'s `isinstance(node, RouterNode)` conditional-edge handling
    treats them identically.
    """

    def _setup_choices(self,
                       choices: List[str],
                       input_field: str = "messages",
                       reads: Optional[List[str]] = None) -> None:
        """Initialize the routing state shared by every router subclass.

        Concrete routers call this from their own `__init__` (RouterNode is a
        mixin, not a base with its own `__init__`). Stores the choice-name list,
        the read config (`reads` wins over `input_field`), and builds one
        `Choice` object per name so `node["label"] > handler` can wire it.
        """
        assert choices, "a router requires choices"
        self.choices_name: List[str] = list(choices)
        self.input_field = input_field
        self.reads = reads if reads is not None else [input_field]
        self.choices = [Choice(name) for name in self.choices_name]

    def route(self, state):
        """Conditional-edge callback: map the chosen label to the next node name.

        Registered with LangGraph via `add_conditional_edges(name, route, ...)`.
        Reads the label this router wrote to `state["decision_{name}"]`, finds
        the matching `Choice`, and returns the name of the node wired to it —
        which LangGraph uses to pick the outgoing edge. Raises `ValueError` if
        the label is unknown or its choice was never wired to a node.
        """
        decision_choice = state[f"decision_{self.name}"]
        for choice in self.choices:
            if decision_choice == choice.name:
                if choice.child is None:
                    raise ValueError(
                        f"Choice {decision_choice} does not have a child. "
                        "Please create an edge from this choice to another node. "
                        "For example: `decision_node['abc'] > other_node`"
                    )
                return choice.child.name
        raise ValueError(f"Choice {decision_choice} not found in choices")

    def __gt__(self, other) -> None:
        """
        Prevent direct edge creation from a routing node.

        Raises:
            ValueError: Always, since edges must be created from choices
        """
        raise ValueError(
            "You must create edge from choices to other nodes. "
            "For example: `decision_node['choice'] > other_node`"
        )

    def __getitem__(self, key: str) -> Choice:
        """
        Get a Choice object by its name.

        Args:
            key: Name of the choice to retrieve

        Returns:
            The Choice object

        Raises:
            ValueError: If the choice name is not found
        """
        for choice in self.choices:
            if key == choice.name:
                return choice
        raise ValueError(f"Choice {key} not found in choices.")


class DecisionNode(RouterNode, LLMNode):
    """
    A node that makes decisions based on LLM output to direct graph flow.

    DecisionNode uses a language model to choose between predefined options,
    determining the next node in the graph based on the choice made.

    Tool use is two-phase so tools NEVER share a call with structured output: the
    base model is bound as a separate `_tool_llm` (used to gather context via
    the shared tool-call loop) and `_route_llm` is the structured-output model
    that returns the forced `choice`.

    Wire it by indexing a choice — `decision["positive"] > handler`; a bare
    `decision > x` raises.

    Examples:
        ```python
        from pttai import AgentNode, DecisionNode, AgenticGraph

        classify = DecisionNode(
            llm=llm,
            node_prompt="Classify the sentiment of the message.",
            choices=["positive", "negative"],
        )
        praise = AgentNode(llm=llm, node_prompt="Thank the happy customer.")
        apologize = AgentNode(llm=llm, node_prompt="Apologize to the unhappy customer.")

        classify["positive"] > praise
        classify["negative"] > apologize

        graph = AgenticGraph(start_node=classify, end_nodes={praise, apologize})
        ```
    """

    def __init__(self,
                 name: Optional[str] = None,
                 llm: Optional[Any] = None,
                 node_prompt: str = "",
                 choices: List[str] = [],
                 tools: Optional[list] = None,
                 input_field: str = "messages",
                 reads: Optional[List[str]] = None,
                 cache_ttl: Optional[int] = None,
                 retry: bool = False) -> None:
        """
        Initialize a DecisionNode.

        Args:
            name: Unique identifier for the node
            llm: Language model instance to be used by this node
            node_prompt: System prompt/instructions for the language model
            choices: List of possible decision options
            tools: Optional tools the node may call to gather context BEFORE
                routing. Same normalization as AgentNode. Tools run in a first
                phase (a tool-call loop on a tool-bound copy of the model); the
                gathered messages then feed the structured-output routing call.
                Tools and structured output never happen in one call.
            input_field: State key to read the message history from.
            reads: State keys this node reads (multi-key form; back-compat
                generalization of input_field). Reads are dispatched by VALUE
                type just like AgentNode — message lists become history, scalars
                are interpolated into node_prompt. Use reads OR input_field.
            cache_ttl: see [Node][pttai.node.Node] — node-level result caching.
            retry: see [Node][pttai.node.Node] — node-level retry on exception.

        Raises:
            AssertionError: If node_prompt or choices is empty
        """
        assert node_prompt, "DecisionNode requires a node_prompt to be set."
        assert choices, "DecisionNode requires choices to be set."
        if llm is None:
            raise ValueError("DecisionNode requires an llm — pass llm=... to the constructor.")
        LLMNode.__init__(self, name=name, llm=llm, node_prompt=node_prompt, tools=tools,
                         cache_ttl=cache_ttl, retry=retry)
        self._setup_choices(choices, input_field=input_field, reads=reads)
        # Force the model to return exactly one valid choice via structured output.
        class OutputModel(BaseModel):
            choice: Literal[tuple(self.choices_name)]
        # Keep the base llm; derive the two single-purpose models. `_tool_llm`
        # gathers context (tools only); `_route_llm` returns the forced choice
        # (structured output only). No single call combines tools + structure.
        self._route_llm = self.llm.with_structured_output(OutputModel)
        self._tool_llm = self.llm.bind_tools(self.tools) if self.tools else None

    def __call__(self, state: dict) -> dict:
        """
        Process the current state and make a decision.

        Args:
            state: Current conversation state containing message history

        Returns:
            A state delta: {f"decision_{name}": <chosen label>, "log": [...]}

        Raises:
            ValueError: If LLM is not set, if choice is invalid, or if choice has no connected node
        """
        if self.llm is None:
            raise ValueError(f"{self.name} requires a LLM to be set before call.")

        # Partition reads by value type: message lists -> history, rest -> scalars.
        history, scalars = partition_reads(state, self.reads)
        sys = self.node_prompt.format_map(scalars) if scalars else self.node_prompt

        if self.tools:
            # Phase 1: gather context by running the tool-call loop on the
            # tool-bound model. Phase 2: route on the gathered context.
            gathered, tool_log, _ = self._run_tool_loop(self._tool_llm, sys, history)
            prompt = [SystemMessage(content=sys)] + history + gathered
            choice = self._route_llm.invoke(prompt).choice
            return {f"decision_{self.name}": choice, "log": tool_log + [f'{self.name}:{choice}']}

        message_w_prompt = [SystemMessage(content=sys)] + history
        response = self._route_llm.invoke(message_w_prompt)  # use llm to decide which choice to make
        choice = response.choice
        # Routing label is written to the dedicated per-node `decision_{name}` field,
        # not injected into `messages`, so it does not pollute the conversation history.
        return {f"decision_{self.name}": choice, "log": [f'{self.name}:{choice}']}
