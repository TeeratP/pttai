"""
Decision node implementation for the Agentic Framework.
"""

from typing import Any, Dict, List, Literal, Optional, Union
from pttai.node import Node, Branch, Spread
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
        
class RouterNode(Node):
    """
    Base for routing nodes: owns the Choice list, ``route()``, and the
    ``node["choice"] > handler`` wiring. Subclasses decide HOW the label is
    chosen — ``DecisionNode`` via an LLM, ``ConditionNode`` via a Python callable.

    Both write the chosen label to the dedicated ``decision`` state field (read
    by ``route()``), never into ``messages``. Graph build/routing sites treat
    every ``RouterNode`` identically (see graph.py).
    """

    def __init__(self,
                 name: Optional[str] = None,
                 choices: List[str] = None,
                 llm: Optional[Any] = None,
                 node_prompt: str = "",
                 input_field: str = "messages",
                 reads: Optional[List[str]] = None,
                 cache_ttl: Optional[int] = None,
                 retry: bool = False) -> None:
        assert choices, "a router requires choices"
        super().__init__(name, llm, node_prompt, cache_ttl=cache_ttl, retry=retry)
        self.choices_name: List[str] = list(choices)
        self.input_field = input_field
        self.reads = reads if reads is not None else [input_field]
        self.choices = [Choice(name) for name in self.choices_name]

    def route(self, state):

        decision_choice = state['decision']
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


class DecisionNode(RouterNode):
    """
    A node that makes decisions based on LLM output to direct graph flow.

    DecisionNode uses a language model to choose between predefined options,
    determining the next node in the graph based on the choice made.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 llm: Optional[Any] = None,
                 node_prompt: str = "",
                 choices: List[str] = [],
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
            input_field: State key to read the message history from.
            reads: State keys this node reads (multi-key form; back-compat
                generalization of input_field). Reads are dispatched by VALUE
                type just like AgentNode — message lists become history, scalars
                are interpolated into node_prompt. Use reads OR input_field.
            cache_ttl/retry: see Node — node-level caching/retry.

        Raises:
            AssertionError: If node_prompt or choices is empty
        """
        assert node_prompt, "DecisionNode requires a node_prompt to be set."
        assert choices, "DecisionNode requires choices to be set."
        super().__init__(name, choices, llm=llm, node_prompt=node_prompt,
                         input_field=input_field, reads=reads,
                         cache_ttl=cache_ttl, retry=retry)
        # Force the model to return exactly one valid choice via structured output.
        class OutputModel(BaseModel):
            choice: Literal[tuple(self.choices_name)]
        self.llm = self.llm.with_structured_output(OutputModel)

    def __call__(self, state):
        """
        Process the current state and make a decision.
        
        Args:
            state: Current conversation state containing message history
            
        Returns:
            A state delta: {"decision": <chosen label>, "log": [...]}

        Raises:
            ValueError: If LLM is not set, if choice is invalid, or if choice has no connected node
        """
        if self.llm is None:
            raise ValueError(f"{self.name} requires a LLM to be set before call.")

        # Partition reads by value type: message lists -> history, rest -> scalars.
        history, scalars = partition_reads(state, self.reads)
        sys = self.node_prompt.format_map(scalars) if scalars else self.node_prompt
        message_w_prompt = [SystemMessage(content=sys)] + history
        response = self.llm.invoke(message_w_prompt)  # use llm to decide which choice to make
        choice = response.choice
        # Routing label is written to the dedicated `decision` field, not injected
        # into `messages`, so it does not pollute the conversation history.
        return {"decision": choice, "log": [f'{self.name}:{choice}']}
