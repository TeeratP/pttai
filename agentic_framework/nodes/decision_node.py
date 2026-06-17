"""
Decision node implementation for the Agentic Framework.
"""

from typing import Any, Dict, List, Literal, Optional, Union
from agentic_framework.node import Node
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
        self.child = other
        return other
        
class DecisionNode(Node):
    """
    A node that makes decisions based on LLM output to direct graph flow.
    
    DecisionNode uses a language model to choose between predefined options,
    determining the next node in the graph based on the choice made.
    """
    
    def __init__(self, 
                 name: str = 'decision_node', 
                 llm: Optional[Any] = None, 
                 node_prompt: str = "", 
                 choices: List[str] = []) -> None:
        """
        Initialize a DecisionNode.
        
        Args:
            name: Unique identifier for the node
            llm: Language model instance to be used by this node
            node_prompt: System prompt/instructions for the language model
            choices: List of possible decision options
            
        Raises:
            AssertionError: If node_prompt or choices is empty
        """
        assert node_prompt, "DecisionNode requires a node_prompt to be set."
        assert choices, "DecisionNode requires choices to be set."
        super().__init__(name, llm, node_prompt)
        self.choices_name: List[str] = choices
        self._create_choices()
        
    def _create_choices(self) -> None:
        """
        Initialize the choices and configure LLM for structured output.
        """
        class OutputModel(BaseModel):
            choice: Literal[tuple(self.choices_name)]
            
        self.llm = self.llm.with_structured_output(OutputModel)
        self.choices = [Choice(name) for name in self.choices_name]
        self.choices_created = True
        
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

        if 'messages' not in state:
            raise ValueError("State must contain a 'messages' key")

        message_w_prompt = [SystemMessage(content=self.node_prompt)] + state['messages']
        response = self.llm.invoke(message_w_prompt)  # use llm to decide which choice to make
        choice = response.choice
        # Routing label is written to the dedicated `decision` field, not injected
        # into `messages`, so it does not pollute the conversation history.
        return {"decision": choice, "log": [f'{self.name}:{choice}']}

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
        Prevent direct edge creation from DecisionNode.
        
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
