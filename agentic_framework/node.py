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
    
    def __init__(self, name: str, llm: Optional[Any] = None, node_prompt: str = "") -> None:
        """
        Initialize a new Node.
        
        Args:
            name: Unique identifier for the node
            llm: Language model instance to be used by this node
            node_prompt: System prompt/instructions for the language model
        """
        self.name = name
        self.llm = llm
        self.node_prompt = node_prompt

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
        if self.llm is None:
            raise ValueError(f"{self.name} requires a LLM to be set.")
    
    def set_llm(self, llm: Any) -> None:
        """
        Set the language model for this node.
        
        Args:
            llm: Language model instance
        """
        self.llm = llm
