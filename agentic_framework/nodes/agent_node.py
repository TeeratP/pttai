"""
Agent node implementation for the Agentic Framework.
"""

from hmac import new
from typing import Any, Dict, Optional, Union
from agentic_framework.node import Node
from langchain_core.messages import SystemMessage
from agentic_framework.state import AgenticState

class AgentNode(Node):
    """
    A node that represents an agent capable of processing messages and generating responses.
    
    AgentNode uses a language model to process incoming messages and generate appropriate
    responses based on its configured prompt and system message.
    """
    
    def __init__(self, 
                 name: str = 'agent_node', 
                 llm: Optional[Any] = None, 
                 node_prompt: str = "you are a helpful assistant") -> None:
        """
        Initialize an AgentNode.
        
        Args:
            name: Unique identifier for the node
            llm: Language model instance to be used by this node
            node_prompt: System prompt/instructions for the language model
        """
        super().__init__(name, llm, node_prompt)
        self.child = None
        
    def __call__(self, state):
        """
        Process the current state and generate a response.
        
        Args:
            state: Current conversation state containing message history
            
        Returns:
            Updated state with the agent's response appended
            
        Raises:
            ValueError: If LLM is not set or if state is invalid
        """
        if self.llm is None:
            raise ValueError(f"{self.name} requires a LLM to be set before call.")
        
        if 'messages' not in state:
            raise ValueError("State must contain a 'messages' key")
            
        message_w_prompt = state['messages']
        message_w_prompt = [SystemMessage(content=self.node_prompt)] + message_w_prompt
        response = self.llm.invoke(message_w_prompt)
        new_state = state
        new_state['messages'].append(response)
        if 'log' in new_state:
            new_state['log'].append(f'{self.name}:{response.content}')
        
        return new_state
    
    def __gt__(self, other):
        """
        Create edge from this node to another node.
        
        Args:
            other: The node to create an edge to
            
        Returns:
            The other node to allow for chain building
        """
        self.child = other
        return other
