"""
Input node implementation for the Agentic Framework.
"""
from typing import Any, Optional, List

from pttai.node import Node
from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

class InputNode(Node):
    def __init__(self,
                 name: str = 'input_node',
                 node_prompt: str = "Please review the following message and provide feedback.",
                 n: int = 1,
                 cache_ttl: Optional[int] = None,
                 retry: bool = False,
                 ) -> None:
        """
        Initialize an InputNode.

        Args:
            name: Unique identifier for the node
            cache_ttl/retry: see Node — node-level caching/retry.
        """
        super().__init__(name, cache_ttl=cache_ttl, retry=retry)
        self.node_prompt = node_prompt
        self.n = n
        
    def __call__(self, state):
        
        if 'messages' not in state:
            raise ValueError("State must contain a 'messages' key")
        if self.n <= 0:
            disp_msg = ""
        else:
            disp_msg = state["messages"][-self.n].content
            
        human_message = interrupt(
                {
                    self.node_prompt: disp_msg,
                }
            )

        return {"messages": [HumanMessage(content=human_message)]}
    