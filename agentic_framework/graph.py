"""
Graph implementation for the Agentic Framework.
"""

from typing import Set, Union
from langgraph.graph import Graph, START, END
from agentic_framework.nodes.agent_node import AgentNode
from agentic_framework.nodes.decision_node import DecisionNode

class AgenticGraph(Graph):
    """
    A graph implementation for managing agent and decision nodes in a workflow.
    
    This class extends the base Graph class to provide specialized handling of
    AgentNode and DecisionNode types, allowing for the construction of complex
    agent-based workflows.
    """
    
    def __init__(self, start_node: Union[AgentNode, DecisionNode], end_nodes: Set[AgentNode]):
        """
        Initialize the AgenticGraph.
        
        Args:
            start_node: The initial node in the graph
            end_nodes: Set of nodes that represent terminal states
        """
        super().__init__()
        
        self.start_node = start_node
        self.end_nodes = end_nodes
        
        self.registered_node: Set = set()
        self.build_graph()
        
    def build_graph(self) -> None:
        """Build the graph structure starting from the start node."""
        self._build_graph(self.start_node, START)
        
    def _build_graph(self, node: Union[AgentNode, DecisionNode], prev_node: Union[str, AgentNode, DecisionNode]) -> None:
        """
        Recursively build the graph structure.
        
        Args:
            node: Current node being processed
            prev_node: Previous node in the graph
            
        Raises:
            TypeError: If node is neither AgentNode nor DecisionNode
        """
        if node in self.registered_node:
            return
        
        self.registered_node.add(node)
        
        if isinstance(node, AgentNode):
            self.add_node(node, node.name)
            if isinstance(prev_node, DecisionNode):  # normal case
                self.add_edge(prev_node, node.name)
            # else: edge from decision node is already added in DecisionNode
            
            if node in self.end_nodes:
                self.add_edge(node, END)  # only case that creates forward edge is when node is end node
            else:
                self._build_graph(node.child, node)  # recursively build graph
            
        elif isinstance(node, DecisionNode):
            self.add_conditional_edges(prev_node, node)  # node performs as function that returns name of next node
            for choice in node.choices:
                self._build_graph(choice.child, node)
        else:
            raise TypeError(f"Unsupported node type: {type(node)}. Must be either AgentNode or DecisionNode")
