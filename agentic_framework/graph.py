"""
Graph implementation for the Agentic Framework.
"""

from typing import Literal, Set, Union
from langgraph.graph import StateGraph, START, END, MessagesState
from agentic_framework.nodes.agent_node import AgentNode
from agentic_framework.nodes.decision_node import DecisionNode

class AgenticGraph(StateGraph):
    """
    A graph implementation for managing agent and decision nodes in a workflow.
    
    This class extends the base Graph class to provide specialized handling of
    AgentNode and DecisionNode types, allowing for the construction of complex
    agent-based workflows.
    """
    
    def __init__(self, state: MessagesState, start_node, end_nodes) -> None:
        """
        Initialize the AgenticGraph.
        
        Args:
            start_node: The initial node in the graph
            end_nodes: Set of nodes that represent terminal states
        """
        super().__init__(state)
        
        self.start_node = start_node
        self.end_nodes = end_nodes
        
        self._seen_nodes: Set = set()
        self.build_graph()
        
    def build_graph(self) -> None:
        """Build the graph structure starting from the start node."""
        self._build_graph(self.start_node, START)
        
    def _build_graph(self, node, prev_node) -> None:
        """
        Recursively build the graph structure.
        
        Args:
            node: Current node being processed
            prev_node: Previous node in the graph
            
        Raises:
            TypeError: If node is neither AgentNode nor DecisionNode
        """
        prev_node_name = prev_node if prev_node == START else prev_node.name
        curr_node_name = node.name
        
        if curr_node_name in self._seen_nodes:
            return
        
        self._seen_nodes.add(curr_node_name)
        
        if isinstance(node, AgentNode):
                        
            self.add_node(curr_node_name, node)
            if not isinstance(prev_node, DecisionNode):  # normal case
                self.add_edge(prev_node_name, curr_node_name)
            # else: edge from decision node is already added in DecisionNode
            
            if node in self.end_nodes:
                self.add_edge(curr_node_name, END)  # only case that creates forward edge is when node is end node
            else:
                self._build_graph(node.child, node)  # recursively build graph
            
        elif isinstance(node, DecisionNode):
            self.add_conditional_edges(prev_node_name, node, [x.child.name for x in node.choices])  # node performs as function that returns name of next node
            for choice in node.choices:
                self._build_graph(choice.child, node)
        else:
            raise TypeError(f"Unsupported node type: {type(node)}. Must be either AgentNode or DecisionNode")
