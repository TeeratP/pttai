"""
Graph implementation for the Agentic Framework.
"""

from typing import Literal, Set, Union
from langgraph.graph import StateGraph, START, END, MessagesState
from agentic_framework.nodes import AgentNode, InputNode, DecisionNode

class AgenticGraph(StateGraph):
    """
    A graph implementation for managing agent and decision nodes in a workflow.
    
    This class extends the base Graph class to provide specialized handling of
    AgentNode and DecisionNode types, allowing for the construction of complex
    agent-based workflows.
    """
    
    def __init__(self, state: MessagesState, start_node, end_nodes, name: str = 'graph') -> None:
        """
        Initialize the AgenticGraph.
        
        Args:
            start_node: The initial node in the graph
            end_nodes: Set of nodes that represent terminal states
        """
        super().__init__(state)
        
        self.name = name
        self.start_node = start_node
        self.end_nodes = end_nodes if isinstance(end_nodes, (list, tuple, set)) else [end_nodes]
        self.child = None
        
        self._seen_nodes: Set = set()
        self.build_graph()
        self.compiled_graph = self.compile()
        
    def build_graph(self) -> None:
        """Build the graph structure starting from the start node."""
        self._build_graph(self.start_node, START)
        
    def __call__(self, state):
        """
        Process the current state through the graph.
        
        Args:
            state: Current conversation state containing message history
            
        Returns:
            Updated state after processing through the graph
        """
        return self.compiled_graph.invoke(state)
    
    def invoke(self, state):
        """
        Process the current state through the graph.
        
        Args:
            state: Current conversation state containing message history
            
        Returns:
            Updated state after processing through the graph
        """
        return self.compiled_graph.invoke(state)
        
    def compile(self, checkpointer = None, *, store = None, interrupt_before = None, interrupt_after = None, debug = False):
        return super().compile(checkpointer, store=store, interrupt_before=interrupt_before, interrupt_after=interrupt_after, debug=debug)
            
    def _repr_mimebundle_(self, **kwargs):
        """
        Provide a MIME bundle representation of the compiled graph for Jupyter.
        """
        if not hasattr(self, 'compiled_graph') or self.compiled_graph is None:
            return {
                "text/plain": "The graph has not been compiled yet."
            }

        # Check if the compiled graph has its own `_repr_mimebundle_` method
        if hasattr(self.compiled_graph, '_repr_mimebundle_'):
            return self.compiled_graph._repr_mimebundle_(**kwargs)

        # Fallback if no visualization is available
        return {
            "text/plain": "The compiled graph does not support visualization."
        }
        
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
        
        if isinstance(node, AgentNode) or isinstance(node, InputNode):
            
            self.add_node(curr_node_name, node)
            if not isinstance(prev_node, DecisionNode):  # normal case
                self.add_edge(prev_node_name, curr_node_name)
            # else: edge from decision node is already added in DecisionNode
            
            if node in self.end_nodes:
                self.add_edge(curr_node_name, END)  # only case that creates forward edge is when node is end node
            else:
                self._build_graph(node.child, node)  # recursively build graph
            
        elif isinstance(node, DecisionNode):
            
            self.add_node(curr_node_name, node)
            self.add_edge(prev_node_name, curr_node_name)
            self.add_conditional_edges(curr_node_name, node.route, [x.child.name for x in node.choices])  # node performs as function that returns name of next node
            for choice in node.choices:
                self._build_graph(choice.child, node)
        
        elif isinstance(node, AgenticGraph):
            
            self.add_node(curr_node_name, node.compiled_graph)
            if not isinstance(prev_node, DecisionNode):  # normal case
                self.add_edge(prev_node_name, curr_node_name)
            # else: edge from decision node is already added in DecisionNode
            
            if node in self.end_nodes:
                self.add_edge(curr_node_name, END)  # only case that creates forward edge is when node is end node
            else:
                self._build_graph(node.child, node)  # recursively build graph
            
        else:
            raise TypeError(f"Unsupported node type: {type(node)}. Must be either AgentNode or DecisionNode")

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