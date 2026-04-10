"""
Agent node implementation for the Agentic Framework.
"""
from typing import Any, Optional, List
from agentic_framework.node import Node
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langchain.tools import StructuredTool, BaseTool
import json

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
        self.tool_available = False
        
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
        
        state['messages'].append(response)
        if 'log' in state:
            state['log'].append(f'{self.name}:{response.content}')
        
        if self.tool_available:
            state = self.call_tools(state)
        
        return state
    
    def bind_tools(self, tools):
        """
        Bind a tool to the agent node.
        
        Args:
            tools: list of tools to bind to the agent node
        """
        self.tools = []
        
        if not isinstance(tools, List):
            tools = [tools]
        
        for tool in tools:
            
            # if tool is already a langgraph tool, use it as is
            if isinstance(tool, StructuredTool) or isinstance(tool, BaseTool):
                pass
            
            # if tool is function, use StructuredTool to wrap it
            elif callable(tool):
                tool = StructuredTool.from_function(
                        func = tool, 
                        name = tool.__name__, 
                        description = tool.__doc__)
                
            self.tools.append(tool)
        
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.tool_available = True
        self.llm = self.llm.bind_tools(tools)

    def call_tools(self, state):
        """
        check if the ai_message has a call to a tool and process it
        then call llm again until no more calls to tools are found
        
        Args:
            state: Current conversation state containing message history
            
        Returns:
            Updated state with the agent's response appended
        """ 
        ai_message = state['messages'][-1]
        
        if not isinstance(ai_message, AIMessage):
            return state
        
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            for tool_call in ai_message.tool_calls:
                tool_result = self.tools_by_name[tool_call["name"]].invoke(
                    tool_call["args"]
                )
                
                tool_output = ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
                
                state['messages'].append(tool_output)
                
                if 'log' in state:
                    state['log'].append(f'tools:{tool_call["name"]}, args:{tool_call["args"]}, result:{tool_result}')
                
            state = self.__call__(state)
        
        return state