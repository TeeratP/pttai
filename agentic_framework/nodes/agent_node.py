"""
Agent node implementation for the Agentic Framework.
"""
from typing import Any, Optional, List
from agentic_framework.node import Node
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import StructuredTool, BaseTool
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
                 node_prompt: str = "you are a helpful assistant",
                 max_tool_iterations: int = 25,
                 input_field: str = "messages",
                 output_field: str = "messages",
                 reasoning_effort: Optional[str] = None,
                 cache_ttl: Optional[int] = None,
                 retry: bool = False) -> None:
        """
        Initialize an AgentNode.

        Args:
            name: Unique identifier for the node
            llm: Language model instance to be used by this node
            node_prompt: System prompt/instructions for the language model
            max_tool_iterations: Safety cap on the internal tool-call loop; a
                model that keeps requesting tools beyond this raises RuntimeError.
            input_field: State key to read the message history from.
            output_field: State key to write to. When "messages" (default) the
                produced messages are appended to the conversation; any other key
                receives the final response's string content instead (useful for
                transform nodes — that key must exist in your state schema).
            reasoning_effort: Reasoning effort for reasoning-capable models
                (e.g. "low"/"medium"/"high" on gpt-5.x). Passed as a per-call
                kwarg to the LLM. (DecisionNode does not expose this — reasoning
                effort conflicts with structured output on current OpenAI models.)
            cache_ttl/retry: see Node — node-level caching/retry.
        """
        super().__init__(name, llm, node_prompt, cache_ttl=cache_ttl, retry=retry)
        self.child = None
        self.tool_available = False
        self.max_tool_iterations = max_tool_iterations
        self.input_field = input_field
        self.output_field = output_field
        self.reasoning_effort = reasoning_effort
        # Per-call kwarg (survives bind_tools, unlike a pre-bound reasoning_effort).
        self._invoke_kwargs = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}

    def __call__(self, state):
        """
        Process the current state and generate a response.

        Runs the agent and, when tools are bound, its internal tool-call loop,
        accumulating every message produced this turn (the AIMessage, any
        ToolMessages, and follow-up AIMessages) into a single delta. Returns
        only the delta — the state reducers append it to canonical state.

        Args:
            state: Current conversation state containing message history

        Returns:
            A state delta: {"messages": [...new...], "log": [...new...]}

        Raises:
            ValueError: If LLM is not set or if state is invalid
        """
        if self.llm is None:
            raise ValueError(f"{self.name} requires a LLM to be set before call.")

        if self.input_field not in state:
            raise ValueError(f"State must contain a {self.input_field!r} key")

        history = state[self.input_field]
        new_messages = []
        new_log = []

        prompt = [SystemMessage(content=self.node_prompt)] + history
        response = self.llm.invoke(prompt, **self._invoke_kwargs)
        new_messages.append(response)
        new_log.append(f'{self.name}:{response.content}')

        if self.tool_available:
            iterations = 0
            while isinstance(new_messages[-1], AIMessage) and getattr(new_messages[-1], "tool_calls", None):
                if iterations >= self.max_tool_iterations:
                    raise RuntimeError(
                        f"{self.name} exceeded max_tool_iterations={self.max_tool_iterations}; "
                        "the model kept requesting tools without finishing."
                    )
                iterations += 1
                ai_message = new_messages[-1]
                for tool_call in ai_message.tool_calls:
                    tool_result = self.tools_by_name[tool_call["name"]].invoke(
                        tool_call["args"]
                    )
                    new_messages.append(
                        ToolMessage(
                            content=json.dumps(tool_result),
                            name=tool_call["name"],
                            tool_call_id=tool_call["id"],
                        )
                    )
                    new_log.append(f'tools:{tool_call["name"]}, args:{tool_call["args"]}, result:{tool_result}')
                prompt = [SystemMessage(content=self.node_prompt)] + history + new_messages
                response = self.llm.invoke(prompt, **self._invoke_kwargs)
                new_messages.append(response)
                new_log.append(f'{self.name}:{response.content}')

        if self.output_field == "messages":
            return {"messages": new_messages, "log": new_log}
        # Transform node: write the final response content to the named field.
        return {self.output_field: new_messages[-1].content, "log": new_log}

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
        self.llm = self.llm.bind_tools(self.tools)
