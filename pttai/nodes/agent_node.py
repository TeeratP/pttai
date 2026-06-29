"""
Agent node implementation for the Agentic Framework.
"""
from typing import Any, Optional, List
from pttai.node import Node
from pttai.nodes._fields import partition_reads
from pttai.state import merge_token_usage
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import StructuredTool, BaseTool
from pydantic import create_model
import json


def _usage_delta(response):
    """Return ``{model_name: usage_metadata}`` for one LLM response, or ``{}``
    when it carries no ``usage_metadata`` (fakes / structured-output path)."""
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return {}
    model = (getattr(response, "response_metadata", None) or {}).get("model_name", "unknown")
    return {model: usage}


def _is_openai(llm) -> bool:
    """Duck-type whether ``llm`` is a langchain_openai ChatOpenAI, WITHOUT
    importing langchain_openai. Unwraps the RunnableBinding that ``bind_tools``
    produces so detection still works after tools are bound."""
    target = getattr(llm, "bound", llm)  # RunnableBinding -> underlying model
    cls = type(target)
    return cls.__name__ == "ChatOpenAI" or cls.__module__.startswith("langchain_openai")

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
                 reads: Optional[List[str]] = None,
                 writes: Optional[List[str]] = None,
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
            reads: State keys this node reads (multi-key form; back-compat
                generalization of input_field). Each read is dispatched by VALUE
                type: a non-empty list of messages becomes conversation history
                (concatenated in order), anything else is a scalar interpolated
                into node_prompt via .format_map. Use reads OR input_field.
            writes: State keys this node writes (multi-key form; back-compat
                generalization of output_field). ["messages"] appends to the
                conversation; a single scalar key writes the final response
                content; two or more scalar keys switch to structured output (a
                str field per key, no tool loop). Use writes OR output_field.
            reasoning_effort: Reasoning effort for reasoning-capable models
                (e.g. "low"/"medium"/"high" on gpt-5.x). Passed as a per-call
                kwarg to the LLM. (DecisionNode does not expose this — reasoning
                effort conflicts with structured output on current OpenAI models.)
            cache_ttl/retry: see Node — node-level caching/retry.
        """
        super().__init__(name, llm, node_prompt, cache_ttl=cache_ttl, retry=retry)
        self.tool_available = False
        self.max_tool_iterations = max_tool_iterations
        self.input_field = input_field
        self.output_field = output_field
        # reads/writes win if both forms are given (document: use one or the other).
        self.reads = reads if reads is not None else [input_field]
        self.writes = writes if writes is not None else [output_field]
        self.reasoning_effort = reasoning_effort
        # Per-call kwarg (survives bind_tools, unlike a pre-bound reasoning_effort).
        self._invoke_kwargs = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
        # OpenAI prompt-cache routing — set by AgenticGraph when prompt_cache=True.
        self._prompt_cache_enabled = False
        self._prompt_cache_key = None

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

        # Partition reads by value type: message lists -> history, rest -> scalars.
        history, scalars = partition_reads(state, self.reads)
        new_messages = []
        new_log = []

        # Only interpolate when there is a scalar read, so prompts containing
        # literal braces (e.g. JSON) with no scalars pass through verbatim.
        sys = self.node_prompt.format_map(scalars) if scalars else self.node_prompt

        scalar_writes = [w for w in self.writes if w != "messages"]
        if "messages" in self.writes and scalar_writes:
            raise ValueError(
                "a node writes either the conversation ('messages') or "
                "structured fields, not both")

        prompt = [SystemMessage(content=sys)] + history

        # >=2 scalar writes -> structured output (str field per key, no tool loop,
        # no reasoning_effort — mirrors DecisionNode).
        # ponytail: all structured fields are typed `str` in v1; a future
        # `output_model: type[BaseModel]` param is the escape hatch for
        # typed/nested output — not built now.
        if len(scalar_writes) >= 2:
            Model = create_model("Out", **{w: (str, ...) for w in scalar_writes})
            out = self.llm.with_structured_output(Model).invoke(prompt)
            values = {w: getattr(out, w) for w in scalar_writes}
            return values | {"log": [f"{self.name}:{values}"]}

        # Per-call kwargs: reasoning_effort (if set) plus an OpenAI
        # prompt_cache_key when prompt caching is on AND the model is OpenAI.
        invoke_kwargs = dict(self._invoke_kwargs)
        if self._prompt_cache_enabled and _is_openai(self.llm):
            invoke_kwargs["prompt_cache_key"] = self._prompt_cache_key

        response = self.llm.invoke(prompt, **invoke_kwargs)
        new_messages.append(response)
        new_log.append(f'{self.name}:{response.content}')
        token = _usage_delta(response)

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
                prompt = [SystemMessage(content=sys)] + history + new_messages
                response = self.llm.invoke(prompt, **invoke_kwargs)
                new_messages.append(response)
                new_log.append(f'{self.name}:{response.content}')
                token = merge_token_usage(token, _usage_delta(response))

        if not scalar_writes:  # writes == ["messages"] (default)
            delta = {"messages": new_messages, "log": new_log}
        else:
            # Transform node: write the final response content to the single field.
            delta = {scalar_writes[0]: new_messages[-1].content, "log": new_log}
        if token:  # only emit when at least one call reported usage_metadata
            delta["token"] = token
        return delta

    def bind_tools(self, tools):
        """
        Bind a tool to the agent node.

        Args:
            tools: list of tools to bind to the agent node
        """
        # Structured output preempts the free-form tool loop on OpenAI/LangGraph.
        if len([w for w in self.writes if w != "messages"]) >= 2:
            raise ValueError(
                "multi-field structured output (writes=[...]) cannot be combined "
                "with bind_tools in v1")

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
