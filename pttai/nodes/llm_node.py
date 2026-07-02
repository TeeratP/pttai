"""
LLMNode: shared base for nodes that call a language model.

Owns the LLM handle, tool normalization/registry, the tool-call loop, and the
OpenAI prompt-cache fields. `AgentNode` and `DecisionNode` both subclass it
(the latter via multiple inheritance with the `RouterNode` routing mixin) and
reuse `_run_tool_loop`.
"""
import json
from typing import Any, List, Optional

from pttai.node import Node
from pttai.state import merge_token_usage
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import StructuredTool, BaseTool


def _usage_delta(response):
    """Return `{model_name: usage_metadata}` for one LLM response, or `{}`
    when it carries no `usage_metadata` (fakes / structured-output path)."""
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return {}
    model = (getattr(response, "response_metadata", None) or {}).get("model_name", "unknown")
    return {model: usage}


def _is_openai(llm) -> bool:
    """Duck-type whether `llm` is a langchain_openai ChatOpenAI, WITHOUT
    importing langchain_openai. Unwraps the RunnableBinding that binding tools
    produces so detection still works after tools are bound."""
    target = getattr(llm, "bound", llm)  # RunnableBinding -> underlying model
    cls = type(target)
    return cls.__name__ == "ChatOpenAI" or cls.__module__.startswith("langchain_openai")


class LLMNode(Node):
    """Base for LLM-backed nodes.

    Holds the model, the normalized tool list / lookup, the tool-loop cap, and
    the prompt-cache fields, and provides `_run_tool_loop` — the shared
    tool-call loop that both `AgentNode` (free-form) and `DecisionNode`
    (two-phase, to gather context before routing) run.

    Subclasses decide how tools bind to the model: `AgentNode` binds them onto
    `self.llm`; `DecisionNode` keeps the base model and binds a separate
    `_tool_llm` so tools never share a call with structured output.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 llm: Optional[Any] = None,
                 node_prompt: str = "",
                 tools: Optional[list] = None,
                 max_tool_iterations: int = 25,
                 cache_ttl: Optional[int] = None,
                 retry: bool = False) -> None:
        """Initialize the shared LLM-node plumbing.

        Args:
            name: Unique identifier for the node (inferred from the assignment
                target when omitted; see [Node][pttai.node.Node]).
            llm: Language model instance this node calls.
            node_prompt: System prompt prepended to the history on each call.
            tools: Tools the node may call. Each is normalized to a LangChain
                tool (`StructuredTool`/`BaseTool` as-is; a plain callable via
                `StructuredTool.from_function`) and stored with a name lookup for
                the tool-call loop. Subclasses decide how they bind to the model.
            max_tool_iterations: Safety cap on the tool-call loop in
                `_run_tool_loop`; exceeding it raises `RuntimeError`.
            cache_ttl: see [Node][pttai.node.Node] — node-level result caching.
            retry: see [Node][pttai.node.Node] — node-level retry on exception.
        """
        super().__init__(name, llm, node_prompt, cache_ttl=cache_ttl, retry=retry)
        self.max_tool_iterations = max_tool_iterations
        self.tools = self._normalize_tools(tools) if tools else []
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.tool_available = bool(self.tools)
        # OpenAI prompt-cache routing — set by AgenticGraph when prompt_cache=True.
        self._prompt_cache_enabled = False
        self._prompt_cache_key = None

    @staticmethod
    def _normalize_tools(tools) -> list:
        """Normalize `tools` to a list of LangChain tool objects.

        A `StructuredTool`/`BaseTool` is used as-is; a plain callable is
        wrapped via `StructuredTool.from_function`. Accepts a single tool or a
        list of tools.
        """
        if not isinstance(tools, List):
            tools = [tools]
        normalized = []
        for tool in tools:
            # already a langgraph/langchain tool -> use as is
            if isinstance(tool, StructuredTool) or isinstance(tool, BaseTool):
                pass
            # a plain callable -> wrap it
            elif callable(tool):
                tool = StructuredTool.from_function(
                    func=tool,
                    name=tool.__name__,
                    description=tool.__doc__)
            normalized.append(tool)
        return normalized

    def _run_tool_loop(self, llm, sys, history, invoke_kwargs=None):
        """Invoke `llm` on `[System(sys)] + history` and run the tool-call
        loop until the model stops requesting tools.

        Returns `(messages, log, token)`: every message produced this turn (the
        AIMessage, any ToolMessages, and follow-up AIMessages), the matching trace
        lines, and the merged per-model token usage.
        """
        invoke_kwargs = invoke_kwargs or {}
        prompt = [SystemMessage(content=sys)] + history
        response = llm.invoke(prompt, **invoke_kwargs)
        new_messages = [response]
        new_log = [f'{self.name}:{response.content}']
        token = _usage_delta(response)

        iterations = 0
        while self.tool_available and isinstance(new_messages[-1], AIMessage) \
                and getattr(new_messages[-1], "tool_calls", None):
            if iterations >= self.max_tool_iterations:
                raise RuntimeError(
                    f"{self.name} exceeded max_tool_iterations={self.max_tool_iterations}; "
                    "the model kept requesting tools without finishing."
                )
            iterations += 1
            ai_message = new_messages[-1]
            for tool_call in ai_message.tool_calls:
                name = tool_call["name"]
                tool = self.tools_by_name.get(name)
                if tool is None:
                    # Hallucinated/unknown tool: surface the error to the model
                    # (so it can self-correct) instead of crashing the run.
                    new_messages.append(
                        ToolMessage(
                            content=f"Error: unknown tool '{name}'",
                            name=name,
                            tool_call_id=tool_call["id"],
                        )
                    )
                    new_log.append(f'tools:{name}, args:{tool_call["args"]}, result:unknown tool')
                    continue
                tool_result = tool.invoke(tool_call["args"])
                # A str result is used as-is (avoids double-quoting); anything
                # else is JSON-encoded with default=str so non-JSON returns
                # (datetime/dataclass/pydantic/set/...) don't crash.
                content = tool_result if isinstance(tool_result, str) \
                    else json.dumps(tool_result, default=str)
                new_messages.append(
                    ToolMessage(
                        content=content,
                        name=name,
                        tool_call_id=tool_call["id"],
                    )
                )
                new_log.append(f'tools:{name}, args:{tool_call["args"]}, result:{tool_result}')
            prompt = [SystemMessage(content=sys)] + history + new_messages
            response = llm.invoke(prompt, **invoke_kwargs)
            new_messages.append(response)
            new_log.append(f'{self.name}:{response.content}')
            token = merge_token_usage(token, _usage_delta(response))

        return new_messages, new_log, token
