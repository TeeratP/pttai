"""The `AgentNode` — pttai's workhorse LLM node.

Prepends `node_prompt` as a `SystemMessage` to the message history, calls the
model, and returns a state delta. On top of the plain call it wraps two modes
inherited/extended from `LLMNode`: an automatic tool-call loop (`tools=[...]`,
which folds together the model node, tool execution, and the loop-back edge that
raw LangGraph makes you wire by hand) and structured output (`writes=[...]` /
`writes={key: type}`) that returns native-typed fields instead of appending to
the conversation.
"""
from typing import Any, Optional, List, Dict, Union
from pttai.nodes.llm_node import LLMNode, _usage_delta, _is_openai
from pttai.nodes._fields import partition_reads
from langchain_core.messages import SystemMessage
from pydantic import create_model


class AgentNode(LLMNode):
    """An LLM agent: prompt the model over the history and return a delta.

    Prepends `node_prompt` as a `SystemMessage` to the history (read from
    `input_field`, default `messages`), calls the LLM, and returns only the keys
    it updates. Two optional modes change what it does with the response:

    - **Tool-call loop** (`tools=[...]`): every requested tool is executed,
      `ToolMessage`s are appended, and the model is re-invoked until it stops
      calling tools (capped by `max_tool_iterations`) — all of it folded into one
      `{"messages": [...]}` delta. Cannot be combined with multi-field structured
      `writes`.
    - **Structured output** (`writes=[...]` with two+ keys, or a
      `writes={key: type}` dict): the model is wrapped with
      `with_structured_output`, so it returns one field per key. The dict form
      types each field, so values come back native (`{"score": int}` -> `9`, an
      int) rather than stringified.

    A non-default single `output_field`/`writes` key writes the final response's
    *content* to that key instead of appending to `messages` (a transform node).

    Examples:
        ```python
        from pttai import AgentNode, AgenticGraph

        def add(a: int, b: int) -> int:      return a + b
        def multiply(a: int, b: int) -> int: return a * b

        agent = AgentNode(llm=llm, tools=[add, multiply])   # name inferred -> "agent"
        graph = AgenticGraph(start_node=agent, end_nodes={agent})

        graph.invoke(message="What is 21 + 21, then times 3?")   # -> 126
        ```
    """

    def __init__(self,
                 name: Optional[str] = None,
                 llm: Optional[Any] = None,
                 node_prompt: str = "you are a helpful assistant",
                 tools: Optional[list] = None,
                 max_tool_iterations: int = 25,
                 input_field: str = "messages",
                 output_field: str = "messages",
                 reads: Optional[List[str]] = None,
                 writes: Optional[Union[List[str], Dict[str, type]]] = None,
                 reasoning_effort: Optional[str] = None,
                 cache_ttl: Optional[int] = None,
                 retry: bool = False) -> None:
        """
        Initialize an AgentNode.

        Args:
            name: Unique identifier for the node
            llm: Language model instance to be used by this node
            node_prompt: System prompt/instructions for the language model
            tools: Tools this agent can call. A `StructuredTool`/`BaseTool` is
                used as-is; a plain callable is wrapped via
                `StructuredTool.from_function`. When set, the node runs an
                internal tool-call loop (capped by `max_tool_iterations`).
                Cannot be combined with multi-field structured `writes`.
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
                generalization of output_field). Accepts either a list[str] or a
                dict[str, type]. ["messages"] appends to the conversation; a
                single scalar key (list form) writes the final response content;
                two or more scalar keys switch to structured output (one field
                per key, no tool loop). A dict[str, type] ALWAYS uses structured
                output (even for one key) and types each field as given, so the
                node returns NATIVE-typed values (e.g. {"score": int} -> 9, an
                int) instead of all-str. A list[str] types every field as str.
                Use writes OR output_field.
            reasoning_effort: Reasoning effort for reasoning-capable models
                (e.g. "low"/"medium"/"high" on gpt-5.x). Passed as a per-call
                kwarg to the LLM. (DecisionNode does not expose this — reasoning
                effort conflicts with structured output on current OpenAI models.)
            cache_ttl: see [Node][pttai.node.Node] — node-level result caching.
            retry: see [Node][pttai.node.Node] — node-level retry on exception.
        """
        super().__init__(name=name, llm=llm, node_prompt=node_prompt, tools=None,
                         max_tool_iterations=max_tool_iterations,
                         cache_ttl=cache_ttl, retry=retry)
        self.input_field = input_field
        self.output_field = output_field
        # reads/writes win if both forms are given (document: use one or the other).
        self.reads = reads if reads is not None else [input_field]
        # Normalize writes to a {key: type} map. A dict is supplied verbatim and
        # marks "typed/structured" mode (always structured, even for one key); a
        # list becomes {key: str} and keeps the count-based behavior (single
        # scalar -> content write, >=2 -> all-str structured). self.writes stays
        # a dict whose KEYS are the write keys, so graph.py's set(node.writes)
        # key-extraction keeps working unchanged.
        writes = writes if writes is not None else [output_field]
        self._writes_typed = isinstance(writes, dict)
        self.writes = dict(writes) if self._writes_typed else {k: str for k in writes}
        self.reasoning_effort = reasoning_effort
        # Per-call kwarg (survives tool binding, unlike a pre-bound reasoning_effort).
        self._invoke_kwargs = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}

        if tools is not None:
            # Structured output preempts the free-form tool loop on OpenAI/LangGraph.
            if self._is_structured([w for w in self.writes if w != "messages"]):
                raise ValueError(
                    "multi-field structured output (writes=[...]) cannot be "
                    "combined with tools in v1")
            self.tools = self._normalize_tools(tools)
            self.tools_by_name = {tool.name: tool for tool in self.tools}
            self.tool_available = True
            self.llm = self.llm.bind_tools(self.tools)

    def _is_structured(self, scalar_writes) -> bool:
        """Whether this node uses structured/typed output for `scalar_writes`.

        Two or more scalar keys always go structured (today's rule). A dict-form
        `writes` ("typed" mode) goes structured even for a single key; a
        list-form single scalar key keeps the .content-write behavior.
        """
        return len(scalar_writes) >= 2 or (self._writes_typed and len(scalar_writes) >= 1)

    def __call__(self, state: dict) -> dict:
        """
        Process the current state and generate a response.

        Runs the agent and, when tools are set, its internal tool-call loop,
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

        # Only interpolate when there is a scalar read, so prompts containing
        # literal braces (e.g. JSON) with no scalars pass through verbatim.
        sys = self.node_prompt.format_map(scalars) if scalars else self.node_prompt

        scalar_writes = [w for w in self.writes if w != "messages"]
        if "messages" in self.writes and scalar_writes:
            raise ValueError(
                "a node writes either the conversation ('messages') or "
                "structured fields, not both")

        prompt = [SystemMessage(content=sys)] + history

        # Structured output (one field per key, no tool loop, no reasoning_effort
        # — mirrors DecisionNode). List-form fields are typed `str`; dict-form
        # ("typed") fields carry the user-declared Python type so values come
        # back NATIVE (int/bool/list/...), not stringified.
        if self._is_structured(scalar_writes):
            Model = create_model("Out", **{w: (self.writes[w], ...) for w in scalar_writes})
            out = self.llm.with_structured_output(Model).invoke(prompt)
            values = {w: getattr(out, w) for w in scalar_writes}
            return values | {"log": [f"{self.name}:{values}"]}

        # Per-call kwargs: reasoning_effort (if set) plus an OpenAI
        # prompt_cache_key when prompt caching is on AND the model is OpenAI.
        invoke_kwargs = dict(self._invoke_kwargs)
        if self._prompt_cache_enabled and _is_openai(self.llm):
            invoke_kwargs["prompt_cache_key"] = self._prompt_cache_key

        new_messages, new_log, token = self._run_tool_loop(self.llm, sys, history, invoke_kwargs)

        if not scalar_writes:  # writes == ["messages"] (default)
            delta = {"messages": new_messages, "log": new_log}
        else:
            # Transform node: write the final response content to the single field.
            delta = {scalar_writes[0]: new_messages[-1].content, "log": new_log}
        if token:  # only emit when at least one call reported usage_metadata
            delta["token"] = token
        return delta
