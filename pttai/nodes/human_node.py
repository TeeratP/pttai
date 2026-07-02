"""
Human-in-the-loop node implementation for the Agentic Framework.
"""
from typing import Callable, Optional, Union

from pttai.node import Node
from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

class HumanNode(Node):
    """Resumable human-in-the-loop step, backed by LangGraph's `interrupt()`.

    When the run reaches a `HumanNode` it pauses and surfaces a payload for a
    person to review (by default the content of a recent message, chosen by
    `n`; override with `show`). The run resumes when you invoke again with
    `Command(resume=value)` on the same `thread_id`; the resume value lands
    `into` a state key — `"messages"` (default) wraps it as a `HumanMessage`
    and appends it, any other key writes the raw value (so a router can gate on
    the answer). Resume requires the graph to be built with a `checkpointer`.

    Examples:
        ```python
        from pttai import AgentNode, HumanNode, AgenticGraph
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.types import Command

        draft = AgentNode(llm=llm, node_prompt="Draft a reply.")
        review = HumanNode(node_prompt="Approve or edit this draft:", n=1)
        finalize = AgentNode(llm=llm, node_prompt="Incorporate the human feedback.")

        draft > review > finalize

        graph = AgenticGraph(start_node=draft, end_nodes={finalize},
                             checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "demo-1"}}
        graph.invoke("Reply to the customer.", config=config)   # pauses at review
        graph.invoke(Command(resume="Looks good, ship it."), config=config)
        ```
    """

    def __init__(self,
                 name: Optional[str] = None,
                 node_prompt: str = "Please review the following message and provide feedback.",
                 n: int = 1,
                 show: Optional[Union[str, Callable]] = None,
                 into: str = "messages",
                 cache_ttl: Optional[int] = None,
                 retry: bool = False,
                 ) -> None:
        """
        Initialize a HumanNode — free-form human-in-the-loop via interrupt/resume.

        Args:
            name: Unique identifier for the node.
            node_prompt: key under which `show` is surfaced in the interrupt payload.
            n: which prior message to surface for review when `show` is None
                (`state["messages"][-n]`; n<=0 shows nothing).
            show: what to surface for review — None => the n-th message's content
                (default behavior); a str => that literal; a callable => show(state).
            into: where the human's free-form resume value lands — "messages"
                (default) wraps it as a HumanMessage and appends; any other key
                writes the raw value to that key (for a router to gate on).
            cache_ttl: see [Node][pttai.node.Node] — node-level result caching.
            retry: see [Node][pttai.node.Node] — node-level retry on exception.
        """
        super().__init__(name, cache_ttl=cache_ttl, retry=retry)
        self.node_prompt = node_prompt
        self.n = n
        self.show = show
        self.into = into

    def __call__(self, state):
        """Pause the run for human input and return the reply as a delta.

        Computes the display payload — `show(state)` / the literal `show` / the
        `n`-th message's content (nothing when `n <= 0` or too few messages) —
        then calls LangGraph's `interrupt()` with `{node_prompt: display}`. On
        the FIRST pass this raises the interrupt and suspends the run; when
        resumed via `Command(resume=value)`, `interrupt()` returns that value,
        which is written `into` the target key (`"messages"` wraps it as a
        `HumanMessage`, otherwise the raw value is written).

        Raises:
            ValueError: If `state` has no `messages` key.
        """
        if 'messages' not in state:
            raise ValueError("State must contain a 'messages' key")
        if self.show is not None:
            disp = self.show(state) if callable(self.show) else self.show
        elif self.n <= 0 or len(state["messages"]) < self.n:
            # n<=0 shows nothing; clamp when fewer than n messages exist so
            # indexing never raises IndexError.
            disp = ""
        else:
            disp = state["messages"][-self.n].content

        reply = interrupt(
                {
                    self.node_prompt: disp,
                }
            )

        if self.into == "messages":
            return {"messages": [HumanMessage(content=reply)], "log": [f"{self.name}:{reply}"]}
        return {self.into: reply, "log": [f"{self.name}:{reply}"]}
