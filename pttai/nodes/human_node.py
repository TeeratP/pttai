"""
Human-in-the-loop node implementation for the Agentic Framework.
"""
from typing import Callable, Optional, Union

from pttai.node import Node
from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

class HumanNode(Node):
    def __init__(self,
                 name: str = 'human_node',
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
                (state["messages"][-n]; n<=0 shows nothing).
            show: what to surface for review — None => the n-th message's content
                (default behavior); a str => that literal; a callable => show(state).
            into: where the human's free-form resume value lands — "messages"
                (default) wraps it as a HumanMessage and appends; any other key
                writes the raw value to that key (for a router to gate on).
            cache_ttl/retry: see Node — node-level caching/retry.
        """
        super().__init__(name, cache_ttl=cache_ttl, retry=retry)
        self.node_prompt = node_prompt
        self.n = n
        self.show = show
        self.into = into

    def __call__(self, state):

        if 'messages' not in state:
            raise ValueError("State must contain a 'messages' key")
        if self.show is not None:
            disp = self.show(state) if callable(self.show) else self.show
        elif self.n <= 0:
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
