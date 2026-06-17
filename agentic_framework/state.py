"""
State management for the Agentic Framework.
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgenticState(TypedDict):
    # add_messages appends new messages, replaces an existing message when IDs
    # match, and coerces bare strings to HumanMessage. It also merges updates
    # from parallel branches instead of clobbering them.
    messages: Annotated[list[AnyMessage], add_messages]
    # operator.add concatenates log fragments from parallel branches.
    log: Annotated[list[str], operator.add]
    # Transient routing key written by DecisionNode and read by its route().
    # Plain (no reducer): last-writer-wins, which is correct because route()
    # always runs immediately after the same node writes it.
    decision: str
