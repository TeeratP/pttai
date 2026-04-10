"""
State management for the Agentic Framework.
"""

from typing import List, TypedDict

class AgenticState(TypedDict):
    messages: List[str]
    log: list[str]
