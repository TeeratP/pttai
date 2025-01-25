"""
State management for the Agentic Framework.
"""

from typing import List
# from langchain.schema import AnyMessage
# from pydantic import BaseModel
from langgraph.graph import MessagesState

class AgenticState(MessagesState):
    log: List[str] = ["START"]
    
