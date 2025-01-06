"""
State management for the Agentic Framework.
"""

from typing import List
from langchain.schema import AnyMessage
from pydantic import BaseModel

class State(BaseModel):
    message: List[AnyMessage] = []
