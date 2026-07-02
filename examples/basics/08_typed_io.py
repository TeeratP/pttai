"""08 · Typed structured I/O — ``reads=[...]``, ``writes={"score": int}``.

``reads`` are dispatched by value type: a message-list read becomes conversation
history; a scalar read is interpolated into ``node_prompt``. A dict-form
``writes`` switches the node to structured output and returns NATIVE-typed values
(here an ``int``, not a string).

    python examples/basics/08_typed_io.py
"""

import os
import sys

_EX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # examples/
sys.path.insert(0, os.path.dirname(_EX))  # repo root -> `import nae` works from a bare clone
sys.path.insert(0, _EX)  # -> `from _llm import get_llm`
from _llm import get_llm


def nae_version() -> int:
    from nae import AgentNode, AgenticGraph

    rate = AgentNode(
        llm=get_llm(),
        node_prompt="Score this review from 1 to 10: {review}",
        reads=["review"],            # a scalar -> interpolated into the prompt
        writes={"score": int},       # typed structured output -> native int
    )
    graph = AgenticGraph(start_node=rate, end_nodes={rate})
    out = graph.invoke("rate it", review="Fantastic product, would buy again!")
    return out["score"]


# --- equivalent in raw LangGraph ---
def langgraph_version() -> int:
    from typing import Annotated
    import operator
    from langgraph.graph import StateGraph, START, END
    from langchain_core.messages import SystemMessage
    from pydantic import BaseModel
    from typing_extensions import TypedDict

    class Score(BaseModel):
        score: int

    class State(TypedDict):
        review: str
        score: int
        log: Annotated[list, operator.add]

    llm = get_llm()
    model = llm.with_structured_output(Score)

    def rate(state: State):
        prompt = [SystemMessage(f"Score this review from 1 to 10: {state['review']}")]
        return {"score": model.invoke(prompt).score}

    builder = StateGraph(State)
    builder.add_node("rate", rate)
    builder.add_edge(START, "rate")
    builder.add_edge("rate", END)
    graph = builder.compile()

    return graph.invoke({"review": "Fantastic product, would buy again!"})["score"]


if __name__ == "__main__":
    s1 = nae_version()
    s2 = langgraph_version()
    print(f"[nae]      score={s1!r} (type {type(s1).__name__})")
    print(f"[langgraph]  score={s2!r} (type {type(s2).__name__})")
