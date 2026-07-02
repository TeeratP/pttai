"""03 · Sequential chain — ``a > b > c``.

The ``>`` operator wires nodes into a chain. Each node appends to the shared
conversation, so ``c`` sees what ``a`` and ``b`` produced.

    python examples/basics/03_sequential.py
"""

import os
import sys

_EX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # examples/
sys.path.insert(0, os.path.dirname(_EX))  # repo root -> `import nae` works from a bare clone
sys.path.insert(0, _EX)  # -> `from _llm import get_llm`
from _llm import get_llm


def nae_version() -> str:
    from nae import AgentNode, AgenticGraph

    outline = AgentNode(llm=get_llm(), node_prompt="Outline the answer.")
    draft = AgentNode(llm=get_llm(), node_prompt="Write a draft from the outline.")
    polish = AgentNode(llm=get_llm(), node_prompt="Polish the draft.")

    outline > draft > polish        # wire the chain

    graph = AgenticGraph(start_node=outline, end_nodes={polish})
    return graph.invoke("Explain what a monad is.")["messages"][-1].content


# --- equivalent in raw LangGraph ---
def langgraph_version() -> str:
    from langgraph.graph import StateGraph, MessagesState, START, END
    from langchain_core.messages import SystemMessage

    llm = get_llm()

    def step(prompt):
        def node(state: MessagesState):
            return {"messages": [llm.invoke([SystemMessage(prompt)] + state["messages"])]}
        return node

    builder = StateGraph(MessagesState)
    builder.add_node("outline", step("Outline the answer."))
    builder.add_node("draft", step("Write a draft from the outline."))
    builder.add_node("polish", step("Polish the draft."))
    builder.add_edge(START, "outline")
    builder.add_edge("outline", "draft")
    builder.add_edge("draft", "polish")
    builder.add_edge("polish", END)
    graph = builder.compile()

    result = graph.invoke({"messages": [{"role": "user", "content": "Explain what a monad is."}]})
    return result["messages"][-1].content


if __name__ == "__main__":
    print("[nae]     ", nae_version())
    print("[langgraph] ", langgraph_version())
