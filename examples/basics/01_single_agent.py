"""01 · Single agent — one LLM node, in and out.

The smallest useful graph: one ``AgentNode``. nae infers the state schema
(``messages``) and the node name for you.

Run offline (no key) or with OPENAI_API_KEY set:

    python examples/basics/01_single_agent.py
"""

import os
import sys

_EX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # examples/
sys.path.insert(0, os.path.dirname(_EX))  # repo root -> `import nae` works from a bare clone
sys.path.insert(0, _EX)  # -> `from _llm import get_llm`
from _llm import get_llm


def nae_version() -> str:
    from nae import AgentNode, AgenticGraph

    agent = AgentNode(llm=get_llm(), node_prompt="You are a helpful assistant.")
    graph = AgenticGraph(start_node=agent, end_nodes={agent})   # schema inferred
    return graph.invoke("Say hello.")["messages"][-1].content


# --- equivalent in raw LangGraph ---
def langgraph_version() -> str:
    from langgraph.graph import StateGraph, MessagesState, START, END
    from langchain_core.messages import SystemMessage

    llm = get_llm()

    def call_model(state: MessagesState):
        prompt = [SystemMessage("You are a helpful assistant.")] + state["messages"]
        return {"messages": [llm.invoke(prompt)]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", call_model)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    graph = builder.compile()

    result = graph.invoke({"messages": [{"role": "user", "content": "Say hello."}]})
    return result["messages"][-1].content


if __name__ == "__main__":
    print("[nae]     ", nae_version())
    print("[langgraph] ", langgraph_version())
