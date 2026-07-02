# nae vs. raw LangGraph

The same tool-using agent — an LLM that calls `add` / `multiply` in a loop until
it has the answer. Ask it *"What is 21 + 21, then times 3?"* and both print **126**.

**Graph-building code: 2 lines vs. 10.**

---

### nae

```python
from nae import AgentNode, AgenticGraph

agent = AgentNode(name="agent", llm=llm, tools=[add, multiply])
graph = AgenticGraph(start_node=agent, end_nodes={agent})   # schema-free

graph.invoke(message="What is 21 + 21, then times 3?")      # -> 126
```

### raw LangGraph

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

llm_with_tools = llm.bind_tools([add, multiply])

def call_model(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_node("tools", ToolNode([add, multiply]))
builder.add_edge(START, "call_model")
builder.add_conditional_edges("call_model", tools_condition)  # tools? -> "tools" : END
builder.add_edge("tools", "call_model")                       # loop back to the model
graph = builder.compile()

graph.invoke({"messages": [{"role": "user", "content": "What is 21 + 21, then times 3?"}]})  # -> 126
```

---

**Identical behavior** — same tools, same loop, same answer (126). nae folds
the model node, the `ToolNode`, the `tools_condition` edge and the loop-back edge
into **one `AgentNode`** with a built-in tool-call loop, and infers the state
schema for you. Runnable both ways: [`vs_langgraph.py`](./vs_langgraph.py).
