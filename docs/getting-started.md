# Getting started

## Install

Not on PyPI yet — install from source:

```bash
git clone https://github.com/TeeratP/pttai && cd pttai
python -m venv .venv && source .venv/bin/activate
pip install -e ".[openai]"          # core + langchain-openai & python-dotenv
```

Requires **Python ≥ 3.10** (core deps: LangGraph ≥ 1.0, langchain-core ≥ 1.0,
Pydantic 2). Other extras:

- `[rag]` — `langchain-chroma` for `ChromaRAG`
- `[dev]` — `pytest`
- `[docs]` — `mkdocs-material` (this site)

For live model calls, set `OPENAI_API_KEY` in your environment or a `.env` file.

## pttai vs. raw LangGraph — 3 lines vs. 10

The same tool-using agent — an LLM that calls `add` / `multiply` in a loop until
it has the answer. Ask it *"What is 21 + 21, then times 3?"* and both print
**126**. The only thing that differs is how much graph plumbing you write.

```python
# pttai
from pttai import AgentNode, AgenticGraph

agent = AgentNode(name="agent", llm=llm, tools=[add, multiply])
graph = AgenticGraph(start_node=agent, end_nodes={agent})   # schema-free

graph.invoke(message="What is 21 + 21, then times 3?")      # -> 126
```

```python
# raw LangGraph
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

Identical behavior — same tools, same loop, same answer. pttai folds the model
node, the `ToolNode`, the `tools_condition` edge and the loop-back edge into
**one `AgentNode`** with a built-in tool-call loop, and infers the state schema
for you. Both versions run side by side in
[`examples/vs_langgraph.py`](https://github.com/TeeratP/pttai/blob/main/examples/vs_langgraph.py).

## A 30-second multi-agent panel

A question goes to `frame` (which sharpens it into one concrete decision), fans
out to three rival personas — optimist / skeptic / pragmatist — who argue
**concurrently**, then `verdict` weighs every argument into a one-paragraph
ruling. The whole thing is the one wiring line in the middle.

```python
from pttai import AgentNode, AgenticGraph, fanout
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-5.4-nano")

frame = AgentNode(name="frame", llm=llm, node_prompt=(
    "Restate the user's question as ONE sharp, concrete decision. One sentence."))
optimist = AgentNode(name="optimist", llm=llm, node_prompt=(
    "Relentless optimist. Argue FOR the bold move — two strongest upsides."))
skeptic = AgentNode(name="skeptic", llm=llm, node_prompt=(
    "Hard-nosed skeptic. Argue AGAINST — the two biggest risks."))
pragmatist = AgentNode(name="pragmatist", llm=llm, node_prompt=(
    "Pragmatist. Propose the smallest concrete next step that de-risks it."))
verdict = AgentNode(name="verdict", llm=llm, node_prompt=(
    "You are the chair. Weigh all three above into a balanced one-paragraph verdict."))

# The line that matters: the three personas run IN PARALLEL, then join at `verdict`.
frame > fanout(optimist, skeptic, pragmatist) > verdict

panel = AgenticGraph(start_node=frame, end_nodes={verdict})   # schema-free

out = panel.invoke(message="Should an early-stage SaaS rewrite its monolith into microservices?")
print(out["messages"][-1].content)        # the verdict
panel.summary()                           # the topology table
print(out["token"])                       # per-model token totals
```

Runs from a single paste with just `OPENAI_API_KEY` set. Full version:
[`examples/panel.py`](https://github.com/TeeratP/pttai/blob/main/examples/panel.py).

## vs. LangChain's Functional API

The closest comparison isn't raw graphs — it's LangChain's own Functional API
(`@entrypoint` / `@task`), which also lets you skip explicit graph wiring. The
difference is **visibility of control flow**:

| | Functional API (`@entrypoint`/`@task`) | pttai |
|---|---|---|
| Control flow | hidden in plain Python (loops, `if`, `await`) | an explicit, declarative DAG |
| Fan-out / join | you orchestrate futures by hand | `fanout(...)` / `.map("field")`, one line |
| Inspect the topology | run it and trace | `summary()` prints the static DAG |
| Catch dataflow bugs | at runtime | at **compile time**, before any invoke |

Both are concise. pttai's edge is that the topology is *inspectable and
validatable* — you can see the fan-out/join/map-reduce structure, render it, and
have the compiler reject read-before-written bugs.

## Running it

```bash
python -m pytest tests/                # full suite, no API calls (a scripted FakeLLM stands in)
python examples/parallel_usage.py      # offline tour: parallel + map-reduce + validation
python examples/panel.py               # live multi-agent panel (needs OPENAI_API_KEY)
python examples/vs_langgraph.py        # the 3-vs-10 comparison, both ways (needs OPENAI_API_KEY)
```

Next: learn the [node types](node-types.md), then work through the
[basics](examples/basics.md) and [architectures](examples/architectures.md) galleries.
