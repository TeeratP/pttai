# Node types

pttai ships four node types. All are callables (`__call__(state) -> delta`)
invoked by LangGraph with the shared state; each returns **only the keys it
updates** and never mutates state in place, which is what keeps checkpointing,
parallel-branch merges, and subgraph composition correct.

The two LLM-backed nodes (`AgentNode`, `DecisionNode`) share a common
[`LLMNode`](api/advanced/llmnode.md) base that owns the model binding, the
tool-call loop, and token accounting — subclass it if you build your own
LLM-backed node type.

!!! note
    The snippets below assume `llm = ...` is any LangChain chat model, plus
    `from pttai import AgentNode, DecisionNode, ConditionNode, HumanNode,
    AgenticGraph`. Full signatures live in the API reference, linked from each
    section.

## `AgentNode`

Reach for [`AgentNode`](api/nodes/agentnode.md) for any "have a model do
something" step. It prepends your `node_prompt` as a system message, calls the
LLM, and returns the result. Give it `tools=[...]` — bare callables are wrapped
as `StructuredTool`s — and it runs a built-in loop: execute each tool call,
append the result, re-invoke the model, until the model stops asking.

```python
def word_count(text: str) -> int:
    """Count the words in a text."""
    return len(text.split())

agent = AgentNode(llm=llm, tools=[word_count],
                  node_prompt="Answer, calling tools when they help.")
graph = AgenticGraph(start_node=agent, end_nodes={agent})
print(graph.invoke(message="How many words in 'to be or not to be'?")["messages"][-1].content)
```

Typed IO is on the same node: `reads=[...]` interpolates scalar keys into the
prompt (`{key}` placeholders), and `writes={"score": int}` returns native-typed
structured output instead of appending to the conversation. An optional
`reasoning_effort` (`"low"`/`"medium"`/`"high"`) applies to gpt-5.x models.

## `DecisionNode`

Use [`DecisionNode`](api/nodes/decisionnode.md) when *the model* should pick
the path — classify, triage, route by intent. It constrains the model's output
to a `Literal` of your `choices`, so it can only return a valid branch; you
wire each branch by indexing the choice (`decide > x` is an error by design).

```python
cheer = AgentNode(llm=llm, node_prompt="Celebrate the good news with the user.")
console = AgentNode(llm=llm, node_prompt="Acknowledge the problem; offer one next step.")
decide = DecisionNode(llm=llm, choices=["positive", "negative"],
                      node_prompt="Is the sentiment of the message positive or negative?")

decide["positive"] > cheer
decide["negative"] > console

graph = AgenticGraph(start_node=decide, end_nodes={cheer, console})
out = graph.invoke(message="We closed our first enterprise customer!")   # -> cheer
```

The chosen label goes to a private per-node state channel, never into the
conversation (see the defaults below). `DecisionNode` also accepts
`tools=[...]`: it runs a tool-gathering loop first, then routes via structured
output — the two never share a call.

## `ConditionNode`

[`ConditionNode`](api/nodes/conditionnode.md) is deterministic branching with
**no LLM**. Reach for it when the branch is decidable in plain Python — a
length check, a loop counter, a flag another node wrote. Routing is free,
reproducible, and prompt-less, which makes it the natural way to cap loops.

```python
def is_short(state) -> str:
    return "short" if len(state["messages"][-1].content) < 20 else "long"

route = ConditionNode(condition=is_short, choices=["short", "long"])
brief = AgentNode(llm=llm, node_prompt="Give a one-line answer.")
detailed = AgentNode(llm=llm, node_prompt="Give a thorough answer.")

route["short"] > brief
route["long"] > detailed

graph = AgenticGraph(start_node=route, end_nodes={brief, detailed})
out = graph.invoke(message="Hi")   # -> brief
```

A loop-capping gate that counts rounds in the `log` channel is shown in
[Examples](examples.md#capping-loops-offline).

## `HumanNode`

[`HumanNode`](api/nodes/humannode.md) is resumable human-in-the-loop, backed by
LangGraph's `interrupt()`. The run pauses and surfaces a payload for review —
by default the previous message (`n=1`), or anything you pass as `show`. Invoke
again with `Command(resume=...)` on the same thread and the reply flows back
into state (into `messages`, or any key a router can gate on, via `into=`).

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

draft = AgentNode(llm=llm, node_prompt="Draft a reply.")
review = HumanNode(node_prompt="Approve or edit this draft:", n=1)
finalize = AgentNode(llm=llm, node_prompt="Incorporate the human feedback.")

draft > review > finalize
graph = AgenticGraph(start_node=draft, end_nodes={finalize},
                     checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "demo-1"}}

first = graph.invoke("Reply to the customer.", config=config)      # pauses at `review`
final = graph.invoke(Command(resume="Looks good, ship it."), config=config)
```

Pause/resume needs a `checkpointer` on the graph and a `thread_id` in the
config — without them there is nowhere to store the paused run.

## Defaults and shared knobs

- **`max_tool_iterations=25`** caps the tool-call loop on `AgentNode` (and
  `DecisionNode`'s gathering phase), so a confused model cannot loop forever.
- **Routing channels are per-node.** Each `DecisionNode`/`ConditionNode` writes
  its choice to its own `decision_{name}` key (e.g. `out["decision_route"]`),
  auto-registered in the schema — parallel routers never clobber each other.
- **`cache_ttl` / `retry`** are accepted by every node type: `cache_ttl` wraps
  the node in a LangGraph `CachePolicy` (the graph auto-provides an
  `InMemoryCache`), and `retry` attaches a `RetryPolicy` for exceptions.
- **Graphs compose as nodes.** An `AgenticGraph` can be wired inside a larger
  one (`graph_0 > graph_1`) — see
  [`examples/basics/10_graph_composition.py`](https://github.com/TeeratP/pttai/blob/main/examples/basics/10_graph_composition.py).

## RAG tools

`make_retriever_tool` wraps any LangChain retriever (anything with
`.invoke(query) -> docs`) as a tool you hand to `AgentNode(tools=[...])`; with
the `[rag]` extra, `ChromaRAG` bundles a Chroma store behind the same
interface. See the [RAG tools API](api/advanced/tools.md) and the end-to-end
pipeline in
[`examples/nlp/rag_qa.py`](https://github.com/TeeratP/pttai/blob/main/examples/nlp/rag_qa.py).

See every node in action, one file at a time, in the [Examples](examples.md)
gallery.
