# Quickstart

This page takes you from install to a running multi-agent graph in a few
minutes. It ends with the part that saves you real debugging time: the
validator catching a bug before a single model call.

## 1. Install

```bash
pip install pttai

# pttai works with any LangChain chat model — install the provider you want:
pip install langchain-openai python-dotenv   # OpenAI (used on this page)
pip install langchain-anthropic              # Anthropic
pip install langchain-google-genai           # Google

export OPENAI_API_KEY=sk-...   # or your provider's key, or a .env file
```

Nodes take the model via `llm=` — pass any LangChain `BaseChatModel`. One
caveat: `reasoning_effort` and structured-output routing are tuned to OpenAI
gpt-5.x semantics; wiring, the tool-call loop, and routing are otherwise
provider-neutral.

Requires Python ≥ 3.10. To work on pttai itself, clone the repo and
`pip install -e ".[openai,dev]"` (extras: `[rag]` for `ChromaRAG`, `[docs]` for
this site).

## 2. Hello, agent

Three lines make a complete graph: one node, one constructor, one invoke.

```python
from pttai import AgentNode, AgenticGraph
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-5.4-nano")   # any LangChain chat model works

hello = AgentNode(llm=llm, node_prompt="Answer in one concise paragraph.")
graph = AgenticGraph(start_node=hello, end_nodes={hello})

out = graph.invoke(message="What does a build-time graph validator buy me?")
print(out["messages"][-1].content)
```

The node's name is inferred from the variable (`hello`), and the state schema
is inferred too. Pass `tools=[...]` to give the node a built-in tool-call loop
— see [Node types](node-types.md) for what each node can do.

## 3. Fan out: a three-persona panel

Now something raw wiring makes tedious: a question goes to `frame`, fans out to
three personas that argue **concurrently**, and joins at `verdict`. The whole
topology is the one wiring line near the bottom.

```python
from pttai import AgentNode, AgenticGraph, fanout

frame = AgentNode(llm=llm, node_prompt=(
    "Restate the user's question as ONE sharp, concrete decision. One sentence."))
optimist = AgentNode(llm=llm, node_prompt=(
    "Relentless optimist. Argue FOR the bold move — two strongest upsides."))
skeptic = AgentNode(llm=llm, node_prompt=(
    "Hard-nosed skeptic. Argue AGAINST — the two biggest risks."))
pragmatist = AgentNode(llm=llm, node_prompt=(
    "Pragmatist. Propose the smallest concrete next step that de-risks it."))
verdict = AgentNode(llm=llm, node_prompt=(
    "You are the chair. Weigh all three above into a balanced one-paragraph verdict."))

frame > fanout(optimist, skeptic, pragmatist) > verdict   # parallel, then join

panel = AgenticGraph(start_node=frame, end_nodes={verdict})

out = panel.invoke(message="Should an early-stage SaaS rewrite its monolith into microservices?")
print(out["messages"][-1].content)        # the verdict
panel.summary()                           # the topology table
print(out["token"])                       # per-model token totals
```

`fanout(...)` runs the three branches in parallel and defers the join, so
`verdict` fires once after all three finish. `summary()` prints a topology
table and `out["token"]` accumulates usage across every call — both covered in
[State & observability](state.md). Full version:
[`examples/panel.py`](https://github.com/TeeratP/pttai/blob/main/examples/panel.py).

## 4. Break it — and let the validator tell you

Wire a graph with a real dataflow bug: `early` reads a `summary` key that only
`late`, downstream of it, produces.

```python
from pttai import AgentNode, AgenticGraph

early = AgentNode(llm=llm, reads=["summary"],
                  node_prompt="Polish this summary: {summary}")
late  = AgentNode(llm=llm, writes=["summary"],
                  node_prompt="Summarize the conversation in two sentences.")

early > late
AgenticGraph(start_node=early, end_nodes={late})   # raises before any model call
```

The constructor fails immediately, naming the node, the key, and the real
producer:

```
pttai.validation.GraphValidationError: AgenticGraph 'graph': 1 error(s), 0 warning(s)
  [error] early: reads computed key 'summary' but no upstream node produces it before this node (produced by: ['late'], none of which are upstream); available keys here: ['log', 'messages', 'token']
```

Raw LangGraph compiles this graph and only fails at runtime, after `early` has
already burned a model call. The fix is the wiring order — write before read:

```python
late > early
graph = AgenticGraph(start_node=late, end_nodes={early})   # builds cleanly
```

See [The validator](validator.md) for all seven bug classes it catches, each
with its verbatim error message.

## 5. Where to go next

- **[Node types](node-types.md)** — routing with `DecisionNode` /
  `ConditionNode`, human-in-the-loop with `HumanNode`, typed `reads`/`writes`.
- **[Examples](examples.md)** — 21 runnable files, each with a raw-LangGraph
  equivalent in the same file; they run offline with no API key.
- **[Coming from LangGraph](coming-from-langgraph.md)** — a direct API mapping.
- **Retrieval:** wrap any LangChain retriever as a tool with
  `make_retriever_tool` (see [Node types](node-types.md#rag-tools)) — a
  complete RAG pipeline is in
  [`examples/nlp/rag_qa.py`](https://github.com/TeeratP/pttai/blob/main/examples/nlp/rag_qa.py).
