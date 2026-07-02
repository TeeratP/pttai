# Quickstart: build + validate a RAG pipeline in 5 minutes

This walkthrough builds a small **retrieval-augmented generation** agent, has
pttai **validate** it at build time, then runs it. By the end you'll have seen
the three things that make pttai worth using: the `>` wiring, `AgentNode` with a
retriever tool, and the compile-time validator.

## 1. Install (1 min)

```bash
pip install pttai

# pttai works with ANY LangChain chat model — install the provider you want:
pip install langchain-openai python-dotenv   # OpenAI (used in this walkthrough)
pip install langchain-anthropic              # Anthropic
pip install langchain-google-genai           # Google

export OPENAI_API_KEY=sk-...   # or put it in a .env file
```

Nodes take the model via `llm=` — pass any LangChain `BaseChatModel`. (A
`pttai[openai]` extra exists as a shortcut for `langchain-openai` +
`python-dotenv`, but it's convenience only, not a requirement.)

Or from source for development:

```bash
git clone https://github.com/TeeratP/pttai && cd pttai
python -m venv .venv && source .venv/bin/activate
pip install -e ".[openai]"
```

## 2. Wrap a retriever as a tool (1 min)

pttai ships `make_retriever_tool`, which wraps **any** LangChain retriever
(anything with `.invoke(query) -> docs`) as a tool an `AgentNode` can call. Here
we use a tiny in-memory retriever so the example is self-contained — swap in
Chroma, FAISS, or a managed vector store unchanged.

```python
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pttai.tools import make_retriever_tool

DOCS = [
    Document(page_content="pttai compiles a `>`-wired node graph to a LangGraph StateGraph."),
    Document(page_content="AgenticGraph validates dataflow at build time and fails on read-before-write."),
    Document(page_content="fanout(...) runs branches in parallel and joins once after all finish."),
]

class TinyRetriever(BaseRetriever):
    def _get_relevant_documents(self, query, *, run_manager=None):
        # naive keyword match — stand-in for a real vector store
        return [d for d in DOCS if any(w in d.page_content.lower()
                                       for w in query.lower().split())] or DOCS

search = make_retriever_tool(
    TinyRetriever(),
    name="search_docs",
    description="Search the pttai knowledge base for relevant context.",
)
```

!!! tip "Using Chroma instead"
    With the `[rag]` extra installed, use the bundled convenience wrapper:
    ```python
    from pttai.tools import ChromaRAG
    rag = ChromaRAG(embeddings)          # any LangChain embeddings
    rag.add_texts(["...", "..."])
    search = rag.as_tool()               # same StructuredTool
    ```

## 3. Build the graph (1 min)

A single `AgentNode` with the retriever tool bound is a complete RAG loop: the
model decides when to search, reads the results, and answers. `AgenticGraph`
compiles it — and runs the validator as it does.

```python
from langchain_openai import ChatOpenAI
from pttai import AgentNode, AgenticGraph

llm = ChatOpenAI(model="gpt-5.4-nano")
# swap for any LangChain chat model, e.g.:
#   from langchain_anthropic import ChatAnthropic;              llm = ChatAnthropic(model="claude-opus-4-8")
#   from langchain_google_genai import ChatGoogleGenerativeAI;  llm = ChatGoogleGenerativeAI(model="gemini-...")

rag = AgentNode(
    name="rag",
    llm=llm,
    tools=[search],
    node_prompt=(
        "Answer the user's question. ALWAYS call search_docs first to ground "
        "your answer in the knowledge base, then cite what you found."
    ),
)

graph = AgenticGraph(start_node=rag, end_nodes={rag})   # schema-free; validates here
```

The `AgentNode` folds the model call, the tool execution, and the loop-back into
one node — no `ToolNode`, no `tools_condition`, no manual edges.

## 4. Validate before you spend a token (1 min)

Validation already ran inside the constructor above (it raises
`GraphValidationError` on a hard error). Inspect the topology and any warnings
explicitly with `summary()`:

```python
graph.summary()
```

```
AgenticGraph 'graph'   state=AgenticState
initial: log, messages, token
-----------------------------------------------------------
node  type       reads     writes        available
rag   AgentNode  messages  log,messages  log,messages,token
-----------------------------------------------------------
1 nodes · 0 errors · 0 warning(s)
```

To see the validator *catch* a bug, add a downstream node that reads a key
nothing produces — the build fails immediately with a message naming the node
and key (see the [validator page](validator.md) for every bug class it catches).

## 5. Run it

```python
out = graph.invoke(message="How does pttai handle parallel branches?")
print(out["messages"][-1].content)   # grounded answer, citing the fanout doc
print(out["token"])                  # per-model token usage for the whole run
```

## Where to go next

- **[The validator](validator.md)** — every dataflow bug it catches, with the
  verbatim error. This is the reason to build on pttai.
- **[Node types](node-types.md)** — `DecisionNode` routing, `HumanNode`
  interrupt/resume, typed multi-key `reads`/`writes`.
- **[Examples → Architectures](examples/architectures.md)** — ReAct, routing,
  orchestrator-workers, reflection, and more, each runnable offline.
- **[Coming from LangGraph](coming-from-langgraph.md)** — a direct API mapping.
