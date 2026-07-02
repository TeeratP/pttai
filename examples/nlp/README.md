# pttai NLP pipelines — concrete NLP tasks, both ways

pttai is a declarative DSL over LangGraph whose typed per-node `reads`/`writes`
enable a build-time dataflow lint: you wire language-model steps with `>` and it
compiles down to a native LangGraph `StateGraph`. NLP pipelines are a use case,
not pttai's identity — these examples just make the DSL concrete. Each file here
builds a real NLP pipeline the **pttai way first**, then a runnable
`# --- equivalent in raw LangGraph ---` block, so you see the same task both ways.

## Running

Everything runs **offline** with no API key — the shared helper
(`examples/_llm.py`, imported as `get_llm()`) returns a scripted fake chat model
when `OPENAI_API_KEY` is unset. Set `OPENAI_API_KEY` and the exact same pipelines
call **real OpenAI** (`gpt-5.4-nano`) instead — no code change.

```bash
python examples/nlp/rag_qa.py                       # one file
for f in examples/nlp/*.py; do python "$f"; done    # all of them
```

## Index

| File | NLP pipeline |
|------|--------------|
| `rag_qa.py` | **RAG QA** — retrieve over a small corpus, answer grounded in the passages (`make_retriever_tool` + an `AgentNode` retrieve→answer loop) |
| `extract_summarize.py` | **Extract → summarize** — typed structured information extraction (`writes={...}`) feeding an abstractive summary step |
| `doc_triage.py` | **Document triage** — classify an incoming document (`DecisionNode`) and route to a specialized handler |

`rag_qa.py` uses pttai's real `make_retriever_tool` wrapping a tiny in-memory
keyword retriever, so it runs offline; swap in `ChromaRAG` (the `.[rag]` extra)
plus real embeddings for a production vector store.
