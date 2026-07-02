"""Parallel topology, map-reduce, typed multi-key IO, and build-time validation.

Runs entirely OFFLINE on a tiny inline fake LLM (no OPENAI_API_KEY needed):

    python examples/parallel_usage.py

It demonstrates, in order:
  1. Parallel fan-out + join     start > fanout(worker_a, worker_b) > combine
  2. Map-reduce                  dispatch > summarize.map("docs") > reduce
  3. Typed multi-key state IO    AgentNode(reads=[...], writes=[...])
  4. graph.summary()            the Keras-style table
  5. Build-time validation      a deliberately broken graph caught at build

For a LIVE model demo (tool call -> decision -> branch), see examples/sample_usage.py.
"""

import itertools
import operator
from types import SimpleNamespace
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph.message import add_messages

from nae import AgentNode, AgenticGraph, AgenticState, fanout
from nae.validation import GraphValidationError


# --- a minimal inline fake LLM so the script runs without an API key ---------
_ids = itertools.count()


def _ai(content: str) -> AIMessage:
    # Distinct id per message so the add_messages reducer appends (never dedupes).
    return AIMessage(content=content, id=f"ai-{next(_ids)}")


class FakeLLM:
    """Scripted stand-in for a chat model.

    - default: every .invoke() returns a fixed reply.
    - echo=True: replies with the last message's content (used by the map worker
      so each item produces a distinguishable summary).
    - structured={...}: .with_structured_output(Model).invoke() yields those
      fields (used by the multi-key structured-write node).
    """

    def __init__(self, reply="ok", echo=False, structured=None):
        self.reply, self.echo, self.structured = reply, echo, structured

    def invoke(self, messages, **kwargs):
        if self.echo:
            return _ai(f"summary({messages[-1].content})")
        return _ai(self.reply)

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, model):
        values = self.structured
        return SimpleNamespace(invoke=lambda messages, **kw: SimpleNamespace(**values))


def banner(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# --- 1. Parallel fan-out + join ----------------------------------------------
# `start` fans out to two workers that run CONCURRENTLY; `combine` is a deferred
# join that runs ONCE after both finish. `a > fanout(b, c) > d` and the bracket
# form `a > [b, c] > d` wire identically.
def demo_parallel():
    banner("1. Parallel fan-out + join:  start > fanout(worker_a, worker_b) > combine")

    start = AgentNode(name="start", llm=FakeLLM(reply="dispatching"))
    worker_a = AgentNode(name="worker_a", llm=FakeLLM(reply="result-A"))
    worker_b = AgentNode(name="worker_b", llm=FakeLLM(reply="result-B"))
    combine = AgentNode(name="combine", llm=FakeLLM(reply="combined A+B"))

    start > fanout(worker_a, worker_b) > combine

    graph = AgenticGraph(state=AgenticState, start_node=start, end_nodes={combine})
    out = graph.invoke({"messages": [HumanMessage(content="go")], "log": []})

    print("log (both branches ran, join ran once):")
    for line in out["log"]:
        print("  -", line)
    return graph  # reused for the summary() demo below


# --- 2. Map-reduce -----------------------------------------------------------
# `summarize.map("docs")` fans the worker out via LangGraph `Send`, once per item
# of state["docs"], all in parallel; `reduce` joins once after. The spread field
# must be a real state channel, so we extend the state with `docs`.
class MapState(AgenticState):
    docs: list


def demo_mapreduce():
    banner('2. Map-reduce:  dispatch > summarize.map("docs") > reduce')

    dispatch = AgentNode(name="dispatch", llm=FakeLLM(reply="dispatching docs"))
    # echo=True -> each parallel worker summarizes its own item (workers must
    # output `messages`, the default).
    summarize = AgentNode(name="summarize", llm=FakeLLM(echo=True))
    reduce = AgentNode(name="reduce", llm=FakeLLM(reply="final digest"))

    dispatch > summarize.map("docs") > reduce

    graph = AgenticGraph(state=MapState, start_node=dispatch, end_nodes={reduce})
    out = graph.invoke({
        "messages": [HumanMessage(content="summarize these")],
        "log": [],
        "docs": ["alpha", "beta", "gamma"],
    })

    summaries = [m.content for m in out["messages"] if m.content.startswith("summary(")]
    print("per-item summaries (one parallel worker per doc):")
    for s in summaries:
        print("  -", s)


# --- 3. Typed multi-key state IO ---------------------------------------------
# `reads` is dispatched by VALUE TYPE: a message-list read becomes conversation
# history; a scalar read is interpolated into node_prompt. Two+ scalar `writes`
# switch the node to structured output (one str field per key).
class ReviewState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    log: Annotated[list[str], operator.add]
    topic: str       # plain entry key (read, never written -> inferred input)
    sentiment: str   # written by the structured node
    score: str


def demo_multikey():
    banner('3. Typed multi-key IO:  AgentNode(reads=["messages","topic"], writes=["sentiment","score"])')

    classify = AgentNode(
        name="classify",
        llm=FakeLLM(structured={"sentiment": "positive", "score": "9"}),
        node_prompt="Rate the {topic} discussion. Return sentiment and score.",
        reads=["messages", "topic"],     # messages -> history, topic -> interpolated scalar
        writes=["sentiment", "score"],   # two scalars -> structured output
    )

    graph = AgenticGraph(state=ReviewState, start_node=classify, end_nodes={classify})
    out = graph.invoke({
        "messages": [HumanMessage(content="loved it")],
        "log": [],
        "topic": "product",
        "sentiment": "",
        "score": "",
    })
    print(f"structured writes -> sentiment={out['sentiment']!r}  score={out['score']!r}")


# --- 4. graph.summary() ------------------------------------------------------
def demo_summary(graph):
    banner("4. graph.summary()  — Keras-style table of reads / writes / available keys")
    graph.summary()


# --- 5. Build-time validation ------------------------------------------------
# `early` reads the computed key `summary`, but its only writer `late` runs
# DOWNSTREAM -> a guaranteed runtime KeyError. The validator catches it at BUILD
# time and FAILS the build with a message that names the key and the real writer.
class SummaryState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    log: Annotated[list[str], operator.add]
    summary: str


def demo_validation():
    banner("5. Build-time validation  — a deliberately broken (read-before-written) graph")

    early = AgentNode(name="early", llm=FakeLLM(), reads=["summary"])   # reads it first...
    late = AgentNode(name="late", llm=FakeLLM(), output_field="summary")  # ...written here, later
    early > late

    try:
        AgenticGraph(state=SummaryState, start_node=early, end_nodes={late})
    except GraphValidationError as e:
        print("caught GraphValidationError at build time:\n")
        print(e)


def main():
    parallel_graph = demo_parallel()
    demo_mapreduce()
    demo_multikey()
    demo_summary(parallel_graph)
    demo_validation()


if __name__ == "__main__":
    main()
