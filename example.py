"""pttai scratch sandbox — runnable script twin of example.ipynb.

Live OpenAI model via langchain_openai. Set OPENAI_API_KEY in the environment or a
`.env` at the repo root, then:

    python example.py

Showcases the schema-free API and the seven shipped features:
  1. parallel fan-out + join        start > fanout(a, b) > combine
  2. per-model token usage          out["token"]                         (#1)
  3. opt-in OpenAI prompt caching    AgenticGraph(prompt_cache=True)       (#2)
  4. map-reduce                      dispatch > summarize.map("docs") > reduce
  5. schema-free typed multi-key IO  writes={"sentiment": str, "score": int}  (#3/#5/#7)
  6. graph.summary()                 the Keras-style table
  7. build-time validation           read-before-write caught on default state (#4/#7)

Local-only — gitignored, like example.ipynb and CLAUDE.md.
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from pttai import AgentNode, AgenticGraph, AgenticState, fanout
from pttai.validation import GraphValidationError

load_dotenv()
llm = ChatOpenAI(model="gpt-5.4-nano")  # stateless per call -> every node reuses it


def banner(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# --- 1+2+3. Parallel fan-out + join, token usage, prompt caching --------------
def demo_parallel():
    banner("1. Parallel fan-out + join   (schema-free, message= shorthand)")

    start = AgentNode(name="start", llm=llm,
                      node_prompt="Restate the user's request in one short line.")
    worker_a = AgentNode(name="worker_a", llm=llm,
                         node_prompt="List two PROS of the topic in the conversation. Be brief.")
    worker_b = AgentNode(name="worker_b", llm=llm,
                         node_prompt="List two CONS of the topic in the conversation. Be brief.")
    combine = AgentNode(name="combine", llm=llm,
                        node_prompt="Weigh the pros and cons above into a one-sentence verdict.")

    start > fanout(worker_a, worker_b) > combine
    # equivalent bracket form:  start > [worker_a, worker_b] > combine

    # no state= -> standard AgenticState;  prompt_cache=True -> opt-in OpenAI caching (#2)
    graph = AgenticGraph(start_node=start, end_nodes={combine}, prompt_cache=True)
    out = graph.invoke(message="Should a small team adopt a monorepo?")  # message= shorthand

    print("final verdict:", out["messages"][-1].content)
    print("\ntrace (both branches ran, join ran once):")
    for line in out["log"]:
        print("  -", line)
    print("\ntoken usage per model (#1):")
    for model, usage in out["token"].items():
        print(f"  {model}: {usage.get('total_tokens')} total "
              f"({usage.get('input_tokens')} in / {usage.get('output_tokens')} out)")
    return graph  # reused by the summary() demo


# --- 4. Map-reduce ------------------------------------------------------------
# `docs` is a real list data-channel (the spread field), so it lives on a small
# custom state extending the standard one — the one place a schema is still earned.
class MapState(AgenticState):
    docs: list


def demo_mapreduce():
    banner('4. Map-reduce   dispatch > summarize.map("docs") > reduce')

    dispatch = AgentNode(name="dispatch", llm=llm,
                         node_prompt="Acknowledge that you'll summarize the documents.")
    summarize = AgentNode(name="summarize", llm=llm,
                          node_prompt="Summarize the message into a single short sentence.")
    reduce = AgentNode(name="reduce", llm=llm,
                       node_prompt="Combine the per-document summaries above into one digest.")

    dispatch > summarize.map("docs") > reduce

    graph = AgenticGraph(state=MapState, start_node=dispatch, end_nodes={reduce})
    out = graph.invoke(
        message="summarize these",
        docs=[
            "LangGraph models agent workflows as a stateful graph of nodes and edges.",
            "Keras became the default high-level API for TensorFlow by winning on ergonomics.",
            "pttai compiles a Pythonic '>' DSL down to a native LangGraph StateGraph.",
        ],
    )
    print("final digest:", out["messages"][-1].content)


# --- 5. Schema-free typed multi-key IO ---------------------------------------
def demo_multikey():
    banner('5. Schema-free typed IO   writes={"sentiment": str, "score": int}')

    # No custom TypedDict: `topic` (read, never written) auto-registers as an input;
    # `sentiment`/`score` (written) auto-register too. dict writes -> typed output.
    classify = AgentNode(
        name="classify",
        llm=llm,
        node_prompt="Rate the {topic} discussion. Return a sentiment and a 1-10 score.",
        reads=["messages", "topic"],
        writes={"sentiment": str, "score": int},
    )

    graph = AgenticGraph(start_node=classify, end_nodes={classify})  # no state=
    out = graph.invoke(message="Honestly I loved it, fast and clean.", topic="product")
    print(f"sentiment={out['sentiment']!r}  score={out['score']!r}  "
          f"(score is a real {type(out['score']).__name__})")


# --- 6. graph.summary() ------------------------------------------------------
def demo_summary(graph):
    banner("6. graph.summary()   — Keras-style reads / writes / available table")
    graph.summary()


# --- 7. Build-time validation ------------------------------------------------
def demo_validation():
    banner("7. Build-time validation   — read-before-write caught on default state")

    early = AgentNode(name="early", llm=llm, reads=["summary"])     # reads it first...
    late = AgentNode(name="late", llm=llm, output_field="summary")  # ...written here, later
    early > late

    try:
        AgenticGraph(start_node=early, end_nodes={late})  # no state=
        print("no error (unexpected)")
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
