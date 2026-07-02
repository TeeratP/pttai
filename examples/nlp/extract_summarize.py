"""Extract then summarize — typed structured extraction feeding a summary chain.

Two-step NLP pipeline. The first node does structured information extraction:
a dict-form ``writes={...}`` switches it to structured output and returns
NATIVE-typed fields (``severity`` comes back an ``int``). The second node reads
those fields and writes an abstractive one-line ``summary`` to its own state
channel via ``output_field``. The validator confirms the summary node's scalar
reads are all produced upstream before the build compiles.

    python examples/nlp/extract_summarize.py
"""

import os
import sys

_EX = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # examples/
sys.path.insert(0, os.path.dirname(_EX))  # repo root -> `import nae` works from a bare clone
sys.path.insert(0, _EX)  # -> `from _llm import get_llm`
from _llm import get_llm

TICKET = (
    "Subject: App crashes on export. Every time I export a report to PDF the "
    "desktop app freezes and closes. This is blocking my whole team."
)


def nae_version() -> dict:
    from nae import AgentNode, AgenticGraph

    extract = AgentNode(
        llm=get_llm(),
        node_prompt="Extract structured fields from this support ticket: {ticket}",
        reads=["ticket"],                                   # input, interpolated into the prompt
        writes={"product": str, "issue_type": str, "severity": int},  # typed structured output
    )
    summarize = AgentNode(
        llm=get_llm(),
        node_prompt="Write a one-sentence triage summary: a severity-{severity} "
                    "{issue_type} affecting {product}.",
        reads=["product", "issue_type", "severity"],        # scalars produced upstream
        writes=["summary"],
        output_field="summary",                             # final text -> `summary` channel
    )

    extract > summarize                                     # wire the chain

    graph = AgenticGraph(start_node=extract, end_nodes={summarize})
    out = graph.invoke("triage this", ticket=TICKET)
    return {"product": out["product"], "issue_type": out["issue_type"],
            "severity": out["severity"], "summary": out["summary"]}


# --- equivalent in raw LangGraph ---
def langgraph_version() -> dict:
    import operator
    from typing import Annotated
    from typing_extensions import TypedDict
    from langgraph.graph import StateGraph, START, END
    from langchain_core.messages import SystemMessage
    from pydantic import BaseModel

    class Fields(BaseModel):
        product: str
        issue_type: str
        severity: int

    class State(TypedDict):
        ticket: str
        product: str
        issue_type: str
        severity: int
        summary: str
        log: Annotated[list, operator.add]

    llm = get_llm()
    extractor = llm.with_structured_output(Fields)

    def extract(state: State):
        fields = extractor.invoke([SystemMessage(f"Extract structured fields from this support ticket: {state['ticket']}")])
        return {"product": fields.product, "issue_type": fields.issue_type, "severity": fields.severity}

    def summarize(state: State):
        prompt = (f"Write a one-sentence triage summary: a severity-{state['severity']} "
                  f"{state['issue_type']} affecting {state['product']}.")
        return {"summary": llm.invoke([SystemMessage(prompt)]).content}

    builder = StateGraph(State)
    builder.add_node("extract", extract)
    builder.add_node("summarize", summarize)
    builder.add_edge(START, "extract")
    builder.add_edge("extract", "summarize")
    builder.add_edge("summarize", END)
    graph = builder.compile()

    out = graph.invoke({"ticket": TICKET})
    return {"product": out["product"], "issue_type": out["issue_type"],
            "severity": out["severity"], "summary": out["summary"]}


if __name__ == "__main__":
    print("[nae]     ", nae_version())
    print("[langgraph] ", langgraph_version())
