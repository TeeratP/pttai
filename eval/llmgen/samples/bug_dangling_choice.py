"""Sample (buggy): dangling choice. The `technical` branch is never wired.
Validator flags at build; raw LangGraph KeyErrors when that branch is taken."""
from pttai import AgentNode, DecisionNode, AgenticGraph


def build_graph(llm):
    triage = DecisionNode(name="triage", llm=llm,
                          node_prompt="Classify the support message.",
                          choices=["billing", "technical"])
    billing = AgentNode(name="billing", llm=llm, node_prompt="Draft a billing reply.")
    triage["billing"] > billing          # `technical` choice left unwired
    return AgenticGraph(start_node=triage, end_nodes={billing})
