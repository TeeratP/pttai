"""Sample (clean): triage via DecisionNode with every choice wired. Not flagged."""
from pttai import AgentNode, DecisionNode, AgenticGraph


def build_graph(llm):
    triage = DecisionNode(name="triage", llm=llm,
                          node_prompt="Classify the support message.",
                          choices=["billing", "technical", "other"])
    billing = AgentNode(name="billing", llm=llm, node_prompt="Draft a billing reply.")
    technical = AgentNode(name="technical", llm=llm, node_prompt="Draft a technical reply.")
    other = AgentNode(name="other", llm=llm, node_prompt="Draft a general reply.")
    triage["billing"] > billing
    triage["technical"] > technical
    triage["other"] > other
    return AgenticGraph(start_node=triage, end_nodes={billing, technical, other})
