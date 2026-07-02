"""Sample (clean): extract -> summarize chain. Should NOT be flagged.

Stand-in for a real generated pipeline; reused shape from examples/basics.
"""
from pttai import AgentNode, AgenticGraph


def build_graph(llm):
    extract = AgentNode(name="extract", llm=llm,
                        node_prompt="Extract the key entities and claims from the document.")
    summarize = AgentNode(name="summarize", llm=llm,
                          node_prompt="Write a short summary using the extracted information above.")
    extract > summarize
    return AgenticGraph(start_node=extract, end_nodes={summarize})
