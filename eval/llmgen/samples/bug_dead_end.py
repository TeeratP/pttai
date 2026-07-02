"""Sample (buggy): dead-end node. `writeup` fans out from `plan` but is left
out of end_nodes and has no outgoing edge -> a dangling terminal. The build
rejects it; raw LangGraph would just halt silently on that branch."""
from pttai import AgentNode, AgenticGraph, fanout


def build_graph(llm):
    plan = AgentNode(name="plan", llm=llm, node_prompt="Plan the research.")
    research = AgentNode(name="research", llm=llm, node_prompt="Do the research.")
    writeup = AgentNode(name="writeup", llm=llm, node_prompt="Write it up.")
    plan > fanout(research, writeup)     # writeup has no child and is not an end node
    return AgenticGraph(start_node=plan, end_nodes={research})
