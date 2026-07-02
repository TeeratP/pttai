"""Sample (buggy): read-before-write. `summarize` reads `draft` before its
producer runs. Validator flags at build; raw LangGraph KeyErrors at runtime."""
from pttai import AgentNode, AgenticGraph


def build_graph(llm):
    start = AgentNode(name="start", llm=llm, node_prompt="Start.")
    summarize = AgentNode(name="summarize", llm=llm, node_prompt="Summarize {draft}",
                          reads=["draft"], writes=["messages"])
    drafter = AgentNode(name="drafter", llm=llm, node_prompt="Write a draft.",
                        writes={"draft": str})
    start > summarize > drafter          # summarize reads `draft` before drafter runs
    return AgenticGraph(start_node=start, end_nodes={drafter})
