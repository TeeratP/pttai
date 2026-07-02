"""Sample (clean): parallel personas fan out then join at a verdict. Not flagged.

Writes go to the reduced `messages` channel, so the parallel join is safe.
"""
from pttai import AgentNode, AgenticGraph, fanout


def build_graph(llm):
    frame = AgentNode(name="frame", llm=llm, node_prompt="Restate as one sharp decision.")
    optimist = AgentNode(name="optimist", llm=llm, node_prompt="Argue for the bold move.")
    skeptic = AgentNode(name="skeptic", llm=llm, node_prompt="Argue against it.")
    pragmatist = AgentNode(name="pragmatist", llm=llm, node_prompt="Propose the smallest step.")
    verdict = AgentNode(name="verdict", llm=llm, node_prompt="Weigh all three into a verdict.")
    frame > fanout(optimist, skeptic, pragmatist) > verdict
    return AgenticGraph(start_node=frame, end_nodes={verdict})
