"""Sample (buggy): prompt-placeholder mismatch. `reader`'s prompt interpolates
`{subject}` but only `topic` is a declared read -> guaranteed runtime KeyError.
Validator flags at build."""
from pttai import AgentNode, AgenticGraph


def build_graph(llm):
    prod = AgentNode(name="prod", llm=llm, node_prompt="Produce a topic.",
                     writes={"topic": str})
    reader = AgentNode(name="reader", llm=llm, node_prompt="Write about {subject}",
                       reads=["topic"])          # {subject} is not a declared read
    prod > reader
    return AgenticGraph(start_node=prod, end_nodes={reader})
