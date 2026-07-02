"""Sample (buggy): concurrent write to a reducer-less key. Parallel `b` and `c`
both write `result`, which has no reducer -> InvalidUpdateError at runtime.
Validator flags at build."""
from pttai import AgentNode, AgenticGraph, fanout, AgenticState


class _State(AgenticState):
    result: str                                  # plain key: no reducer


def build_graph(llm):
    a = AgentNode(name="a", llm=llm, node_prompt="Start.")
    b = AgentNode(name="b", llm=llm, node_prompt="Branch b.", output_field="result")
    c = AgentNode(name="c", llm=llm, node_prompt="Branch c.", output_field="result")
    j = AgentNode(name="j", llm=llm, node_prompt="Join.", reads=["result"])
    (a > fanout(b, c)) > j                        # b and c both write `result` in parallel
    return AgenticGraph(state=_State, start_node=a, end_nodes={j})
