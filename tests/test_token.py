"""Per-model `token` state channel: AgentNode emits usage, the reducer sums it."""

from langchain_core.messages import HumanMessage

from nae import fanout
from nae.graph import AgenticGraph
from nae.nodes import AgentNode
from nae.state import merge_token_usage


def usage(i, o):
    """A usage_metadata breakdown like a real AIMessage carries."""
    return {"input_tokens": i, "output_tokens": o, "total_tokens": i + o,
            "input_token_details": {"audio": 0, "cache_read": 0},
            "output_token_details": {"audio": 0, "reasoning": 0}}


def double(x: int) -> int:
    """Double x."""
    return x * 2


# --- reducer unit tests --------------------------------------------------

def test_reducer_unions_and_deep_sums():
    left = {"m1": usage(10, 5)}
    right = {"m1": usage(20, 7), "m2": usage(1, 1)}
    merged = merge_token_usage(left, right)
    assert merged["m1"]["input_tokens"] == 30
    assert merged["m1"]["output_tokens"] == 12
    assert merged["m1"]["total_tokens"] == 42
    # nested *_details are deep-summed too
    assert merged["m1"]["input_token_details"]["cache_read"] == 0
    assert merged["m2"]["total_tokens"] == 2


def test_reducer_commutative_and_handles_none_left():
    left = {"m1": usage(10, 5)}
    right = {"m1": usage(20, 7)}
    assert merge_token_usage(left, right) == merge_token_usage(right, left)
    # first update: a missing/None left is treated as {}
    assert merge_token_usage(None, {"m1": usage(1, 1)}) == {"m1": usage(1, 1)}


# --- AgentNode emission --------------------------------------------------

def test_single_call_records_usage(t):
    llm = t.FakeLLM(responses=[t.ai("ok", usage_metadata=usage(10, 5), model_name="m1")])
    node = AgentNode(name="n", llm=llm)
    delta = node({"messages": [HumanMessage(content="hi")], "log": []})
    assert delta["token"] == {"m1": usage(10, 5)}


def test_tool_loop_deep_sums(t):
    # The tool-call loop fires two invokes in one turn; both report usage on the
    # same model -> the node's delta deep-sums them.
    first = t.ai(tool_calls=[{"name": "double", "args": {"x": 2}, "id": "c1", "type": "tool_call"}],
                 usage_metadata=usage(10, 5), model_name="m1")
    second = t.ai("done", usage_metadata=usage(20, 7), model_name="m1")
    node = AgentNode(name="n", llm=t.FakeLLM(responses=[first, second]), tools=[double])
    delta = node({"messages": [HumanMessage(content="hi")], "log": []})
    assert delta["token"]["m1"]["input_tokens"] == 30
    assert delta["token"]["m1"]["output_tokens"] == 12
    assert delta["token"]["m1"]["total_tokens"] == 42


def test_missing_usage_contributes_nothing(t):
    # A fake with no usage_metadata -> the node emits no `token` key, and the
    # channel stays empty.
    n = AgentNode(name="n", llm=t.FakeLLM(responses=[t.ai("ok")]))
    delta = n({"messages": [HumanMessage(content="hi")], "log": []})
    assert "token" not in delta
    g = AgenticGraph(start_node=n, end_nodes=n)
    n2 = AgentNode(name="n2", llm=t.FakeLLM(responses=[t.ai("ok")]))
    g2 = AgenticGraph(start_node=n2, end_nodes=n2)
    out = g2.invoke({"messages": ["hi"], "log": []})
    assert out.get("token", {}) == {}


# --- end-to-end through a graph -----------------------------------------

def test_invoke_without_seeding_token(t):
    # Invoking WITHOUT seeding `token` works (the channel defaults to {}).
    n = AgentNode(name="n", llm=t.FakeLLM(responses=[t.ai("ok", usage_metadata=usage(3, 2), model_name="m1")]))
    g = AgenticGraph(start_node=n, end_nodes=n)
    out = g.invoke({"messages": ["hi"], "log": []})
    assert out["token"] == {"m1": usage(3, 2)}


def test_parallel_fanout_sums_per_model(t):
    # A fan-out of two workers, each calling the same model, sums per-model
    # totals across the join.
    a = AgentNode(name="a", llm=t.FakeLLM(responses=[t.ai("a", usage_metadata=usage(5, 5), model_name="m1")]))
    b = AgentNode(name="b", llm=t.FakeLLM(responses=[t.ai("b", usage_metadata=usage(10, 10), model_name="m1")]))
    c = AgentNode(name="c", llm=t.FakeLLM(responses=[t.ai("c", usage_metadata=usage(20, 20), model_name="m1")]))
    d = AgentNode(name="d", llm=t.FakeLLM(responses=[t.ai("d", usage_metadata=usage(1, 1), model_name="m1")]))
    a > fanout(b, c) > d
    g = AgenticGraph(start_node=a, end_nodes={d})
    out = g.invoke({"messages": ["hi"], "log": []})
    assert out["token"]["m1"]["input_tokens"] == 36   # 5+10+20+1
    assert out["token"]["m1"]["output_tokens"] == 36
    assert out["token"]["m1"]["total_tokens"] == 72
