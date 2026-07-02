"""Issue #2: auto OpenAI prompt_cache_key (opt-in, per-run or user-supplied)."""

from nae.graph import AgenticGraph
from nae.nodes import AgentNode


def test_kwarg_passed_when_openai_and_enabled(t):
    llm = t.ChatOpenAI(responses=[t.ai("ok")])
    n = AgentNode(name="n", llm=llm)
    g = AgenticGraph(start_node=n, end_nodes=n, prompt_cache=True)
    g.invoke({"messages": ["hi"], "log": []})
    assert "prompt_cache_key" in llm.last_kwargs
    assert llm.last_kwargs["prompt_cache_key"]  # non-empty auto key


def test_kwarg_absent_when_disabled(t):
    llm = t.ChatOpenAI(responses=[t.ai("ok")])
    n = AgentNode(name="n", llm=llm)
    g = AgenticGraph(start_node=n, end_nodes=n)  # prompt_cache off (default)
    g.invoke({"messages": ["hi"], "log": []})
    assert "prompt_cache_key" not in llm.last_kwargs


def test_kwarg_absent_for_non_openai(t):
    llm = t.FakeLLM(responses=[t.ai("ok")])  # not OpenAI
    n = AgentNode(name="n", llm=llm)
    g = AgenticGraph(start_node=n, end_nodes=n, prompt_cache=True)
    g.invoke({"messages": ["hi"], "log": []})
    assert "prompt_cache_key" not in llm.last_kwargs


def test_user_supplied_key_used_verbatim(t):
    llm = t.ChatOpenAI(responses=[t.ai("ok")])
    n = AgentNode(name="n", llm=llm)
    g = AgenticGraph(start_node=n, end_nodes=n,
                     prompt_cache=True, prompt_cache_key="my-key")
    g.invoke({"messages": ["hi"], "log": []})
    assert llm.last_kwargs["prompt_cache_key"] == "my-key"


def test_auto_key_stable_within_run(t):
    # Two OpenAI nodes in one graph share the same auto key in a run.
    a = AgentNode(name="a", llm=t.ChatOpenAI(responses=[t.ai("a")]))
    b = AgentNode(name="b", llm=t.ChatOpenAI(responses=[t.ai("b")]))
    a > b
    g = AgenticGraph(start_node=a, end_nodes=b, prompt_cache=True)
    g.invoke({"messages": ["hi"], "log": []})
    assert a.llm.last_kwargs["prompt_cache_key"] == b.llm.last_kwargs["prompt_cache_key"]
