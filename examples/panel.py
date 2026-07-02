"""panel.py — a multi-agent panel that fans out to rival personas, IN PARALLEL.

A question goes to `frame` (which sharpens it into one concrete decision), fans
out to three personas — optimist / skeptic / pragmatist — who argue
CONCURRENTLY, then `verdict` weighs every argument into a one-paragraph ruling.
Shows nae's differentiators in ~20 lines: the `>` wiring, `fanout(...)`
parallel branches with a deferred join, schema-free state, `summary()`, and
per-model token totals from `out["token"]`.

Run:   python examples/panel.py
Needs: OPENAI_API_KEY  (in your environment or a .env file)
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from nae import AgentNode, AgenticGraph, fanout

load_dotenv()
llm = ChatOpenAI(model="gpt-5.4-nano")

frame = AgentNode(name="frame", llm=llm, node_prompt=(
    "Restate the user's question as ONE sharp, concrete decision to be made. "
    "One sentence, no preamble."))

optimist = AgentNode(name="optimist", llm=llm, node_prompt=(
    "You are a relentless optimist. Argue FOR the bold move. Give the two "
    "strongest upside reasons. Two sentences, no hedging."))
skeptic = AgentNode(name="skeptic", llm=llm, node_prompt=(
    "You are a hard-nosed skeptic. Argue AGAINST. Name the two biggest risks "
    "or failure modes. Two sentences, no hedging."))
pragmatist = AgentNode(name="pragmatist", llm=llm, node_prompt=(
    "You are a pragmatist. Ignore both the hype and the fear. Propose the "
    "smallest concrete next step that de-risks the decision. Two sentences."))

verdict = AgentNode(name="verdict", llm=llm, node_prompt=(
    "You are the chair. Weigh the optimist, skeptic, and pragmatist above and "
    "deliver a balanced one-paragraph verdict with a clear recommendation."))

# The line that matters: the three personas run IN PARALLEL, then join at `verdict`.
frame > fanout(optimist, skeptic, pragmatist) > verdict

panel = AgenticGraph(start_node=frame, end_nodes={verdict})  # schema-free: default state

QUESTION = (
    "Should an early-stage SaaS startup rewrite its monolith into microservices "
    "to win enterprise customers?")
out = panel.invoke(message=QUESTION)

print("\nVERDICT\n", out["messages"][-1].content)
panel.summary()
print("\nTOKENS PER MODEL")
for model, usage in out["token"].items():
    print(f"  {model}: {usage['total_tokens']} total "
          f"({usage['input_tokens']} in / {usage['output_tokens']} out)")
