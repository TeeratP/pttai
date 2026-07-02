"""
Sample usage of the Agentic Framework.

Builds a small graph that wires three node types with the `>` DSL:

    randomizer (AgentNode + tool)
          │
          ▼
    classifier (DecisionNode)
        ╱        ╲
   "positive"   "negative"
      ▼              ▼
  positive_handler  negative_handler   (AgentNodes)

The randomizer calls a Python tool, the classifier routes on the result, and
one of the two handlers produces the final reply. The shared state carries the
conversation (`messages`), a human-readable trace (`log`), and the routing
label (`decision_classifier`, the classifier node's per-node channel).

Setup:
    pip install -e ".[openai]"        # langchain-openai + python-dotenv
    echo "OPENAI_API_KEY=sk-..." > .env   # or copy .env.example

Run:
    python examples/sample_usage.py
"""

import random

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from nae.graph import AgenticGraph
from nae.nodes import AgentNode, DecisionNode
from nae.state import AgenticState


def random_number(maximum: int) -> int:
    """Return a random integer between 1 and `maximum` (inclusive)."""
    return random.randint(1, maximum)


def build_graph(llm) -> AgenticGraph:
    # An agent that can call the random_number tool. tools= wraps the bare
    # function as a StructuredTool and runs the tool-call loop automatically.
    randomizer = AgentNode(
        name="randomizer",
        llm=llm,
        node_prompt="Use the random_number tool to pick a number based on the "
                    "user's request, then state the number you got.",
        tools=[random_number],
    )

    # A decision node returns one of `choices` (structured output) into its
    # per-node `decision_{name}` state field; the framework routes on it.
    classifier = DecisionNode(
        name="classifier",
        llm=llm,
        node_prompt="If the number mentioned is greater than 5, answer "
                    "'positive'. Otherwise answer 'negative'.",
        choices=["positive", "negative"],
    )

    positive_handler = AgentNode(
        name="positive_handler",
        llm=llm,
        node_prompt="Report the number to the user in an upbeat, celebratory tone.",
        reasoning_effort="low",  # gpt-5.x reasoning effort, passed per call
    )
    negative_handler = AgentNode(
        name="negative_handler",
        llm=llm,
        node_prompt="Report the number to the user in a gloomy, melodramatic tone.",
        reasoning_effort="low",
    )

    # Wire the graph with the `>` operator.
    randomizer > classifier
    classifier["positive"] > positive_handler
    classifier["negative"] > negative_handler

    return AgenticGraph(
        state=AgenticState,
        start_node=randomizer,
        end_nodes={positive_handler, negative_handler},
    )


def main() -> None:
    load_dotenv()
    llm = ChatOpenAI(model="gpt-5.4-nano")
    graph = build_graph(llm)

    # `log` is conventionally seeded with [] so every node appends a trace line.
    initial_state = {
        "messages": [HumanMessage(content="Give me a random number up to 10.")],
        "log": [],
    }

    result = graph.invoke(initial_state)

    print("routed to :", result["decision_classifier"])
    print("final reply:", result["messages"][-1].content)
    print("\ntrace:")
    for line in result["log"]:
        print("  -", line)

    # To watch execution node-by-node instead of waiting for the final state:
    #     for update in graph.stream(initial_state):
    #         print(update)   # {node_name: delta} as each node completes


if __name__ == "__main__":
    main()
