"""Generated pipeline for task 'sentiment_route' (sample 1).

TASK: Classify the sentiment of a product review as positive or negative, then route to a handler that writes an appropriate response.
"""

from pttai import AgentNode, DecisionNode, AgenticGraph


def build_graph(llm):
    router = DecisionNode(
        name="router",
        llm=llm,
        node_prompt="Classify the sentiment of the product review as positive or negative.",
        choices=["positive", "negative"],
        input_field="messages",
    )

    positive = AgentNode(
        name="positive_handler",
        llm=llm,
        node_prompt=(
            "You are a helpful customer support agent.\n"
            "The review sentiment is POSITIVE.\n"
            "Write an appropriate brief response that thanks the customer and highlights the product's strengths. "
            "Keep it friendly and concise."
        ),
    )

    negative = AgentNode(
        name="negative_handler",
        llm=llm,
        node_prompt=(
            "You are a helpful customer support agent.\n"
            "The review sentiment is NEGATIVE.\n"
            "Write an appropriate brief response that acknowledges the issue, apologizes, and suggests a practical next step "
            "(e.g., troubleshooting, replacement, or contacting support). Keep it concise."
        ),
    )

    # Route via LLM decision node.
    router["positive"] > positive
    router["negative"] > negative

    return AgenticGraph(start_node=router, end_nodes={positive, negative})
