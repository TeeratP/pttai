"""Generated pipeline for task 'sentiment_route' (sample 2).

TASK: Classify the sentiment of a product review as positive or negative, then route to a handler that writes an appropriate response.
"""

from pttai import AgentNode, DecisionNode, AgenticGraph


def build_graph(llm):
    classify = DecisionNode(
        name="classify",
        llm=llm,
        node_prompt=(
            "Classify the sentiment of the product review as exactly one of: "
            "'positive' or 'negative'. "
            "Use the review content provided in the conversation messages."
        ),
        choices=["positive", "negative"],
        input_field="messages",
    )

    positive_handler = AgentNode(
        name="positive_handler",
        llm=llm,
        node_prompt=(
            "Write a friendly response to the customer based on their review. "
            "The sentiment is POSITIVE. Be concise, appreciative, and invite "
            "them to share more details if relevant.\n\nReview is in the conversation."
        ),
        writes={"response": str},
        reads=["messages"],
        output_field="response",
    )

    negative_handler = AgentNode(
        name="negative_handler",
        llm=llm,
        node_prompt=(
            "Write a helpful response to the customer based on their review. "
            "The sentiment is NEGATIVE. Be empathetic, acknowledge concerns, "
            "and suggest next steps or questions to resolve the issue.\n\n"
            "Review is in the conversation."
        ),
        writes={"response": str},
        reads=["messages"],
        output_field="response",
    )

    classify["positive"] > positive_handler
    classify["negative"] > negative_handler

    return AgenticGraph(start_node=classify, end_nodes={positive_handler, negative_handler})
