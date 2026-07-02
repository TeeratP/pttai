"""Generated pipeline for task 'sentiment_route' (sample 3).

TASK: Classify the sentiment of a product review as positive or negative, then route to a handler that writes an appropriate response.
"""

from pttai import AgentNode, DecisionNode, AgenticGraph


def build_graph(llm):
    classify = DecisionNode(
        name="classify",
        llm=llm,
        node_prompt=(
            "Classify the sentiment of the user's product review.\n"
            "Return exactly one choice: 'positive' or 'negative'.\n\n"
            "Review:\n{messages}"
        ),
        choices=["positive", "negative"],
        input_field="messages",
        reads=["messages"],
    )

    positive_handler = AgentNode(
        name="positive_handler",
        llm=llm,
        node_prompt=(
            "Write a friendly, appropriate response to the customer.\n"
            "The sentiment is POSITIVE.\n"
            "Review:\n{messages}"
        ),
        input_field="messages",
        output_field="messages",
        reads=["messages"],
    )

    negative_handler = AgentNode(
        name="negative_handler",
        llm=llm,
        node_prompt=(
            "Write a helpful, empathetic response to the customer.\n"
            "The sentiment is NEGATIVE.\n"
            "Offer a resolution or next step.\n"
            "Review:\n{messages}"
        ),
        input_field="messages",
        output_field="messages",
        reads=["messages"],
    )

    classify["positive"] > positive_handler
    classify["negative"] > negative_handler

    return AgenticGraph(start_node=classify, end_nodes={positive_handler, negative_handler})
