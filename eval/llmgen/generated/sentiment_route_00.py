"""Generated pipeline for task 'sentiment_route' (sample 0).

TASK: Classify the sentiment of a product review as positive or negative, then route to a handler that writes an appropriate response.
"""

from pttai import AgentNode, DecisionNode, AgenticGraph


def build_graph(llm):
    classify = DecisionNode(
        name="classify",
        llm=llm,
        node_prompt=(
            "Classify the sentiment of the user's product review as either "
            "'positive' or 'negative'.\n\n"
            "Return exactly one of the choices."
        ),
        choices=["positive", "negative"],
    )

    positive_handler = AgentNode(
        name="positive_handler",
        llm=llm,
        node_prompt=(
            "Write a brief, friendly response to the user based on their POSITIVE review.\n"
            "Do not mention that you are doing sentiment classification.\n"
            "Be specific: reference at least one strength from the review.\n"
            "Respond helpfully and professionally."
        ),
    )

    negative_handler = AgentNode(
        name="negative_handler",
        llm=llm,
        node_prompt=(
            "Write a brief, empathetic response to the user based on their NEGATIVE review.\n"
            "Do not mention that you are doing sentiment classification.\n"
            "Acknowledge the issue and suggest a constructive next step.\n"
            "Respond helpfully and professionally."
        ),
    )

    # Route based on DecisionNode choice
    classify["positive"] > positive_handler
    classify["negative"] > negative_handler

    return AgenticGraph(start_node=classify, end_nodes={positive_handler, negative_handler})
