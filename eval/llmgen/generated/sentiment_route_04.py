"""Generated pipeline for task 'sentiment_route' (sample 4).

TASK: Classify the sentiment of a product review as positive or negative, then route to a handler that writes an appropriate response.
"""

from pttai import AgentNode, DecisionNode, AgenticGraph


def build_graph(llm):
    # Classify sentiment into a constrained choice.
    classify = DecisionNode(
        name="classify",
        llm=llm,
        node_prompt=(
            "Classify the sentiment of the following product review as either "
            '"positive" or "negative". Respond with only the label.'
        ),
        choices=["positive", "negative"],
        reads=["messages"],
        input_field="messages",
    )

    # Handlers: write an appropriate response into the conversation.
    positive_handler = AgentNode(
        name="positive_handler",
        llm=llm,
        node_prompt=(
            "You are a customer-support assistant. The review is positive.\n\n"
            "Review:\n{review}\n\n"
            "Write a brief, friendly response thanking the customer and highlighting"
            " two specific positives mentioned (or, if none are explicit, the most likely strengths)."
        ),
        reads=["review"],
        writes=["messages"],
    )

    negative_handler = AgentNode(
        name="negative_handler",
        llm=llm,
        node_prompt=(
            "You are a customer-support assistant. The review is negative.\n\n"
            "Review:\n{review}\n\n"
            "Write a brief response acknowledging the issue, apologizing, and offering"
            " one concrete next step to resolve it."
        ),
        reads=["review"],
        writes=["messages"],
    )

    # Ensure both handlers have access to the review text from the input messages.
    # We ask the LLM to extract a single scalar "review" from the conversation history,
    # so {review} interpolation works and avoids placeholder read mismatches.
    extract_review = AgentNode(
        name="extract_review",
        llm=llm,
        node_prompt=(
            "Extract the product review text from the conversation.\n"
            "Return only the review text as a single string.\n"
            "If multiple messages exist, use the last user message content."
        ),
        reads=["messages"],
        writes=["review"],
        # output_field default is messages; but we override by writes=["review"]
        # which makes the node write the extracted string to state["review"].
    )

    # Wiring:
    extract_review > classify
    classify["positive"] > positive_handler
    classify["negative"] > negative_handler

    return AgenticGraph(start_node=extract_review, end_nodes={positive_handler, negative_handler})
