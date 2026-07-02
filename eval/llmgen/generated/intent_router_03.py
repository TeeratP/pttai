"""Generated pipeline for task 'intent_router' (sample 3).

TASK: Classify a chat message into one of several intents -- weather, news, reminder, small talk, or unknown -- and route it to a matching responder.
"""

from pttai import AgentNode, DecisionNode, AgenticGraph


def build_graph(llm):
    router = DecisionNode(
        name="router",
        llm=llm,
        node_prompt=(
            "Classify the user's message into exactly ONE intent.\n"
            "Intents:\n"
            "- weather: asks about weather, forecasts, temperature, rain, etc.\n"
            "- news: asks for news, headlines, current events.\n"
            "- reminder: asks to set, schedule, or remember something.\n"
            "- small_talk: greetings, jokes, compliments, general conversation.\n"
            "- unknown: none of the above.\n\n"
            "Return the chosen intent label only."
        ),
        choices=["weather", "news", "reminder", "small_talk", "unknown"],
        input_field="messages",
        reads=["messages"],
    )

    weather = AgentNode(
        name="responder_weather",
        llm=llm,
        node_prompt=(
            "You are a helpful assistant that responds to weather questions.\n"
            "If the user did not specify a location, ask a clarifying question.\n"
            "Keep the response concise and friendly."
        ),
        reads=["messages"],
    )

    news = AgentNode(
        name="responder_news",
        llm=llm,
        node_prompt=(
            "You are a helpful assistant that responds to news requests.\n"
            "If no topic or region is specified, ask a clarifying question.\n"
            "Keep the response concise and friendly."
        ),
        reads=["messages"],
    )

    reminder = AgentNode(
        name="responder_reminder",
        llm=llm,
        node_prompt=(
            "You are a helpful assistant for creating reminders.\n"
            "Extract the reminder content and any time/date/timezone details.\n"
            "If the user hasn't provided enough scheduling info, ask follow-up questions.\n"
            "Confirm what will be remembered and when."
        ),
        reads=["messages"],
    )

    small_talk = AgentNode(
        name="responder_small_talk",
        llm=llm,
        node_prompt=(
            "You are a friendly assistant for small talk.\n"
            "Be personable, concise, and helpful. If appropriate, ask a gentle follow-up."
        ),
        reads=["messages"],
    )

    unknown = AgentNode(
        name="responder_unknown",
        llm=llm,
        node_prompt=(
            "You are a helpful assistant when the user's intent is unclear.\n"
            "Ask what they need in a short, clarifying way, offering examples (weather, news, reminders, small talk)."
        ),
        reads=["messages"],
    )

    router["weather"] > weather
    router["news"] > news
    router["reminder"] > reminder
    router["small_talk"] > small_talk
    router["unknown"] > unknown

    return AgenticGraph(start_node=router, end_nodes={weather, news, reminder, small_talk, unknown})
