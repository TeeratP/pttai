"""Generated pipeline for task 'intent_router' (sample 0).

TASK: Classify a chat message into one of several intents -- weather, news, reminder, small talk, or unknown -- and route it to a matching responder.
"""

from pttai import AgentNode, DecisionNode, AgenticGraph


def build_graph(llm):
    router = DecisionNode(
        name="router",
        llm=llm,
        node_prompt=(
            "Classify the user's last message into exactly ONE intent.\n"
            "Intents:\n"
            "- weather: asking about weather/forecast/temperature/rain/snow\n"
            "- news: asking about news/headlines/current events\n"
            "- reminder: asking to remember/set a reminder/schedule/alert\n"
            "- small talk: greetings, thanks, jokes, general conversation\n"
            "- unknown: anything else\n\n"
            "Return only the intent label."
        ),
        choices=["weather", "news", "reminder", "small talk", "unknown"],
        input_field="messages",
        # We keep all nodes simple: each responder reads/writes only via messages history.
    )

    weather_responder = AgentNode(
        name="weather_responder",
        llm=llm,
        node_prompt=(
            "You are a helpful assistant for weather queries.\n"
            "Given the conversation so far, respond with a concise, actionable weather answer.\n"
            "If the user didn't specify a location/time, ask a clarifying question."
        ),
    )

    news_responder = AgentNode(
        name="news_responder",
        llm=llm,
        node_prompt=(
            "You are a helpful assistant for news queries.\n"
            "Respond with concise, relevant news-style information or a clarifying question "
            "about what topic/region/timeframe the user wants."
        ),
    )

    reminder_responder = AgentNode(
        name="reminder_responder",
        llm=llm,
        node_prompt=(
            "You are a helpful assistant for reminders.\n"
            "Extract the user's requested reminder details (what, when, and any recurrence).\n"
            "If details are missing, ask targeted follow-up questions."
        ),
    )

    small_talk_responder = AgentNode(
        name="small_talk_responder",
        llm=llm,
        node_prompt=(
            "You are a friendly assistant for small talk.\n"
            "Respond warmly and naturally, staying on topic with the user's message."
        ),
    )

    unknown_responder = AgentNode(
        name="unknown_responder",
        llm=llm,
        node_prompt=(
            "You are a helpful assistant.\n"
            "The user's request doesn't clearly match weather, news, or reminders.\n"
            "Ask a brief clarifying question and offer examples of what you can help with."
        ),
    )

    router["weather"] > weather_responder
    router["news"] > news_responder
    router["reminder"] > reminder_responder
    router["small talk"] > small_talk_responder
    router["unknown"] > unknown_responder

    return AgenticGraph(
        start_node=router,
        end_nodes={
            weather_responder,
            news_responder,
            reminder_responder,
            small_talk_responder,
            unknown_responder,
        },
    )
