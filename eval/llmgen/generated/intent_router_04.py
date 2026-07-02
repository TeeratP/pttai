"""Generated pipeline for task 'intent_router' (sample 4).

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
            "- weather: asks about current/forecast weather, temperature, rain, storms\n"
            "- news: asks for news, headlines, current events\n"
            "- reminder: asks to set, schedule, or remember something (including 'remind me')\n"
            "- small_talk: casual conversation (hi/hello/how are you/thanks/bye/jokes)\n"
            "- unknown: anything else or unclear\n\n"
            "Return only the selected intent."
        ),
        choices=["weather", "news", "reminder", "small_talk", "unknown"],
        input_field="messages",
    )

    weather_responder = AgentNode(
        name="weather_responder",
        llm=llm,
        node_prompt=(
            "You are a helpful assistant for weather queries.\n"
            "Use the user's message to answer with what they asked for.\n"
            "If the location or time frame is missing, ask a concise clarifying question."
        ),
    )

    news_responder = AgentNode(
        name="news_responder",
        llm=llm,
        node_prompt=(
            "You are a helpful assistant for news queries.\n"
            "Answer based on what the user asked (topic and timeframe).\n"
            "If specifics are missing, ask a concise clarifying question."
        ),
    )

    reminder_responder = AgentNode(
        name="reminder_responder",
        llm=llm,
        node_prompt=(
            "You are a helpful assistant that helps the user create reminders.\n"
            "Extract what the user wants to remember and when.\n"
            "If the time/date is missing or ambiguous, ask a concise clarifying question.\n"
            "Then confirm the reminder details in natural language."
        ),
    )

    small_talk_responder = AgentNode(
        name="small_talk_responder",
        llm=llm,
        node_prompt=(
            "You are a friendly assistant for small talk.\n"
            "Respond warmly and appropriately to the user's message."
        ),
    )

    unknown_responder = AgentNode(
        name="unknown_responder",
        llm=llm,
        node_prompt=(
            "You are a general assistant.\n"
            "The user's request doesn't clearly match weather, news, reminders, or small talk.\n"
            "Ask a concise clarifying question or offer a brief menu of what you can help with."
        ),
    )

    router["weather"] > weather_responder
    router["news"] > news_responder
    router["reminder"] > reminder_responder
    router["small_talk"] > small_talk_responder
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
