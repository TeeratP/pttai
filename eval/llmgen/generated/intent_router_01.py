"""Generated pipeline for task 'intent_router' (sample 1).

TASK: Classify a chat message into one of several intents -- weather, news, reminder, small talk, or unknown -- and route it to a matching responder.
"""

from pttai import AgentNode, DecisionNode, AgenticGraph


def build_graph(llm):
    # Router: constrained by structured-output choices
    route_intent = DecisionNode(
        name="route_intent",
        llm=llm,
        node_prompt=(
            "Classify the user's last message into exactly one intent.\n"
            "Intents:\n"
            "- weather: asks about weather, forecasts, temperature, rain, snow, wind\n"
            "- news: asks for news, headlines, events, top stories\n"
            "- reminder: requests to set, schedule, or remember something at a time\n"
            "- small_talk: greetings, jokes, compliments, 'how are you', casual conversation\n"
            "- unknown: anything else or unclear\n\n"
            "Return only the best matching intent."
        ),
        choices=["weather", "news", "reminder", "small_talk", "unknown"],
        input_field="messages",
    )

    # Responder nodes (each uses the full conversation history)
    responder_weather = AgentNode(
        name="responder_weather",
        llm=llm,
        node_prompt=(
            "You are a helpful assistant for weather queries.\n"
            "Answer the user's request clearly. If location or timeframe is missing, ask a brief follow-up question."
        ),
        writes={"messages": "messages"},
    )

    responder_news = AgentNode(
        name="responder_news",
        llm=llm,
        node_prompt=(
            "You are a helpful assistant for news requests.\n"
            "Provide a concise summary of the news the user asks for.\n"
            "If the user doesn't specify topics or regions, ask one short clarifying question."
        ),
        writes={"messages": "messages"},
    )

    responder_reminder = AgentNode(
        name="responder_reminder",
        llm=llm,
        node_prompt=(
            "You are a reminder assistant.\n"
            "If the user requests a reminder, confirm what to remember and when.\n"
            "If time/date is missing, ask a short follow-up to determine it."
        ),
        writes={"messages": "messages"},
    )

    responder_small_talk = AgentNode(
        name="responder_small_talk",
        llm=llm,
        node_prompt=(
            "You are a friendly assistant for small talk.\n"
            "Respond warmly and naturally, then (if helpful) ask what else the user needs."
        ),
        writes={"messages": "messages"},
    )

    responder_unknown = AgentNode(
        name="responder_unknown",
        llm=llm,
        node_prompt=(
            "You are a general assistant.\n"
            "The intent was unclear or not matched.\n"
            "Ask a brief clarifying question and offer what kinds of help you can provide (weather, news, reminders, or general chat)."
        ),
        writes={"messages": "messages"},
    )

    route_intent["weather"] > responder_weather
    route_intent["news"] > responder_news
    route_intent["reminder"] > responder_reminder
    route_intent["small_talk"] > responder_small_talk
    route_intent["unknown"] > responder_unknown

    return AgenticGraph(start_node=route_intent, end_nodes={responder_weather, responder_news, responder_reminder, responder_small_talk, responder_unknown})
