"""Generated pipeline for task 'intent_router' (sample 2).

TASK: Classify a chat message into one of several intents -- weather, news, reminder, small talk, or unknown -- and route it to a matching responder.
"""

from pttai import AgentNode, DecisionNode, AgenticGraph


def build_graph(llm):
    classify = DecisionNode(
        name="classify",
        llm=llm,
        node_prompt=(
            "Classify the user's intent into ONE label from the allowed choices.\n"
            "Rules:\n"
            "- weather: asks about current/forecast weather, temperature, rain, etc.\n"
            "- news: asks for headlines, breaking news, current events.\n"
            "- reminder: asks to remember something, set a reminder, schedule, or be alerted.\n"
            "- small_talk: greetings, thanks, jokes, general chit-chat.\n"
            "- unknown: anything else or unclear.\n\n"
            "Return only the label."
        ),
        choices=["weather", "news", "reminder", "small_talk", "unknown"],
        input_field="messages",
        reads=["messages"],
    )

    weather = AgentNode(
        name="weather_responder",
        llm=llm,
        node_prompt=(
            "You are a helpful assistant specialized in weather.\n"
            "Given the conversation, ask clarifying questions if needed (location, time frame), "
            "and provide a concise response."
        ),
        reads=["messages"],
    )

    news = AgentNode(
        name="news_responder",
        llm=llm,
        node_prompt=(
            "You are a helpful assistant specialized in news.\n"
            "Given the conversation, respond with relevant headlines/topics. "
            "If no specific source/topic is provided, ask a brief clarifying question."
        ),
        reads=["messages"],
    )

    reminder = AgentNode(
        name="reminder_responder",
        llm=llm,
        node_prompt=(
            "You are a helpful assistant for setting reminders.\n"
            "Extract the reminder content and any time/date expressions.\n"
            "If anything is missing, ask the minimum clarifying question needed.\n"
            "Confirm what you will remind the user about."
        ),
        reads=["messages"],
    )

    small_talk = AgentNode(
        name="small_talk_responder",
        llm=llm,
        node_prompt=(
            "You are a friendly assistant for small talk.\n"
            "Respond naturally and helpfully. Keep it brief unless the user asks for more."
        ),
        reads=["messages"],
    )

    unknown = AgentNode(
        name="unknown_responder",
        llm=llm,
        node_prompt=(
            "You are a general assistant.\n"
            "The user's request doesn't clearly match known intents. "
            "Ask a short clarifying question and offer to help."
        ),
        reads=["messages"],
    )

    classify["weather"] > weather
    classify["news"] > news
    classify["reminder"] > reminder
    classify["small_talk"] > small_talk
    classify["unknown"] > unknown

    return AgenticGraph(start_node=classify, end_nodes={weather, news, reminder, small_talk, unknown})
