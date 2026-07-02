"""Generated pipeline for task 'dialogue_next_turn' (sample 1).

TASK: From a customer conversation, work out the requested action and the details still needed, then write the next agent turn.
"""

from typing import Optional

from pttai import AgentNode, AgenticGraph, DecisionNode


def build_graph(llm):
    # 1) Frame: extract the requested action + structured details needed
    frame = AgentNode(
        name="frame",
        llm=llm,
        node_prompt=(
            "You are a customer-support action planner.\n"
            "From the conversation so far, determine:\n"
            "1) The requested action the customer wants (one concise action verb phrase).\n"
            "2) The concrete details that are still needed to perform that action.\n"
            "3) Any assumptions you can safely make vs. questions you must ask.\n\n"
            "Return your output as:\n"
            "- Action: <...>\n"
            "- Needed details (bullet list): <...>\n"
            "- Next questions (bullet list): <...>\n"
            "- Draft plan: <short steps>\n\n"
            "Conversation:\n{messages}"
        ),
        reads=["messages"],
        writes={"action": str, "needed_details": str, "next_questions": str, "draft_plan": str},
        input_field="messages",
    )

    # 2) Decide whether we can proceed or must ask questions
    route = DecisionNode(
        name="route",
        llm=llm,
        node_prompt=(
            "Decide what should happen next given the framed action-planning output.\n"
            "Choose exactly one.\n\n"
            "If the needed details are already sufficient to proceed without asking the customer,"
            " choose 'proceed'.\n"
            "If important required details are missing and you must ask the customer,"
            " choose 'ask_questions'.\n\n"
            "Framed context:\n"
            "Action: {action}\n"
            "Needed details: {needed_details}\n"
            "Next questions: {next_questions}\n"
            "Draft plan: {draft_plan}"
        ),
        choices=["ask_questions", "proceed"],
        reads=["action", "needed_details", "next_questions", "draft_plan"],
    )

    # 3a) Ask questions turn
    ask_agent = AgentNode(
        name="ask_agent",
        llm=llm,
        node_prompt=(
            "You are a helpful customer-support agent.\n"
            "Write the NEXT agent turn to the customer.\n\n"
            "You MUST:\n"
            "- Acknowledge the requested action.\n"
            "- Ask ONLY the missing details needed to proceed.\n"
            "- Keep it concise and clear.\n"
            "- If you have optional clarifications, phrase them as optional.\n\n"
            "Action: {action}\n"
            "Needed details (still needed): {needed_details}\n"
            "Next questions (required): {next_questions}\n"
            "Draft plan: {draft_plan}"
        ),
        reads=["action", "needed_details", "next_questions", "draft_plan"],
        writes=["messages"],
    )

    # 3b) Proceed turn
    proceed_agent = AgentNode(
        name="proceed_agent",
        llm=llm,
        node_prompt=(
            "You are a customer-support agent.\n"
            "Write the NEXT agent turn.\n\n"
            "You MUST:\n"
            "- Confirm the requested action.\n"
            "- Provide what you can do now (a clear short plan).\n"
            "- Call out any assumptions you are making.\n"
            "- If there are still minor optional questions, include them at the end as optional.\n\n"
            "Action: {action}\n"
            "Needed details: {needed_details}\n"
            "Next questions: {next_questions}\n"
            "Draft plan: {draft_plan}\n\n"
            "Conversation context:\n{messages}"
        ),
        reads=["action", "needed_details", "next_questions", "draft_plan", "messages"],
        writes=["messages"],
    )

    # Wire routing
    route["ask_questions"] > ask_agent
    route["proceed"] > proceed_agent

    # Frame feeds route; route then feeds either ask or proceed.
    frame > route

    return AgenticGraph(start_node=frame, end_nodes={ask_agent, proceed_agent})
