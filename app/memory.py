# =====================================================
# app/memory.py
# Production Conversation Memory
# =====================================================
from langchain_core.messages import HumanMessage, SystemMessage
from app.llm import llm


def summarize_conversation(messages):
    conversation = "%n".join(
        f"{msg.type}: {msg.content}"
        for msg in messages
    )

    response = llm.invoke(
        [
            SystemMessage(
                content = """
            Summarize the conversation.

            Preserve:
            - user goals
            - important decisions
            - uploaded documents
            - technical context

            Keep it under 250 words.
            """
            ),
            HumanMessage(content=conversation),
        ]
    )
    return response.content


def trim_messages(messages, max_tokens):

    selected = []
    total = 0

    for msg in reversed(messages):

        tokens = estimate_tokens(msg.content)

        if total + tokens > max_tokens:
            break

        selected.append(msg)
        total += tokens

    return list(reversed(selected))



def build_context(summary, recent_message):

    context = []

    if summary:
        context.append(
            SystemMessage(
                content=f"""Conversation summary:{summary}"""
            )
        )
    context.extend(recent_message)
    return context

def estimate_tokens(text: str) -> int:
    """
    Rough token estimate.

    1 token ≈ 4 characters
    """
    return len(text) // 4