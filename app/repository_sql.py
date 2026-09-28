# =====================================================
# app/repository.py
# SQLite helper functions
# =====================================================

from langgraph.graph.state import CompiledStateGraph


def load_chat_history(
    chatbot: CompiledStateGraph,
    thread_id: str
):

    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    if not state.values:
        return []

    return state.values.get("messages", [])


def load_chat_sessions(checkpointer):

    chats = {}
    seen = set()

    for cp in checkpointer.list(None):

        thread_id = cp.config["configurable"]["thread_id"]

        if thread_id in seen:
            continue

        seen.add(thread_id)

        values = cp.checkpoint.get("channel_values", {})

        chats[thread_id] = {
            "title": values.get("title", "New Chat")
        }

    return chats


def delete_chat(checkpointer, thread_id: str):

    checkpointer.delete_thread(thread_id)