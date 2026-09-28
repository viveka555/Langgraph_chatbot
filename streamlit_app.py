
import uuid
import streamlit as st

from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessageChunk

# LangGraph
from app.graph import chatbot, checkpointer

# SQLite Repository
from app.repository_sql import (
    load_chat_history,
    load_chat_sessions,
    delete_chat,
)

# RAG
from app.rag import ingest_pdf
from app.config import DOCUMENTS_DIR


# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 LangGraph Chatbot")
st.caption("Built with LangGraph + Streamlit")


# =====================================================
# Session State Initialization
# =====================================================

if "chats" not in st.session_state:
    st.session_state.chats = load_chat_sessions(checkpointer)

if "current_thread" not in st.session_state:

    if st.session_state.chats:
        st.session_state.current_thread = next(
            iter(st.session_state.chats)
        )
    else:
        thread_id = str(uuid.uuid4())[:8]
        st.session_state.current_thread = thread_id
        st.session_state.chats[thread_id] = {
            "title": "New Chat"
        }

current_chat = st.session_state.chats[
    st.session_state.current_thread
]


# =====================================================
# Sidebar
# =====================================================

with st.sidebar:

    st.header("💬 Chats")

    # Resume + Delete Chats
    for thread_id, chat in list(st.session_state.chats.items()):

        col1, col2 = st.columns([5, 1])

        with col1:
            if st.button(
                chat["title"],
                key=f"chat_{thread_id}",
                use_container_width=True,
            ):
                st.session_state.current_thread = thread_id
                st.rerun()

        with col2:
            if st.button(
                "🗑️",
                key=f"delete_{thread_id}",
                use_container_width=True,
            ):

                delete_chat(checkpointer, thread_id)

                st.session_state.chats.pop(thread_id, None)

                if st.session_state.chats:
                    st.session_state.current_thread = next(
                        iter(st.session_state.chats)
                    )
                else:
                    new_id = str(uuid.uuid4())[:8]
                    st.session_state.current_thread = new_id
                    st.session_state.chats[new_id] = {
                        "title": "New Chat"
                    }

                st.rerun()

    st.divider()

    # New Chat
    if st.button("➕ New Chat", use_container_width=True):

        new_id = str(uuid.uuid4())[:8]

        st.session_state.current_thread = new_id

        st.session_state.chats[new_id] = {
            "title": "New Chat"
        }

        st.rerun()

    # =================================================
    # Knowledge Base
    # =================================================

    st.divider()
    st.subheader("📄 Knowledge Base")

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    if "last_uploaded" not in st.session_state:
        st.session_state.last_uploaded = None

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
    )

    if uploaded_file:

        save_path = DOCUMENTS_DIR / uploaded_file.name

        if st.session_state.last_uploaded != uploaded_file.name:

            save_path.write_bytes(uploaded_file.getbuffer())

            try:
                with st.spinner("Creating embeddings..."):
                    chunks = ingest_pdf(str(save_path))

                st.session_state.last_uploaded = uploaded_file.name

                st.success(
                    f"Indexed {chunks} chunks successfully!"
                )

            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    pdfs = list(DOCUMENTS_DIR.glob("*.pdf"))
    st.caption(f"Indexed PDFs: {len(pdfs)}")


# =====================================================
# Load Conversation History
# =====================================================

messages = load_chat_history(
    chatbot,
    st.session_state.current_thread,
)

for msg in messages:

    role = "user" if msg.type == "human" else "assistant"

    with st.chat_message(role):
        st.markdown(msg.content)


# =====================================================
# Chat Input
# =====================================================

user_input = st.chat_input("Type your message...")

if user_input and user_input.strip():

    # LangGraph runtime config
    config = {
        "configurable": {
            "thread_id": st.session_state.current_thread
        }
    }

    # Display User Message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate chat title once
    if current_chat["title"] == "New Chat":

        current_chat["title"] = (
            user_input[:25] + "..."
            if len(user_input) > 25
            else user_input
        )

        st.session_state.chats[
            st.session_state.current_thread
        ] = current_chat

    # Stream Assistant
    full_response = ""

    with st.chat_message("assistant"):

        placeholder = st.empty()

        for message, metadata in chatbot.stream(
            {
                "messages": [
                    HumanMessage(content=user_input)
                ],
                "title": current_chat["title"],
            },
            config=config,
            stream_mode="messages",
        ):

            # Ignore planner/tool messages
            if metadata.get("langgraph_node") != "assistant":
                continue

            if isinstance(message, AIMessageChunk):
                full_response += message.content
                placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

    # Refresh UI so SQLite history is reloaded
    st.rerun()