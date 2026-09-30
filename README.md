# 🤖 Production LangGraph RAG Chatbot

A production-style AI chatbot built with **LangGraph**, **Streamlit**, **Groq LLM**, **ChromaDB**, **Sentence Transformers**, and **SQLite**.

The project demonstrates how to build a modular, state-driven AI application with:

- LangGraph workflow orchestration
- Multi-chat conversations
- Persistent conversation history
- Conversation summarization
- Token-aware context management
- Semantic long-term memory
- PDF-based Retrieval-Augmented Generation (RAG)
- Dense vector search
- Cross-Encoder reranking
- Tool calling
- Streaming responses
- Production-oriented project structure

The project is designed as a practical **Agentic AI + RAG engineering project** rather than a simple chatbot demo.

---

## 🚀 Features

### 🤖 Agentic Workflow

- LangGraph-based stateful workflow
- Planner-based routing
- Direct conversation handling
- Tool-based execution
- LangGraph `ToolNode`
- Automatic tool selection through LLM tool binding

### 🧠 Memory

- Persistent conversation history
- SQLite checkpointing
- Conversation summarization
- Token-aware context management
- Semantic long-term memory
- Vector-based memory retrieval
- Thread-specific memories

### 📚 RAG

- PDF document upload
- PDF text extraction
- Configurable chunking
- BGE embeddings
- ChromaDB vector storage
- Similarity search
- Cross-Encoder reranking
- Context-grounded responses

### 🛠 Tools

- Web search
- Calculator
- Stock price lookup
- Document retrieval

### ⚡ Application

- Streamlit UI
- Streaming LLM responses
- Multiple chat sessions
- Chat deletion
- Persistent chat state
- Modular backend architecture

---

# 🏗️ Architecture

The application is divided into several independent responsibilities.

```text
                         ┌─────────────────┐
                         │     User        │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │     Planner     │
                         │                 │
                         │ tool / direct   │
                         └────────┬────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                 direct                       tool
                    │                           │
                    ▼                           ▼
             ┌─────────────┐            ┌─────────────┐
             │   Memory    │            │   Memory    │
             └──────┬──────┘            └──────┬──────┘
                    │                           │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │    Assistant    │
                         │                 │
                         │ LLM + Tools     │
                         └────────┬────────┘
                                  │
                            Tool Call?
                                  │
                         ┌────────┴────────┐
                         │                 │
                        No                Yes
                         │                 │
                         │                 ▼
                         │          ┌─────────────┐
                         │          │   ToolNode  │
                         │          └──────┬──────┘
                         │                 │
                         │                 ▼
                         │          Tool Result
                         │                 │
                         │                 ▼
                         │          ┌─────────────┐
                         └─────────►│  Assistant  │
                                    └──────┬──────┘
                                           │
                                           ▼
                                    Final Response