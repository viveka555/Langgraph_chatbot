# 🤖 LangGraph RAG Chatbot

A production-style AI chatbot built with **LangGraph**, **Streamlit**, **Groq LLM**, **ChromaDB**, and **Sentence Transformers**.  
The application supports multi-chat conversations, persistent memory, PDF-based Retrieval-Augmented Generation (RAG), streaming responses, and semantic document search.

---

## Features

- Multi-chat conversation management
- Persistent memory using SQLite checkpointer
- LangGraph workflow orchestration
- Tool calling (Web Search, Calculator, Stock Price, RAG)
- PDF upload and automatic indexing
- ChromaDB vector database
- BGE embedding model
- Cross-Encoder reranking
- Streaming token responses
- Modular production architecture

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Groq (`openai/gpt-oss-20b`) |
| Framework | LangGraph |
| UI | Streamlit |
| Vector DB | ChromaDB |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Memory | SQLite |
| PDF Parsing | PyPDF |

---

## Project Structure

```text
app/
├── __init__.py
├── config.py
├── embeddings.py
├── graph.py
├── llm.py
├── logger.py
├── rag.py
├── repository_sql.py
├── reranker.py
├── schema.py
├── tools.py
└── vector_db.py

database/
├── chat.db
└── chroma/

documents/

streamlit_app.py
requirements.txt
.env
```

---

## LangGraph Workflow

```text
User
 │
 ▼
Planner
 │
 ├────────────── direct ──────────────┐
 │                                     │
 ▼                                     │
Assistant ── tool call? ──► ToolNode ──┘
 │
 ▼
Final Response
```

### Nodes

| Node | Responsibility |
|------|----------------|
| Planner | Decide whether tools are required |
| Assistant | Generate responses & tool calls |
| ToolNode | Execute registered tools |
| SQLite | Persist conversation history |

---

## Available Tools

| Tool | Purpose |
|------|---------|
| `web_search` | Internet search |
| `calculator` | Arithmetic calculations |
| `get_stock_price` | Live stock prices |
| `retrieve_documents` | Search uploaded PDFs |

---

## RAG Pipeline

```text
PDF Upload
      │
      ▼
Extract Text
      │
      ▼
Chunking
      │
      ▼
Embeddings (BGE)
      │
      ▼
ChromaDB
      │
      ▼
Similarity Search
      │
      ▼
Cross-Encoder Reranker
      │
      ▼
LLM Context
```

---

## Installation

### 1. Clone repository

```bash
git clone <your_repo_url>
cd langgraph-rag-chatbot
```

### 2. Create virtual environment

```bash
python -m venv myvenv
```

### Windows

```bash
myvenv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file.

```env
GROQ_API_KEY=your_groq_api_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key

LLM_MODEL=openai/gpt-oss-20b
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

---

## Run Application

```bash
streamlit run streamlit_app.py
```

---

## Testing the RAG System

1. Open the application.
2. Upload a PDF from the sidebar.
3. Wait until indexing completes.
4. Ask grounded questions such as:

- What were Apple's total net sales in 2025?
- What was Services revenue?
- Calculate YoY revenue growth.
- Which page contains the Consolidated Statements of Operations?

---

## Configuration

`app/config.py`

```python
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K_RESULTS = 3
```

These values control document chunking and retrieval.

---

## Production Design Principles

- Modular architecture
- Single Responsibility Principle
- Persistent conversation memory
- Retrieval-Augmented Generation
- Streaming responses
- Config-driven application
- Separation of UI and business logic

---

## Future Improvements

- LangSmith observability
- Conversation summarization
- Hybrid search (BM25 + Dense)
- Citation highlighting
- Authentication
- Docker deployment
- Cloud vector database
- Agent evaluation framework

---

## Author

**Vicky**

Built as a production-style LangGraph & Agentic AI learning project.