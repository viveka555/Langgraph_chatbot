# =====================================================
# app/config.py
# Central configuration for the application
# =====================================================

from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# =====================================================
# Project Root
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# =====================================================
# Folder Paths
# =====================================================

DATABASE_DIR = BASE_DIR / "database"
DOCUMENTS_DIR = BASE_DIR / "documents"
CHROMA_DIR = DATABASE_DIR / "chroma"

# Create folders automatically
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# SQLite
# =====================================================

SQLITE_DB = DATABASE_DIR / "chat.db"

# =====================================================
# Model Configuration
# =====================================================

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "openai/gpt-oss-20b"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-small-en-v1.5"
)

# =====================================================
# API Keys
# =====================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

ALPHA_VANTAGE_API_KEY = os.getenv(
    "ALPHA_VANTAGE_API_KEY"
)

# =====================================================
# RAG Configuration
# =====================================================

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

TOP_K_RESULTS = 10
DOCUMENTS_DIR = Path("documents")

# =====================================================
# Chat Configuration
# =====================================================

DEFAULT_CHAT_TITLE = "New Chat"

MAX_CHAT_TITLE_LENGTH = 25

PLANNER_PROMPT = """You are a routing agent.

Return ONLY valid JSON.

Schema:
{
  "route": "tool" | "direct",
  "reason": "short explanation"
}

Important:
- route must be a STRING.
- Never return an array like ["direct"]."""

# -----------------------------------------------------
# Environment Validation
# -----------------------------------------------------

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found in .env"
    )