# =====================================================
# app/llm.py
# LLM Factory
# Single responsibility:
# Create and expose the application LLM instance
# =====================================================

from langchain_groq import ChatGroq

from app.config import (
    GROQ_API_KEY,
    LLM_MODEL
)

# Shared LLM instance used across the application
llm = ChatGroq(
    model=LLM_MODEL,
    api_key=GROQ_API_KEY,
    streaming=True
)