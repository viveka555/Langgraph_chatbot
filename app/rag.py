# =====================================================
# app/rag.py
# RAG Pipeline
# Single Responsibility:
# PDF → Chunks → Chroma → Retrieval
# =====================================================

import logging
from pathlib import Path
from pypdf import PdfReader

from app.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS,
)
from app.vector_db import (
    add_documents,
    delete_source,
    similarity_search,
)
from app.reranker import rerank_documents


# -----------------------------------------------------
# Logger
# -----------------------------------------------------

logger = logging.getLogger(__name__)


# -----------------------------------------------------
# Step 1: Extract Text
# -----------------------------------------------------

def extract_text(pdf_path: str) -> list[dict]:
    """
    Extract text from every page of a PDF.

    Returns:
        [
            {
                "page": 1,
                "text": "..."
            }
        ]
    """

    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text()

        if text:
            pages.append(
                {
                    "page": page_number,
                    "text": text.strip(),
                }
            )

    logger.info("Extracted %s pages", len(pages))

    return pages


# -----------------------------------------------------
# Step 2: Create Chunks
# -----------------------------------------------------

def create_chunks(
    pages: list[dict],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split pages into overlapping chunks.

    Returns:
        [
            {
                "page": 3,
                "text": "..."
            }
        ]
    """

    if overlap >= chunk_size:
        raise ValueError(
            "Overlap must be smaller than chunk size."
        )

    chunks = []

    for page in pages:

        text = page["text"]
        start = 0

        while start < len(text):

            end = start + chunk_size

            chunks.append(
                {
                    "page": page["page"],
                    "text": text[start:end],
                }
            )

            start += chunk_size - overlap

    logger.info("Created %s chunks", len(chunks))

    return chunks


# -----------------------------------------------------
# Step 3: Ingest PDF
# -----------------------------------------------------

def ingest_pdf(pdf_path: str) -> int:
    """
    Complete ingestion pipeline.

    Returns:
        Number of stored chunks.
    """

    source = Path(pdf_path).name

    logger.info("Ingesting %s", source)

    pages = extract_text(pdf_path)

    chunks = create_chunks(pages)

    # Remove previous version
    delete_source(source)

    # Store vectors
    add_documents(
        chunks=chunks,
        source=source,
    )

    logger.info("Stored %s chunks", len(chunks))

    return len(chunks)


# -----------------------------------------------------
# Step 4: Retrieve Context
# -----------------------------------------------------

def retrieve_context(
    query: str,
    top_k: int = TOP_K_RESULTS,
) -> list[dict]:
    """
    Retrieve the most relevant chunks.

    Returns:
        [
            {
                "text": "...",
                "page": 24,
                "source": "apple.pdf"
            }
        ]
    """

    results = similarity_search(
        query=query,
        top_k=top_k,
    )

    contexts = []

    for document, metadata in zip(
        results["documents"][0],
        results["metadatas"][0],
    ):
        contexts.append(
            {
                "text": document,
                "page": metadata.get("page", "Unknown"),
                "source": metadata.get("source", "Unknown"),
            }
        )
    # NEW
    contexts = rerank_documents(
    query=query,
    documents=contexts,
    top_k=3,
    )


    logger.info("Retrieved %s chunks", len(contexts))

    return contexts