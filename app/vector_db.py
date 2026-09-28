# =====================================================
# app/vector_db.py
# Vector Repository
# Single Responsibility:
# Store and retrieve embeddings from ChromaDB
# =====================================================

import chromadb

from app.config import CHROMA_DIR, TOP_K_RESULTS
from app.embeddings import (
    create_embedding,
    create_embeddings
)

# -----------------------------------------------------
# Chroma Client
# -----------------------------------------------------

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_or_create_collection(
    name="documents",
    metadata={
        "hnsw:space": "cosine"
    }
)

# -----------------------------------------------------
# Store Document Chunks
# -----------------------------------------------------

def add_documents(
    chunks: list[dict],
    source: str
) -> None:
    """
    Store PDF chunks with embeddings and metadata.

    Args:
        chunks : List of dictionaries
                 [{"page":1, "text":"..."}]
        source : PDF filename
    """

    texts = [chunk["text"] for chunk in chunks]

    embeddings = create_embeddings(texts)

    ids = [
        f"{source}_{i}"
        for i in range(len(chunks))
    ]

    metadatas = [
        {
            "source": source,
            "page": chunk["page"],
            "chunk": i
        }
        for i, chunk in enumerate(chunks)
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

# -----------------------------------------------------
# Semantic Search
# -----------------------------------------------------

def similarity_search(
    query: str,
    top_k: int = TOP_K_RESULTS
) -> dict:
    """
    Retrieve the most relevant document chunks.

    Args:
        query : User question
        top_k : Number of results

    Returns:
        Chroma query result
    """

    query_vector = create_embedding(query)

    return collection.query(
        query_embeddings=[query_vector],
        n_results=top_k
    )

# -----------------------------------------------------
# Delete Source
# -----------------------------------------------------

def delete_source(source: str) -> None:
    """
    Remove all vectors belonging to one PDF.
    """

    collection.delete(
        where={
            "source": source
        }
    )