# =====================================================
# app/embeddings.py
# Embedding Service
# Single Responsibility:
# Convert text into dense vector embeddings
# =====================================================

from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL

# -----------------------------------------------------
# Load embedding model once (Singleton)
# -----------------------------------------------------

embedding_model = SentenceTransformer(EMBEDDING_MODEL)


# -----------------------------------------------------
# Create Single Embedding
# -----------------------------------------------------

def create_embedding(text: str) -> list[float]:
    """
    Convert one text string into a normalized embedding.

    Args:
        text: User query or sentence.

    Returns:
        List[float]: Dense embedding vector.
    """

    vector = embedding_model.encode(
        text,
        normalize_embeddings=True
    )

    return vector.tolist()


# -----------------------------------------------------
# Create Multiple Embeddings
# -----------------------------------------------------

def create_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Convert multiple document chunks into embeddings.

    Args:
        texts: List of text chunks.

    Returns:
        List[List[float]]: Embedding vectors.
    """

    vectors = embedding_model.encode(
        texts,
        normalize_embeddings=True
    )

    return vectors.tolist()