# =====================================================
# app/reranker.py
# Cross Encoder Reranker
# =====================================================

from sentence_transformers import CrossEncoder

# Load once
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank_documents(
    query: str,
    documents: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """
    Reorder retrieved documents using a Cross Encoder.

    Args:
        query: User question
        documents: Retrieved chunks from Chroma
        top_k: Final documents to return

    Returns:
        Top reranked documents
    """

    pairs = [
        (query, doc["text"])
        for doc in documents
    ]

    scores = reranker.predict(pairs)

    for doc, score in zip(documents, scores):
        doc["rerank_score"] = float(score)

    documents.sort(
        key=lambda x: x["rerank_score"],
        reverse=True,
    )

    return documents[:top_k]