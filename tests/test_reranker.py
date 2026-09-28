from app.rag import retrieve_context
from app.reranker import rerank_documents

query = "What was Apple's total revenue in 2025?"

docs = retrieve_context(query, top_k=10)

best = rerank_documents(query, docs)

for doc in best:
    print("=" * 40)
    print(doc["rerank_score"])
    print(doc["page"])
    print(doc["text"][:150])