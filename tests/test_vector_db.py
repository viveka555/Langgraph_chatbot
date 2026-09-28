from app.vector_db import (
    add_documents,
    similarity_search,
    delete_source
)

chunks = [
    "Revenue: Apple's total revenue increased by 15 percent year over year during FY2025.",

    "Profit: Net profit reached 25 billion dollars in FY2025.",

    "Innovation: The company launched several new AI-powered products."
]

delete_source("apple_demo")

add_documents(
    chunks=chunks,
    source="apple_demo"
)

results = similarity_search(
    "How much did revenue grow?",
    top_k=3
)

for doc, distance in zip(
    results["documents"][0],
    results["distances"][0]
):
    print(f"Distance: {distance:.4f}")
    print(doc)
    print("-" * 50)