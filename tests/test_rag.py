from app.rag import ingest_pdf, retrieve_context

# Index document
count = ingest_pdf("documents/aapl_2025.pdf")
print(count)

# Retrieve
results = retrieve_context(
    "What was Apple's revenue?"
)

for r in results:
    print(r)