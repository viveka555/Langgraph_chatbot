# =====================================================
# app/semantic_memory.py
# Production Semantic Memory
# =====================================================

import chromadb

from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import CHROMA_DIR
from app.embeddings import create_embedding
from app.llm import llm

# =====================================================
# Chroma Collection
# =====================================================

client = chromadb.PersistentClient(path=str(CHROMA_DIR))

memory_collection = client.get_or_create_collection(
    name="memories",
    metadata={"hnsw:space": "cosine"},
)

# =====================================================
# Schema
# =====================================================


class MemoryFact(BaseModel):
    category: str
    fact: str


# =====================================================
# Extract Semantic Facts
# =====================================================


def extract_memories(messages) -> list[MemoryFact]:
    """
    Extract stable user facts from a conversation.

    Only store:
    - preferences
    - goals
    - projects
    - long-term information

    Ignore greetings and temporary questions.
    """

    conversation = "\n".join(
        f"{msg.type}: {msg.content}" for msg in messages
    )

    extractor = llm.with_structured_output(list[MemoryFact])

    facts = extractor.invoke(
        [
            SystemMessage(
                content="""
You are a semantic memory extractor.

Extract ONLY long-term user facts.

Valid categories:
- preference
- project
- goal
- profile

Ignore:
- greetings
- temporary questions
- assistant responses
- factual explanations

Return an empty list if nothing should be remembered.
"""
            ),
            HumanMessage(content=conversation),
        ]
    )

    return facts


# =====================================================
# Store Memories
# =====================================================


def store_memories(memories: list[MemoryFact], thread_id: str):
    """
    Save semantic memories into Chroma.
    """

    if not memories:
        return

    documents = [m.fact for m in memories]

    embeddings = [create_embedding(text) for text in documents]

    ids = [
        f"{thread_id}_{i}_{abs(hash(text))}"
        for i, text in enumerate(documents)
    ]

    metadatas = [
        {
            "thread_id": thread_id,
            "category": memory.category,
        }
        for memory in memories
    ]

    memory_collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


# =====================================================
# Retrieve Relevant Memories
# =====================================================


def retrieve_memories(query: str, thread_id: str, top_k: int = 3) -> list[str]:
    """
    Retrieve semantic memories relevant to the current query.
    """

    results = memory_collection.query(
        query_embeddings=[create_embedding(query)],
        n_results=top_k,
        where={"thread_id": thread_id},
    )

    if not results["documents"]:
        return []

    return results["documents"][0]