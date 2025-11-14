from typing import List, Tuple
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from ..config import get_settings
from ..llm.ollama import OllamaProvider


settings = get_settings()
_embedder: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def get_qdrant() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def ensure_collection() -> None:
    client = get_qdrant()
    if settings.qdrant_collection not in [
        c.name for c in client.get_collections().collections
    ]:
        dim = get_embedder().get_sentence_embedding_dimension()
        client.recreate_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=dim, distance="Cosine"),
        )


def store_document(doc_id: str, text: str) -> None:
    ensure_collection()
    client = get_qdrant()
    embedder = get_embedder()
    vector = embedder.encode(text).tolist()
    # Qdrant IDs must be unsigned integers or UUIDs. We store the original
    # document identifier in the payload and use a generated UUID as the point ID.
    point = PointStruct(
        id=str(uuid4()),
        vector=vector,
        payload={"text": text, "doc_id": doc_id},
    )
    client.upsert(collection_name=settings.qdrant_collection, points=[point])


def retrieve_documents(query: str, k: int = 5) -> List[dict]:
    ensure_collection()
    client = get_qdrant()
    embedder = get_embedder()
    q_vec = embedder.encode(query).tolist()
    search_result = client.search(
        collection_name=settings.qdrant_collection,
        query_vector=q_vec,
        limit=k,
        query_filter=Filter(must=[]),
    )
    docs: List[dict] = []
    for r in search_result:
        payload = r.payload or {}
        docs.append({"id": str(r.id), "text": payload.get("text", "")})
    return docs


async def answer_with_context(question: str) -> Tuple[str, List[str]]:
    """Run a simple RAG pipeline: retrieve docs, call LLM with context."""
    docs = retrieve_documents(question, k=5)
    context = "\n\n".join(d["text"] for d in docs)
    retrieved_ids = [d["id"] for d in docs]

    messages = [
        {
            "role": "system",
            "content": "You are a helpful personal assistant. Use the provided context when it is relevant.",
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        },
    ]

    llm = OllamaProvider()
    reply = await llm.generate(messages)
    return reply, retrieved_ids


