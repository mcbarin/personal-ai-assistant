from typing import List, Tuple

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Qdrant as LCQdrant
from qdrant_client import QdrantClient

from .config import get_settings


settings = get_settings()

_embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
_client = QdrantClient(url=settings.qdrant_url)

# Tell LangChain which payload key contains the text we stored during ingestion.
_vectorstore = LCQdrant(
    client=_client,
    collection_name=settings.qdrant_collection,
    embeddings=_embeddings,
    content_payload_key="text",
)

_retriever = _vectorstore.as_retriever(search_kwargs={"k": 5})


def answer_with_context_langchain(question: str) -> Tuple[str, List[str]]:
    """RAG using LangChain's Qdrant retriever + ChatOllama.

    Returns (reply, retrieved_doc_ids).
    """
    # LangChain retrievers in newer versions are Runnable-like and use .invoke;
    # fall back to get_relevant_documents for older versions.
    if hasattr(_retriever, "invoke"):
        docs = _retriever.invoke(question)
    else:
        docs = _retriever.get_relevant_documents(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    retrieved_ids: List[str] = [
        str(doc.metadata.get("doc_id", "")) for doc in docs
    ]

    system_prompt = (
        "You are a concise personal assistant.\n"
        "Use the provided context only as factual background.\n"
        "Always answer the user's question directly and do not ask follow-up "
        "questions about their goals or intentions unless absolutely necessary."
    )

    llm = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.llm_model,
    )

    messages = [
        ("system", system_prompt),
        (
            "user",
            f"Context:\n{context}\n\nQuestion: {question}",
        ),
    ]

    response = llm.invoke(messages)
    reply = response.content if hasattr(response, "content") else str(response)
    return reply, retrieved_ids


