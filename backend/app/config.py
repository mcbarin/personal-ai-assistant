from functools import lru_cache
from typing import Literal, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    api_token: Optional[str] = None

    # Database
    database_url: str = "postgresql+psycopg2://assistant:assistant@db:5432/assistant"

    # LLM provider
    llm_provider: Literal["ollama"] = "ollama"
    llm_model: str = "llama3"
    ollama_base_url: str = "http://host.docker.internal:11434"

    # Qdrant / vector store
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "notes"

    # Google Calendar
    google_credentials_file: str = "google_credentials.json"
    google_token_file: str = "google_token.json"
    google_calendar_id: str = "primary"

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()


