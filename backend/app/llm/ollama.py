from typing import List

import httpx

from ..config import get_settings
from .base import LLMProvider


class OllamaProvider(LLMProvider):
    """LLM provider that calls a local Ollama instance."""

    def __init__(self) -> None:
        settings = get_settings()
        self.base_url = settings.ollama_base_url
        self.model = settings.llm_model

    async def generate(self, messages: List[dict]) -> str:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=60.0) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]


