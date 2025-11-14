from abc import ABC, abstractmethod
from typing import List


class LLMProvider(ABC):
    """Abstract base class for chat LLM providers."""

    @abstractmethod
    async def generate(self, messages: List[dict]) -> str:
        """Generate a reply given a list of chat messages."""
        raise NotImplementedError


