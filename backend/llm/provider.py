import os
from abc import ABC, abstractmethod

from .schemas import CanonicalStrategy


class LLMProvider(ABC):
    @abstractmethod
    def interpret_strategy(self, user_text: str) -> CanonicalStrategy:
        """Convert user text into the existing deterministic strategy schema."""


class LLMProviderError(RuntimeError):
    """Raised when a provider cannot produce a canonical strategy."""


def get_llm_provider() -> LLMProvider:
    provider_name = os.getenv("LLM_PROVIDER", "mock").strip().lower()
    if provider_name == "mock":
        from .mock_provider import MockLLMProvider

        return MockLLMProvider()
    if provider_name == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider()
    raise ValueError(f"Unsupported LLM_PROVIDER '{provider_name}'.")
