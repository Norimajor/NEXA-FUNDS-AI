import json
import os
import socket
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from backend.engine.strategy_interpreter.models import StrategyDefinition

from .openai_provider import CANONICAL_STRATEGY_JSON_SCHEMA, SYSTEM_INSTRUCTIONS, OpenAIProvider
from .provider import LLMProviderError

DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_TIMEOUT = 60


class OllamaProvider(OpenAIProvider):
    """LLM provider backed by Ollama's local HTTP API."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        opener: Callable[..., Any] | None = None,
    ):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL") or DEFAULT_MODEL
        configured_timeout = os.getenv("OLLAMA_TIMEOUT")
        try:
            self.timeout = timeout if timeout is not None else float(configured_timeout or DEFAULT_TIMEOUT)
        except (TypeError, ValueError) as exc:
            raise LLMProviderError("OLLAMA_TIMEOUT must be a positive number.") from exc
        if self.timeout <= 0:
            raise LLMProviderError("OLLAMA_TIMEOUT must be a positive number.")
        self._opener = opener or urlopen

    def interpret_strategy(self, user_text: str) -> StrategyDefinition:
        if not isinstance(user_text, str) or not user_text.strip():
            raise LLMProviderError("Strategy interpretation failed: strategy text is required.")

        request_payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": user_text.strip()},
            ],
            "stream": False,
            "format": CANONICAL_STRATEGY_JSON_SCHEMA,
        }
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(request_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener(request, timeout=self.timeout) as response:
                raw_response = response.read()
            response_payload = json.loads(raw_response.decode("utf-8"))
            content = response_payload["message"]["content"]
            payload = json.loads(content)
            return self._build_strategy(payload)
        except LLMProviderError:
            raise
        except (HTTPError, URLError, TimeoutError, socket.timeout, OSError) as exc:
            raise LLMProviderError(f"Ollama connection failed: {exc}") from exc
        except (KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
            raise LLMProviderError(f"Ollama returned a malformed strategy response: {exc}") from exc
        except Exception as exc:
            raise LLMProviderError(f"Ollama strategy interpretation failed: {exc}") from exc
