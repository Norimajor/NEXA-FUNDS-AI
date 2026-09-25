import json
import os
import unittest
from unittest.mock import patch

from backend.llm.ollama_provider import OllamaProvider
from backend.llm.provider import LLMProviderError, get_llm_provider


def strategy_payload():
    return {
        "name": "Local strategy",
        "symbol": "XAUUSD",
        "timeframe": "M15",
        "direction": "long",
        "entry_conditions": [],
        "exit_conditions": [],
        "filters": [],
        "risk_percent": 1.0,
        "risk_reward": None,
        "allow_reentry": True,
        "max_simultaneous_positions": 1,
        "pyramiding": False,
        "cooldown_bars": 0,
        "exit_policy": "unspecified",
        "stop_loss": None,
        "take_profit": None,
        "position_sizing": "percent_risk",
        "instrument_spec": {},
    }


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


class TestOllamaProvider(unittest.TestCase):
    def test_valid_response_uses_configured_model_and_builds_strategy(self):
        calls = []

        def opener(request, timeout):
            calls.append((request, timeout))
            return FakeResponse({"message": {"content": json.dumps(strategy_payload())}})

        strategy = OllamaProvider(model="test-model", opener=opener).interpret_strategy("Buy XAUUSD.")

        self.assertEqual(strategy.symbol, "XAUUSD")
        self.assertEqual(calls[0][0].full_url, "http://127.0.0.1:11434/api/chat")
        self.assertEqual(json.loads(calls[0][0].data)["model"], "test-model")

    def test_timeout_and_connection_errors_are_explicit(self):
        for error in (TimeoutError("timed out"), ConnectionError("refused")):
            with self.subTest(error=error):
                with self.assertRaisesRegex(LLMProviderError, "Ollama connection failed"):
                    OllamaProvider(opener=lambda *_args, **_kwargs: (_ for _ in ()).throw(error)).interpret_strategy(
                        "Buy XAUUSD."
                    )

    def test_malformed_response_is_rejected(self):
        provider = OllamaProvider(opener=lambda *_args, **_kwargs: FakeResponse({"message": {"content": "not json"}}))
        with self.assertRaisesRegex(LLMProviderError, "malformed strategy response"):
            provider.interpret_strategy("Buy XAUUSD.")

    def test_factory_selects_ollama_and_local_alias(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}):
            self.assertIsInstance(get_llm_provider(), OllamaProvider)
        with patch.dict(os.environ, {"LLM_PROVIDER": "local"}):
            self.assertIsInstance(get_llm_provider(), OllamaProvider)


if __name__ == "__main__":
    unittest.main()
