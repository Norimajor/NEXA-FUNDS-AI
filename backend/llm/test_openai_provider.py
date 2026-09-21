import json
import os
import unittest

from backend.engine.strategy_interpreter.validator import StrategyValidator
from backend.llm.openai_provider import OpenAIProvider
from backend.llm.provider import LLMProviderError, get_llm_provider


class FakeResponse:
    def __init__(self, payload):
        self.output_text = json.dumps(payload)


class FakeResponses:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return FakeResponse(self.payload)


class FakeClient:
    def __init__(self, payload=None, error=None):
        self.responses = FakeResponses(payload, error)


def condition():
    return {
        "indicator": "RSI",
        "operator": "<",
        "value": 30,
        "period": 14,
        "side": None,
        "timeframe": "M15",
        "entry_semantics": "condition",
    }


def strategy_payload(stop_loss, take_profit, risk_reward=2.142857142857143):
    return {
        "name": "OpenAI interpreted strategy",
        "symbol": "XAUUSD",
        "timeframe": "M15",
        "direction": "long",
        "entry_conditions": [condition()],
        "exit_conditions": [],
        "filters": [],
        "risk_percent": 1.0,
        "risk_reward": risk_reward,
        "allow_reentry": True,
        "max_simultaneous_positions": 1,
        "pyramiding": False,
        "cooldown_bars": 0,
        "exit_policy": "explicit" if stop_loss or take_profit else "unspecified",
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "position_sizing": "percent_risk",
        "instrument_spec": {"contract_size": 1.0},
    }


class TestOpenAIProvider(unittest.TestCase):
    def test_percentage_exits_are_mapped_without_unit_conversion(self):
        client = FakeClient(
            strategy_payload(
                {"type": "percentage", "value": 0.7, "multiplier": None, "multiple": None, "specified": True},
                {"type": "percentage", "value": 1.5, "multiplier": None, "multiple": None, "specified": True},
            )
        )
        strategy = OpenAIProvider(client=client, model="test-model").interpret_strategy(
            "Buy XAUUSD when RSI drops below 30 on M15. Take profit 1.5% and stop loss 0.7%."
        )
        self.assertEqual(strategy.symbol, "XAUUSD")
        self.assertEqual(strategy.timeframe, "M15")
        self.assertEqual(strategy.direction, "long")
        self.assertEqual(strategy.stop_loss, {"type": "percentage", "value": 0.7, "specified": True})
        self.assertEqual(strategy.take_profit, {"type": "percentage", "value": 1.5, "specified": True})

    def test_atr_and_r_multiple_units_are_preserved(self):
        client = FakeClient(
            strategy_payload(
                {"type": "atr", "value": None, "multiplier": 2, "multiple": None, "specified": True},
                {"type": "risk_multiple", "value": None, "multiplier": None, "multiple": 3, "specified": True},
                risk_reward=3,
            )
        )
        strategy = OpenAIProvider(client=client).interpret_strategy("Buy XAUUSD using RSI below 30. Use a 2 ATR stop and target 3R.")
        self.assertEqual(strategy.stop_loss, {"type": "atr", "multiplier": 2, "specified": True})
        self.assertEqual(strategy.take_profit, {"type": "risk_multiple", "multiple": 3, "specified": True})

    def test_missing_exits_remain_missing(self):
        client = FakeClient(strategy_payload(None, None, risk_reward=None))
        strategy = OpenAIProvider(client=client).interpret_strategy("Buy XAUUSD when RSI is below 30.")
        self.assertIsNone(strategy.stop_loss)
        self.assertIsNone(strategy.take_profit)
        self.assertIsNone(strategy.risk_reward)

    def test_responses_request_uses_strict_structured_output_and_configured_model(self):
        client = FakeClient(strategy_payload(None, None, risk_reward=None))
        OpenAIProvider(client=client, model="configured-model").interpret_strategy("Buy XAUUSD when RSI is below 30.")
        request = client.responses.calls[0]
        self.assertEqual(request["model"], "configured-model")
        self.assertEqual(request["text"]["format"]["type"], "json_schema")
        self.assertTrue(request["text"]["format"]["strict"])
        self.assertIn("Never invent missing information", request["input"][0]["content"])

    def test_missing_key_is_clear_and_does_not_import_sdk(self):
        previous = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with self.assertRaisesRegex(LLMProviderError, "OPENAI_API_KEY"):
                OpenAIProvider()
        finally:
            if previous is not None:
                os.environ["OPENAI_API_KEY"] = previous

    def test_api_failure_is_wrapped_without_fallback(self):
        client = FakeClient(error=TimeoutError("network timeout"))
        with self.assertRaisesRegex(LLMProviderError, "Strategy interpretation failed"):
            OpenAIProvider(client=client).interpret_strategy("Buy XAUUSD when RSI is below 30.")

    def test_invalid_structured_output_is_rejected_by_deterministic_validator(self):
        payload = strategy_payload(
            {"type": "percentage", "value": -1, "multiplier": None, "multiple": None, "specified": True},
            None,
            risk_reward=-2,
        )
        payload["entry_conditions"][0]["operator"] = "invalid"
        strategy = OpenAIProvider(client=FakeClient(payload)).interpret_strategy("invalid")
        strategy.timeframe = "D1"
        result = StrategyValidator().validate(strategy)
        self.assertFalse(result["valid"])
        self.assertGreaterEqual(len(result["errors"]), 4)

    def test_factory_selects_openai_without_network_call(self):
        previous_provider = os.environ.get("LLM_PROVIDER")
        previous_key = os.environ.pop("OPENAI_API_KEY", None)
        os.environ["LLM_PROVIDER"] = "openai"
        try:
            with self.assertRaisesRegex(LLMProviderError, "OPENAI_API_KEY"):
                get_llm_provider()
        finally:
            if previous_provider is None:
                os.environ.pop("LLM_PROVIDER", None)
            else:
                os.environ["LLM_PROVIDER"] = previous_provider
            if previous_key is not None:
                os.environ["OPENAI_API_KEY"] = previous_key


if __name__ == "__main__":
    unittest.main()
