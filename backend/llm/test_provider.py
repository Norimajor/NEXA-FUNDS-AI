import os
import unittest

from backend.engine.strategy_interpreter.models import StrategyCondition, StrategyDefinition
from backend.engine.strategy_interpreter.validator import StrategyValidator
from backend.llm.mock_provider import MockLLMProvider
from backend.llm.provider import LLMProvider, get_llm_provider


class TestLLMProvider(unittest.TestCase):
    def setUp(self):
        self.provider = MockLLMProvider()

    def test_percentage_exits_are_preserved(self):
        strategy = self.provider.interpret_strategy(
            "Buy XAUUSD when RSI drops below 30 on M15. Take profit 1.5% and stop loss 0.7%."
        )
        self.assertEqual(strategy.symbol, "XAUUSD")
        self.assertEqual(strategy.timeframe, "M15")
        self.assertEqual(strategy.direction, "long")
        self.assertEqual(strategy.entry_conditions[0].indicator, "RSI")
        self.assertEqual(strategy.entry_conditions[0].period, 14)
        self.assertEqual(strategy.entry_conditions[0].operator, "<")
        self.assertEqual(strategy.entry_conditions[0].value, 30.0)
        self.assertEqual(strategy.take_profit, {"type": "percentage", "value": 1.5, "specified": True})
        self.assertEqual(strategy.stop_loss, {"type": "percentage", "value": 0.7, "specified": True})

    def test_atr_and_r_multiple_exits_are_not_normalized(self):
        strategy = self.provider.interpret_strategy("Buy XAUUSD using RSI below 30. Use a 2 ATR stop and target 3R.")
        self.assertEqual(strategy.stop_loss, {"type": "atr", "multiplier": 2.0, "specified": True})
        self.assertEqual(strategy.take_profit, {"type": "risk_multiple", "multiple": 3.0, "specified": True})

    def test_missing_exits_remain_missing(self):
        strategy = self.provider.interpret_strategy("Buy XAUUSD using RSI below 30 on M15.")
        self.assertIsNone(strategy.stop_loss)
        self.assertIsNone(strategy.take_profit)
        self.assertIsNone(strategy.risk_reward)
        self.assertEqual(strategy.exit_policy, "unspecified")

    def test_provider_configuration_defaults_to_mock(self):
        previous = os.environ.pop("LLM_PROVIDER", None)
        try:
            self.assertIsInstance(get_llm_provider(), MockLLMProvider)
        finally:
            if previous is not None:
                os.environ["LLM_PROVIDER"] = previous

    def test_mock_interpretation_passes_existing_validator(self):
        strategy = self.provider.interpret_strategy(
            "Buy XAUUSD when RSI drops below 30 on M15. Take profit 1.5% and stop loss 0.7%."
        )
        self.assertIsInstance(self.provider, LLMProvider)
        self.assertTrue(StrategyValidator().validate(strategy)["valid"])

    def test_invalid_canonical_strategy_is_rejected_deterministically(self):
        strategy = StrategyDefinition(
            name="Invalid provider result",
            symbol="XAUUSD",
            timeframe="M15",
            direction="long",
            entry_conditions=[StrategyCondition(indicator="RSI", operator="not-an-operator", value=30, period=14)],
            risk_percent=-1,
            risk_reward=0,
            stop_loss={"type": "unknown", "value": 1},
            take_profit=None,
        )
        strategy.timeframe = "D1"
        result = StrategyValidator().validate(strategy)
        self.assertFalse(result["valid"])
        self.assertTrue(any("timeframe" in error.lower() for error in result["errors"]))
        self.assertTrue(any("operator" in error.lower() for error in result["errors"]))
        self.assertTrue(any("risk" in error.lower() for error in result["errors"]))
        self.assertTrue(any("stop loss type" in error.lower() for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
