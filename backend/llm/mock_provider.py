import re

from backend.engine.strategy_interpreter.models import StrategyDefinition
from backend.engine.strategy_interpreter.parser import StrategyParser
from .provider import LLMProvider


class MockLLMProvider(LLMProvider):
    def __init__(self):
        self.parser = StrategyParser()

    def interpret_strategy(self, user_text: str) -> StrategyDefinition:
        strategy = self.parser.parse(user_text)
        strategy.stop_loss = self._parse_exit(user_text, r"stop\s+loss\s+(\d+(?:\.\d+)?)\s*%", "percentage", "value")
        if strategy.stop_loss is None:
            strategy.stop_loss = self._parse_exit(user_text, r"(\d+(?:\.\d+)?)\s*ATR\s+stop", "atr", "multiplier")
        strategy.take_profit = self._parse_exit(user_text, r"take\s+profit\s+(\d+(?:\.\d+)?)\s*%", "percentage", "value")
        if strategy.take_profit is None:
            strategy.take_profit = self._parse_exit(user_text, r"target\s+(\d+(?:\.\d+)?)\s*R", "risk_multiple", "multiple")
        strategy.exit_policy = "explicit" if strategy.stop_loss is not None or strategy.take_profit is not None else "unspecified"
        strategy.risk_reward = self._risk_reward(strategy.stop_loss, strategy.take_profit)
        return strategy

    def _parse_exit(self, text: str, pattern: str, exit_type: str, value_key: str):
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            return None
        return {"type": exit_type, value_key: float(match.group(1)), "specified": True}

    def _risk_reward(self, stop_loss, take_profit):
        if stop_loss and take_profit:
            stop_value = stop_loss.get("value", stop_loss.get("multiplier"))
            target_value = take_profit.get("value", take_profit.get("multiple"))
            if stop_loss["type"] == "percentage" and take_profit["type"] == "percentage":
                return float(target_value) / float(stop_value)
            if take_profit["type"] == "risk_multiple":
                return float(target_value)
        return None
