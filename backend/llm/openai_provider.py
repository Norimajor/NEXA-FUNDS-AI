import json
import os
from typing import Any

from backend.engine.strategy_interpreter.models import StrategyCondition, StrategyDefinition

from .provider import LLMProvider, LLMProviderError

DEFAULT_MODEL = "gpt-5.6-luna"

SYSTEM_INSTRUCTIONS = """You are a trading-strategy interpretation engine.

Your job is ONLY to convert the user's natural-language trading strategy into the provided canonical strategy schema.

Never invent missing information, entry conditions, stop-loss values, or take-profit values. Never convert units unless the user explicitly specifies the unit. Preserve percentage, ATR, pips, points, price, and R-multiple units exactly. If exits are missing, leave them missing. Never default missing exits to ATR or to a 2:1 risk/reward. Never generate backtest results, historical market data, performance statistics, optimization results, or walk-forward results. The deterministic backend will validate the resulting strategy.
"""


EXIT_TYPES = ("percentage", "atr", "risk_multiple", "pips", "points", "price")


def _condition_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "indicator": {"type": "string"},
            "operator": {"type": "string"},
            "value": {"type": ["number", "string", "null"]},
            "period": {"type": ["integer", "null"]},
            "side": {"type": ["string", "null"]},
            "timeframe": {"type": ["string", "null"]},
            "entry_semantics": {"type": "string"},
        },
        "required": ["indicator", "operator", "value", "period", "side", "timeframe", "entry_semantics"],
    }


def _exit_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "type": {"type": "string", "enum": list(EXIT_TYPES)},
            "value": {"type": ["number", "null"]},
            "multiplier": {"type": ["number", "null"]},
            "multiple": {"type": ["number", "null"]},
            "specified": {"type": "boolean"},
        },
        "required": ["type", "value", "multiplier", "multiple", "specified"],
    }


CANONICAL_STRATEGY_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string"},
        "symbol": {"type": "string"},
        "timeframe": {"type": "string"},
        "direction": {"type": "string"},
        "entry_conditions": {"type": "array", "items": _condition_schema()},
        "exit_conditions": {"type": "array", "items": _condition_schema()},
        "filters": {"type": "array", "items": _condition_schema()},
        "risk_percent": {"type": ["number", "null"]},
        "risk_reward": {"type": ["number", "null"]},
        "allow_reentry": {"type": "boolean"},
        "max_simultaneous_positions": {"type": "integer"},
        "pyramiding": {"type": "boolean"},
        "cooldown_bars": {"type": "integer"},
        "exit_policy": {"type": "string"},
        "stop_loss": {"anyOf": [_exit_schema(), {"type": "null"}]},
        "take_profit": {"anyOf": [_exit_schema(), {"type": "null"}]},
        "position_sizing": {"type": "string"},
        "instrument_spec": {"type": "object", "additionalProperties": False, "properties": {}},
    },
    "required": [
        "name", "symbol", "timeframe", "direction", "entry_conditions", "exit_conditions", "filters",
        "risk_percent", "risk_reward", "allow_reentry", "max_simultaneous_positions", "pyramiding",
        "cooldown_bars", "exit_policy", "stop_loss", "take_profit", "position_sizing", "instrument_spec",
    ],
}


class OpenAIProvider(LLMProvider):
    def __init__(self, client=None, api_key: str | None = None, model: str | None = None):
        self.model = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
        if client is not None:
            self.client = client
            return
        configured_key = api_key or os.getenv("OPENAI_API_KEY")
        if not configured_key:
            raise LLMProviderError("OpenAI strategy interpretation requires OPENAI_API_KEY.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMProviderError("The OpenAI SDK is not installed. Install the declared openai dependency.") from exc
        self.client = OpenAI(api_key=configured_key)

    def interpret_strategy(self, user_text: str) -> StrategyDefinition:
        if not isinstance(user_text, str) or not user_text.strip():
            raise LLMProviderError("Strategy interpretation failed: strategy text is required.")
        try:
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                    {"role": "user", "content": user_text.strip()},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "canonical_strategy",
                        "strict": True,
                        "schema": CANONICAL_STRATEGY_JSON_SCHEMA,
                    }
                },
            )
            payload = json.loads(response.output_text)
            return self._build_strategy(payload)
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(f"Strategy interpretation failed: {exc}") from exc

    def _build_strategy(self, payload: Any) -> StrategyDefinition:
        if not isinstance(payload, dict):
            raise LLMProviderError("Strategy interpretation failed: structured response was not an object.")
        try:
            return StrategyDefinition(
                name=payload["name"],
                symbol=payload["symbol"],
                timeframe=payload["timeframe"],
                direction=payload["direction"],
                entry_conditions=[self._build_condition(item) for item in payload["entry_conditions"]],
                exit_conditions=[self._build_condition(item) for item in payload["exit_conditions"]],
                filters=[self._build_condition(item) for item in payload["filters"]],
                risk_percent=payload["risk_percent"],
                risk_reward=payload["risk_reward"],
                allow_reentry=payload["allow_reentry"],
                max_simultaneous_positions=payload["max_simultaneous_positions"],
                pyramiding=payload["pyramiding"],
                cooldown_bars=payload["cooldown_bars"],
                exit_policy=payload["exit_policy"],
                stop_loss=self._build_exit(payload["stop_loss"]),
                take_profit=self._build_exit(payload["take_profit"]),
                position_sizing=payload["position_sizing"],
                instrument_spec=payload["instrument_spec"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise LLMProviderError(f"Strategy interpretation failed: malformed canonical strategy: {exc}") from exc

    def _build_condition(self, payload: Any) -> StrategyCondition:
        if not isinstance(payload, dict):
            raise TypeError("condition must be an object")
        return StrategyCondition(
            indicator=payload["indicator"],
            operator=payload["operator"],
            value=payload["value"],
            period=payload["period"],
            side=payload["side"],
            timeframe=payload["timeframe"],
            entry_semantics=payload["entry_semantics"],
        )

    def _build_exit(self, payload: Any) -> dict[str, Any] | None:
        if payload is None:
            return None
        if not isinstance(payload, dict) or payload.get("type") not in EXIT_TYPES:
            raise ValueError("invalid exit type")
        exit_type = payload["type"]
        value_key = {"percentage": "value", "atr": "multiplier", "risk_multiple": "multiple"}.get(exit_type)
        if value_key is None:
            value_key = "value"
        value = payload.get(value_key)
        return {"type": exit_type, value_key: value, "specified": payload["specified"]}
