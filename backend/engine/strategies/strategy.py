from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"
    BOTH = "both"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class RiskMode(str, Enum):
    FIXED_LOT = "fixed_lot"
    PERCENT = "percent"
    FIXED_AMOUNT = "fixed_amount"


class Condition(BaseModel):
    type: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class ExitRule(BaseModel):
    type: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class RiskManagement(BaseModel):
    mode: RiskMode = RiskMode.PERCENT

    risk_value: float = 1.0

    stop_loss: Optional[dict[str, Any]] = None
    take_profit: Optional[dict[str, Any]] = None

    break_even: Optional[dict[str, Any]] = None
    trailing_stop: Optional[dict[str, Any]] = None


class StrategyDefinition(BaseModel):

    name: str

    description: Optional[str] = None

    symbol: str
    timeframe: str

    direction: Direction = Direction.BOTH

    entry_conditions: list[Condition] = Field(
        default_factory=list
    )

    exit_conditions: list[ExitRule] = Field(
        default_factory=list
    )

    risk_management: RiskManagement = Field(
        default_factory=RiskManagement
    )

    filters: list[Condition] = Field(
        default_factory=list
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class StrategyEngine:

    @staticmethod
    def validate(strategy: StrategyDefinition) -> dict:

        errors = []

        if not strategy.name.strip():
            errors.append("Strategy name cannot be empty.")

        if not strategy.symbol.strip():
            errors.append("Symbol cannot be empty.")

        if not strategy.timeframe.strip():
            errors.append("Timeframe cannot be empty.")

        if len(strategy.entry_conditions) == 0:
            errors.append(
                "Strategy must contain at least one entry condition."
            )

        if strategy.risk_management.risk_value <= 0:
            errors.append(
                "Risk value must be greater than zero."
            )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
        }

    @staticmethod
    def summarize(strategy: StrategyDefinition) -> dict:

        return {
            "name": strategy.name,
            "symbol": strategy.symbol.upper(),
            "timeframe": strategy.timeframe.upper(),
            "direction": strategy.direction.value,
            "entry_conditions": len(
                strategy.entry_conditions
            ),
            "exit_conditions": len(
                strategy.exit_conditions
            ),
            "filters": len(strategy.filters),
            "risk_mode": (
                strategy.risk_management.mode.value
            ),
            "risk_value": (
                strategy.risk_management.risk_value
            ),
        }
