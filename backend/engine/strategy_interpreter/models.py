from dataclasses import dataclass, field
from typing import Any


SUPPORTED_TIMEFRAMES = {
    "H4",
    "H1",
    "M30",
    "M15",
    "M5",
    "M1",
}


@dataclass
class StrategyCondition:
    indicator: str
    operator: str
    value: Any = None
    period: int | None = None
    side: str | None = None

    # Timeframe on which this condition must be evaluated.
    # None means use the strategy's primary timeframe.
    timeframe: str | None = None

    def __post_init__(self):
        if self.timeframe is not None:
            self.timeframe = self.timeframe.upper()

            if self.timeframe not in SUPPORTED_TIMEFRAMES:
                raise ValueError(
                    f"Unsupported timeframe '{self.timeframe}'. "
                    f"Supported: {sorted(SUPPORTED_TIMEFRAMES)}"
                )

        self.indicator = self.indicator.upper()
        self.operator = self.operator.lower()


@dataclass
class StrategyDefinition:
    name: str
    symbol: str
    timeframe: str
    direction: str

    entry_conditions: list[StrategyCondition] = field(
        default_factory=list
    )

    exit_conditions: list[StrategyCondition] = field(
        default_factory=list
    )

    filters: list[StrategyCondition] = field(
        default_factory=list
    )

    risk_percent: float = 1.0
    risk_reward: float = 2.0

    def __post_init__(self):
        self.symbol = self.symbol.upper()
        self.timeframe = self.timeframe.upper()
        self.direction = self.direction.lower()

        if self.timeframe not in SUPPORTED_TIMEFRAMES:
            raise ValueError(
                f"Unsupported strategy timeframe "
                f"'{self.timeframe}'. "
                f"Supported: {sorted(SUPPORTED_TIMEFRAMES)}"
            )