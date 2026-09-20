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
    entry_semantics: str = "condition"

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
        self.entry_semantics = self.entry_semantics.lower()
        if self.entry_semantics not in {"condition", "cross", "breakout", "touch", "candle_close"}:
            raise ValueError(f"Unsupported entry semantics: {self.entry_semantics}")


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
    allow_reentry: bool = True
    max_simultaneous_positions: int = 1
    pyramiding: bool = False
    cooldown_bars: int = 0
    exit_policy: str = "atr_stop_and_r_multiple_target"
    stop_loss: dict[str, Any] = field(default_factory=lambda: {"type": "atr", "multiplier": 0.6, "specified": False})
    take_profit: dict[str, Any] = field(default_factory=lambda: {"type": "risk_multiple", "multiple": 2.0, "specified": False})
    position_sizing: str = "percent_risk"
    instrument_spec: dict[str, Any] = field(default_factory=lambda: {
        "contract_size": 1.0,
        "tick_size": None,
        "tick_value": None,
        "minimum_lot": None,
        "lot_step": None,
        "source": "not specified by CSV; price-unit sizing assumption",
    })

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