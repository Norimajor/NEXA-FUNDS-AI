from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class OptionsProvider(Protocol):
    def fetch_contracts(self, *args, **kwargs) -> list["OptionContract"]:
        ...


@dataclass
class OptionContract:
    asset: str
    provider: str
    timestamp: datetime | None = None
    underlying_price: float | None = None
    expiration: str | None = None
    days_to_expiration: int | None = None
    strike: float | None = None
    option_type: str | None = None
    contract_symbol: str | None = None
    bid: float | None = None
    ask: float | None = None
    last_price: float | None = None
    mark_price: float | None = None
    volume: float | None = None
    open_interest: float | None = None
    implied_volatility: float | None = None
    bid_iv: float | None = None
    ask_iv: float | None = None
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    moneyness: float | None = None
    iv: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.timestamp is not None:
            payload["timestamp"] = self.timestamp.isoformat()
        return payload
