from __future__ import annotations

from typing import Any


class ExecutionAdapter:
    def __init__(self, paper_trading: bool = True, live_trading: bool = False):
        self.paper_trading = paper_trading
        self.live_trading = live_trading
        self.executed_signal_ids: set[str] = set()

    def build_signal_response(self, signal: dict[str, Any], risk_result: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "asset": signal.get("asset"),
            "signal": signal.get("signal"),
            "signal_id": signal.get("signal_id"),
            "signal_timestamp": signal.get("signal_timestamp"),
            "signal_expiry": signal.get("signal_expiry"),
            "timestamp": signal.get("signal_timestamp"),
            "expires_at": signal.get("signal_expiry"),
            "entry_reference": signal.get("entry_reference"),
            "stop_loss": signal.get("stop_loss"),
            "take_profit": signal.get("take_profit"),
            "regime": signal.get("regime"),
            "score": signal.get("score"),
            "confidence": signal.get("confidence", signal.get("signal_confidence")),
            "signal_confidence": signal.get("signal_confidence", signal.get("confidence")),
            "features": signal.get("features"),
            "reasons": signal.get("reasons", []),
            "data_quality": signal.get("data_quality", {"sufficient": False, "missing_fields": []}),
            "risk": risk_result or {"allowed": False, "position_size": None, "reason": "NOT_VALIDATED"},
            "paper_trading": self.paper_trading,
            "live_trading": self.live_trading,
        }

    def validate_for_execution(self, signal: dict[str, Any], *, current_price: float | None = None, symbol: str | None = None, spread_pct: float | None = None, session_allowed: bool = True) -> dict[str, Any]:
        if self.live_trading is False:
            return {"allowed": False, "reason": "LIVE_TRADING_DISABLED", "reasons": ["Live trading is disabled by policy."]}
        signal_id = signal.get("signal_id")
        if signal_id in self.executed_signal_ids:
            return {"allowed": False, "reason": "DUPLICATE_SIGNAL", "reasons": ["Signal already executed."]}
        if signal.get("signal") in {None, "HOLD"}:
            return {"allowed": False, "reason": "NO_TRADE_SIGNAL", "reasons": ["Signal does not justify execution."]}
        if signal.get("data_quality", {}).get("sufficient") is False:
            return {"allowed": False, "reason": "INSUFFICIENT_DATA", "reasons": ["Data quality gate failed."]}
        if current_price is not None and signal.get("entry_reference") is not None:
            drift = abs(float(current_price) - float(signal["entry_reference"])) / max(float(signal["entry_reference"]), 1.0)
            if drift > 0.05:
                return {"allowed": False, "reason": "PRICE_DRIFT", "reasons": ["Reference price drift exceeds the safety threshold."]}
        if spread_pct is not None and spread_pct > 0.01:
            return {"allowed": False, "reason": "SPREAD_REJECTED", "reasons": ["Spread is too wide for execution."]}
        if not session_allowed:
            return {"allowed": False, "reason": "SESSION_BLOCKED", "reasons": ["Trading session not allowed."]}
        return {"allowed": True, "reason": None, "reasons": []}

    def mark_executed(self, signal_id: str | None) -> None:
        if signal_id:
            self.executed_signal_ids.add(signal_id)

    def paper_trade_only(self) -> dict[str, Any]:
        return {
            "paper_trading": self.paper_trading,
            "live_trading": self.live_trading,
            "mode": "PAPER_ONLY",
            "message": "No live execution is enabled by default.",
        }
