from __future__ import annotations

from typing import Any

OPTIONS_RISK_CONFIG = {
    "max_position_size_fraction": 0.02,
    "max_daily_loss_fraction": 0.05,
    "max_simultaneous_positions": 1,
    "max_trades_per_session": 2,
    "cooldown_seconds_after_loss": 1800,
    "spread_slippage_limit_pct": 0.01,
    "stale_data_seconds": 300,
    "signal_max_age_seconds": 600,
    "emergency_disable": False,
}


class OptionsRiskManager:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = {**OPTIONS_RISK_CONFIG, **(config or {})}

    def validate_signal(
        self,
        signal: dict[str, Any],
        *,
        account_equity: float | None = None,
        risk_percentage: float | None = None,
        entry_price: float | None = None,
        stop_loss_distance: float | None = None,
        spread_pct: float | None = None,
        signal_age_seconds: int | None = None,
    ) -> dict[str, Any]:
        if signal.get("signal") in {None, "HOLD"}:
            return {"allowed": False, "position_size": None, "reason": "NO_TRADE_SIGNAL", "reasons": []}
        if self.config.get("emergency_disable"):
            return {"allowed": False, "position_size": None, "reason": "EMERGENCY_DISABLE", "reasons": ["Emergency disable is active."]}
        if signal.get("data_quality", {}).get("sufficient") is False:
            return {"allowed": False, "position_size": None, "reason": "INSUFFICIENT_DATA", "reasons": ["Data quality gate failed."]}

        entry_price = entry_price if entry_price is not None else signal.get("entry_reference")
        if entry_price is None:
            return {"allowed": False, "position_size": None, "reason": "MISSING_ENTRY_PRICE", "reasons": ["Entry price missing."]}

        stop_loss_distance = stop_loss_distance if stop_loss_distance is not None else max(float(entry_price) * 0.01, 1.0)
        if stop_loss_distance <= 0:
            return {"allowed": False, "position_size": None, "reason": "INVALID_STOP_DISTANCE", "reasons": ["Stop-loss distance invalid."]}

        if spread_pct is not None and spread_pct > self.config.get("spread_slippage_limit_pct", 0.01):
            return {"allowed": False, "position_size": None, "reason": "SPREAD_REJECTED", "reasons": ["Spread exceeds configured limit."]}

        if signal_age_seconds is None:
            signal_age_seconds = 0
        if signal_age_seconds > self.config.get("signal_max_age_seconds", 600):
            return {"allowed": False, "position_size": None, "reason": "SIGNAL_EXPIRED", "reasons": ["Signal expired."]}

        account_equity = float(account_equity) if account_equity is not None else 100000.0
        risk_percentage = float(risk_percentage) if risk_percentage is not None else self.config.get("max_position_size_fraction", 0.02)
        position_size = account_equity * risk_percentage / max(float(stop_loss_distance), 1.0)
        return {
            "allowed": True,
            "position_size": position_size,
            "reason": None,
            "reasons": [],
            "risk_percentage": risk_percentage,
            "entry_price": entry_price,
            "stop_loss_distance": stop_loss_distance,
        }

    def validate_execution_context(
        self,
        signal: dict[str, Any],
        *,
        current_price: float | None = None,
        symbol: str | None = None,
        spread_pct: float | None = None,
        session_allowed: bool = True,
        executed_signal_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        if executed_signal_ids is None:
            executed_signal_ids = set()
        if signal.get("signal_id") in executed_signal_ids:
            return {"allowed": False, "reason": "DUPLICATE_SIGNAL", "reasons": ["Signal already executed."]}
        if current_price is not None and signal.get("entry_reference") is not None:
            drift = abs(float(current_price) - float(signal["entry_reference"])) / max(float(signal["entry_reference"]), 1.0)
            if drift > 0.05:
                return {"allowed": False, "reason": "PRICE_DRIFT", "reasons": ["Reference drift exceeds threshold."]}
        if spread_pct is not None and spread_pct > self.config.get("spread_slippage_limit_pct", 0.01):
            return {"allowed": False, "reason": "SPREAD_REJECTED", "reasons": ["Spread exceeds limit."]}
        if not session_allowed:
            return {"allowed": False, "reason": "SESSION_BLOCKED", "reasons": ["Trading session not allowed."]}
        return {"allowed": True, "reason": None, "reasons": []}
