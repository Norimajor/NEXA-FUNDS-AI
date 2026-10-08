from __future__ import annotations

import numpy as np


def training_market_trend(
    closes,
    atr: float,
    lookback: int = 50,
    threshold_atr: float = 1.0,
) -> tuple[str, float]:
    """Label the trailing price path without using moving-average rules."""
    values = np.asarray(closes, dtype=float)[-lookback:]
    if (
        len(values) < lookback
        or not np.isfinite(values).all()
        or not np.isfinite(atr)
        or atr <= 0
    ):
        return "RANGE", 0.0
    centered_x = np.arange(lookback, dtype=float)
    centered_x -= centered_x.mean()
    slope = np.dot(centered_x, values - values.mean()) / np.dot(centered_x, centered_x)
    slope_move_atr = float(slope * (lookback - 1) / atr)
    if slope_move_atr >= threshold_atr:
        return "BUY", slope_move_atr
    if slope_move_atr <= -threshold_atr:
        return "SELL", slope_move_atr
    return "RANGE", slope_move_atr


def training_trend_bias(closes, atr: float) -> tuple[float, float]:
    """Return the causal 20-bar ATR-normalized bias used by training labels."""
    values = np.asarray(closes, dtype=float)[-20:]
    if len(values) < 2 or not np.isfinite(atr) or atr <= 0:
        return 0.0, 0.0
    move_atr = float((values[-1] - values[0]) / atr)
    bias = 1.0 if move_atr >= 0.5 else -1.0 if move_atr <= -0.5 else 0.0
    return bias, move_atr


class TrendEngine:
    """Format a trained trend-class prediction for the analysis API."""

    def analyze(self, prediction: dict | None) -> dict:
        if not prediction:
            return {
                "direction": "RANGE",
                "strength": 0.0,
                "reason": "TREND_MODEL_UNAVAILABLE",
                "model_version": "unavailable",
            }
        direction = prediction.get("direction", "RANGE")
        if direction not in {"BUY", "SELL", "RANGE"}:
            direction = "RANGE"
        return {
            "direction": direction,
            "strength": round(float(prediction.get("confidence", 0.0) or 0.0), 3),
            "reason": prediction.get("reason", "ML_TREND_CLASSIFICATION"),
            "model_version": prediction.get("model_version", "unknown"),
        }
