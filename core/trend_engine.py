from __future__ import annotations

import numpy as np
import pandas as pd


def training_trend_bias(closes, atr: float) -> tuple[float, float]:
    """Return the causal 20-bar ATR-normalized bias used by training labels."""
    values = np.asarray(closes, dtype=float)[-20:]
    if len(values) < 2 or not np.isfinite(atr) or atr <= 0:
        return 0.0, 0.0
    move_atr = float((values[-1] - values[0]) / atr)
    bias = 1.0 if move_atr >= 0.5 else -1.0 if move_atr <= -0.5 else 0.0
    return bias, move_atr


class TrendEngine:
    """Determine market bias from current price structure, independent of setup prediction."""

    def analyze(self, df: pd.DataFrame, structure: dict | None = None) -> dict:
        if len(df) < 20:
            return {"direction": "NONE", "strength": 0.0, "reason": "INSUFFICIENT_HISTORY"}

        closes = df["close"].astype(float)
        current = float(closes.iloc[-1])
        atr = float(df.iloc[-1].get("atr_14", 0.0) or 0.0)
        if not np.isfinite(atr) or atr <= 0:
            return {"direction": "NONE", "strength": 0.0, "reason": "ATR_UNAVAILABLE"}

        scores = {"BUY": 0, "SELL": 0}
        evidence = []
        ema50 = float(df.iloc[-1].get("ema_50", np.nan))
        ema200 = float(df.iloc[-1].get("ema_200", np.nan))
        if not np.isfinite(ema50):
            ema50 = float(closes.ewm(span=50, adjust=False).mean().iloc[-1])
        if not np.isfinite(ema200):
            ema200 = float(closes.ewm(span=200, adjust=False).mean().iloc[-1])
        if current < ema50 < ema200:
            scores["SELL"] += 2
            evidence.append("PRICE_BELOW_BEARISH_EMA_STACK")
        elif current > ema50 > ema200:
            scores["BUY"] += 2
            evidence.append("PRICE_ABOVE_BULLISH_EMA_STACK")
        elif current < ema50:
            scores["SELL"] += 1
            evidence.append("PRICE_BELOW_EMA50")
        elif current > ema50:
            scores["BUY"] += 1
            evidence.append("PRICE_ABOVE_EMA50")

        for period in (20, 50):
            if len(closes) <= period:
                continue
            move_atr = (current - float(closes.iloc[-period - 1])) / atr
            if move_atr <= -0.5:
                scores["SELL"] += 1
                evidence.append(f"NEGATIVE_{period}_BAR_MOMENTUM")
            elif move_atr >= 0.5:
                scores["BUY"] += 1
                evidence.append(f"POSITIVE_{period}_BAR_MOMENTUM")

        training_bias, _ = training_trend_bias(closes, atr)
        if training_bias < 0:
            scores["SELL"] += 2
            evidence.append("TRAINING_DEFINED_BEARISH_20_BAR_BIAS")
        elif training_bias > 0:
            scores["BUY"] += 2
            evidence.append("TRAINING_DEFINED_BULLISH_20_BAR_BIAS")

        if structure:
            structure_direction = structure.get("structure")
            if structure_direction == "BEARISH":
                scores["SELL"] += 2
                evidence.append("LOWER_SWING_HIGHS_AND_LOWS")
            elif structure_direction == "BULLISH":
                scores["BUY"] += 2
                evidence.append("HIGHER_SWING_HIGHS_AND_LOWS")

        sample = closes.tail(min(50, len(closes))).to_numpy()
        slope_move_atr = float(np.polyfit(np.arange(len(sample)), sample, 1)[0]) * (len(sample) - 1) / atr
        if slope_move_atr <= -1.5:
            scores["SELL"] += 1
            evidence.append("BEARISH_50_BAR_REGRESSION")
        elif slope_move_atr >= 1.5:
            scores["BUY"] += 1
            evidence.append("BULLISH_50_BAR_REGRESSION")

        lead = max(scores, key=scores.get)
        opposing = "SELL" if lead == "BUY" else "BUY"
        direction = lead if scores[lead] >= 3 and scores[lead] - scores[opposing] >= 2 else "NONE"
        strength = round(scores[lead] / 7.0, 3) if direction != "NONE" else 0.0
        return {
            "direction": direction,
            "strength": min(1.0, strength),
            "reason": "|".join(evidence) if evidence else "MIXED_OR_RANGE_BOUND",
            "bull_score": scores["BUY"],
            "bear_score": scores["SELL"],
            "regression_move_atr": round(slope_move_atr, 3),
        }
