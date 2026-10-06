from __future__ import annotations

from typing import Any

import pandas as pd


def compute_market_regime(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"regime": "REGIME_UNKNOWN", "realized_volatility": None, "atr": None, "atr_percentile": None, "trend_strength": None}

    close = df["close"].astype(float)
    returns = close.pct_change().fillna(0.0)
    realized_volatility = float(returns.std() * (252 ** 0.5)) if not returns.empty else None
    high_low = (df["high"] - df["low"]).astype(float)
    atr = float(high_low.rolling(14, min_periods=1).mean().iloc[-1]) if not high_low.empty else None

    trend_strength = float((close.iloc[-1] - close.iloc[0]) / max(abs(close.iloc[0]), 1e-9)) if len(close) > 1 else None
    if realized_volatility is not None and realized_volatility < 0.1:
        regime = "REGIME_RANGE"
    elif trend_strength is not None and abs(trend_strength) > 0.03:
        regime = "REGIME_EXPANSION"
    elif atr is not None and atr > 0:
        regime = "REGIME_CHOP"
    else:
        regime = "REGIME_UNKNOWN"

    return {
        "regime": regime,
        "realized_volatility": realized_volatility,
        "atr": atr,
        "atr_percentile": None,
        "trend_strength": trend_strength,
    }
