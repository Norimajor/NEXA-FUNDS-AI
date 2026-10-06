from __future__ import annotations

from typing import Any

import pandas as pd


def detect_liquidity_sweep_fvg_setup(df: pd.DataFrame, session_high: float | None = None, session_low: float | None = None) -> dict[str, Any]:
    if df.empty:
        return {"strategy": "LIQUIDITY_SWEEP_FVG", "direction": "NONE", "setup_strength": 0.0, "reason": "INSUFFICIENT_DATA"}

    latest = df.iloc[-1]
    high = session_high if session_high is not None else float(df["high"].max())
    low = session_low if session_low is not None else float(df["low"].min())
    if latest["high"] > high:
        return {"strategy": "LIQUIDITY_SWEEP_FVG", "direction": "SHORT", "entry_reference": latest["close"], "stop_reference": high, "target_reference": low, "setup_strength": 0.5, "reason": "SWEEP_ABOVE_HIGH"}
    if latest["low"] < low:
        return {"strategy": "LIQUIDITY_SWEEP_FVG", "direction": "LONG", "entry_reference": latest["close"], "stop_reference": low, "target_reference": high, "setup_strength": 0.5, "reason": "SWEEP_BELOW_LOW"}
    return {"strategy": "LIQUIDITY_SWEEP_FVG", "direction": "NONE", "entry_reference": latest["close"], "stop_reference": None, "target_reference": None, "setup_strength": 0.0, "reason": "NO_SWEEP"}
