from __future__ import annotations

from typing import Any

import pandas as pd


def detect_fvg(df: pd.DataFrame) -> dict[str, Any]:
    if len(df) < 3:
        return {"bullish_fvg": False, "bearish_fvg": False, "fvg_size": None, "fvg_fill_percentage": None, "latest_fvg": None}

    bullish = False
    bearish = False
    latest = None
    for i in range(2, len(df)):
        prev = df.iloc[i - 2]
        curr = df.iloc[i]
        if prev["high"] < curr["low"]:
            bullish = True
            latest = {"type": "bullish", "high": float(prev["high"]), "low": float(curr["low"]), "size": float(curr["low"] - prev["high"]) }
            break
        if prev["low"] > curr["high"]:
            bearish = True
            latest = {"type": "bearish", "high": float(curr["high"]), "low": float(prev["low"]), "size": float(prev["low"] - curr["high"]) }
            break

    return {
        "bullish_fvg": bullish,
        "bearish_fvg": bearish,
        "fvg_size": latest["size"] if latest else None,
        "fvg_fill_percentage": 0.0,
        "latest_fvg": latest,
    }
