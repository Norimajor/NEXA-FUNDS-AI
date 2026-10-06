from __future__ import annotations

import math
from typing import Any

import pandas as pd


def compute_cvd(df: pd.DataFrame, window: int = 20) -> dict[str, Any]:
    if df.empty:
        return {
            "volume_delta": None,
            "cvd": None,
            "delta_zscore": None,
            "cvd_method": "APPROXIMATE",
            "cvd_series": pd.Series(dtype=float),
        }

    work = df.copy().reset_index(drop=True)
    if {"buy_volume", "sell_volume"}.issubset(work.columns):
        work["volume_delta"] = work["buy_volume"] - work["sell_volume"]
        cvd_series = work["volume_delta"].cumsum()
        method = "DIRECT"
    else:
        clv = ((work["close"] - work["open"]) / (work["high"] - work["low"])) if (work["high"] - work["low"]).abs().gt(0).any() else 0.0
        work["volume_delta"] = (clv * work["volume"]).fillna(0.0)
        cvd_series = work["volume_delta"].cumsum()
        method = "APPROXIMATE"

    rolling_mean = cvd_series.rolling(window=window, min_periods=1).mean()
    rolling_std = cvd_series.rolling(window=window, min_periods=2).std().fillna(0.0)
    delta_zscore = ((cvd_series - rolling_mean) / rolling_std.replace(0, math.nan)).fillna(0.0)

    return {
        "volume_delta": float(work["volume_delta"].iloc[-1]) if not work.empty else None,
        "cvd": float(cvd_series.iloc[-1]) if not cvd_series.empty else None,
        "delta_zscore": float(delta_zscore.iloc[-1]) if not delta_zscore.empty else None,
        "cvd_method": method,
        "cvd_series": cvd_series,
    }


def compute_absorption_features(df: pd.DataFrame, cvd_result: dict[str, Any] | None = None) -> dict[str, Any]:
    if df.empty:
        return {"delta_zscore": None, "price_change": None, "volume_spike": None, "delta_price_efficiency": None, "absorption_score": None, "possible_absorption": False, "absorption_signal": "NONE"}

    latest = df.iloc[-1]
    prior = df.iloc[-2] if len(df) > 1 else latest
    cvd_result = cvd_result or compute_cvd(df)
    delta_z = float(cvd_result.get("delta_zscore") or 0.0)
    price_change = float(latest["close"] - prior["close"])
    volume_spike = float(latest["volume"] / max(previous_vol := float(prior["volume"]), 1e-9)) if not df.empty else 0.0
    delta_price_eff = price_change / max(abs(float(latest["volume"])), 1e-9)
    absorption_score = abs(delta_z) * (1.0 / (1.0 + abs(price_change)))
    possible_absorption = abs(delta_z) > 1.5 and volume_spike > 1.2 and abs(price_change) < 0.8

    return {
        "delta_zscore": delta_z,
        "price_change": price_change,
        "volume_spike": volume_spike,
        "delta_price_efficiency": delta_price_eff,
        "absorption_score": absorption_score,
        "possible_absorption": bool(possible_absorption),
        "absorption_signal": "POSSIBLE_ABSORPTION" if possible_absorption else "NONE",
    }


def detect_cvd_divergence(df: pd.DataFrame, cvd_series: pd.Series | None = None) -> dict[str, Any]:
    if df.empty:
        return {"cvd_bullish_divergence": False, "cvd_bearish_divergence": False, "divergence_strength": 0.0, "reason": "INSUFFICIENT_DATA"}

    cvd_series = cvd_series if cvd_series is not None else compute_cvd(df)["cvd_series"]
    if cvd_series.empty:
        return {"cvd_bullish_divergence": False, "cvd_bearish_divergence": False, "divergence_strength": 0.0, "reason": "INSUFFICIENT_DATA"}

    close = df["close"].astype(float)
    latest_close = float(close.iloc[-1])
    recent_close = float(close.iloc[-5]) if len(close) >= 5 else float(close.iloc[0])
    cvd_last = float(cvd_series.iloc[-1]) if len(cvd_series) > 0 else 0.0
    cvd_prev = float(cvd_series.iloc[-5]) if len(cvd_series) >= 5 else float(cvd_series.iloc[0])

    bullish_div = (latest_close > recent_close) and (cvd_last < cvd_prev)
    bearish_div = (latest_close < recent_close) and (cvd_last > cvd_prev)
    strength = abs(cvd_last - cvd_prev) / max(abs(cvd_prev) + 1e-9, 1.0)

    return {
        "cvd_bullish_divergence": bool(bullish_div),
        "cvd_bearish_divergence": bool(bearish_div),
        "divergence_strength": float(strength),
        "reason": "BULLISH_DIVERGENCE" if bullish_div else "BEARISH_DIVERGENCE" if bearish_div else "NO_DIVERGENCE",
    }
