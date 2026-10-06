from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .fvg import detect_fvg
from .liquidity_sweeps import detect_liquidity_sweep_fvg_setup
from .order_flow import compute_absorption_features, compute_cvd
from .regime import compute_market_regime
from .volume_profile import compute_volume_profile, detect_volume_profile_reversion


class MicrostructureFeatureEngine:
    def __init__(self, value_area_percentage: float = 0.7) -> None:
        self.value_area_percentage = value_area_percentage

    def build(self, df: pd.DataFrame, symbol: str = "XAUUSD", timestamp: datetime | None = None) -> dict[str, Any]:
        if df.empty:
            return {
                "symbol": symbol,
                "timestamp": (timestamp or datetime.now(timezone.utc)).isoformat(),
                "feature_quality": "INSUFFICIENT_DATA",
                "data_quality_score": 0.0,
            }

        data = df.copy().sort_values("timestamp").reset_index(drop=True)
        profile = compute_volume_profile(data, value_area_percentage=self.value_area_percentage)
        value_area = detect_volume_profile_reversion(data, profile)
        cvd = compute_cvd(data)
        absorption = compute_absorption_features(data, cvd)
        fvg = detect_fvg(data)
        regime = compute_market_regime(data)
        sweep = detect_liquidity_sweep_fvg_setup(data, session_high=data["high"].max(), session_low=data["low"].min())

        ts = timestamp or data["timestamp"].iloc[-1]
        feature_vector = {
            "symbol": symbol,
            "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            "poc": profile.get("poc"),
            "vah": profile.get("vah"),
            "val": profile.get("val"),
            "value_area_width": (profile.get("vah") - profile.get("val")) if profile.get("vah") is not None and profile.get("val") is not None else None,
            "distance_to_poc": float(data["close"].iloc[-1]) - float(profile["poc"]) if profile.get("poc") is not None else None,
            "distance_to_vah": float(data["close"].iloc[-1]) - float(profile["vah"]) if profile.get("vah") is not None else None,
            "distance_to_val": float(data["close"].iloc[-1]) - float(profile["val"]) if profile.get("val") is not None else None,
            "inside_value_area": bool(profile.get("val") is not None and profile.get("vah") is not None and (float(data["low"].iloc[-1]) >= profile["val"] and float(data["high"].iloc[-1]) <= profile["vah"])),
            "above_vah": bool(profile.get("vah") is not None and float(data["close"].iloc[-1]) > profile["vah"]),
            "below_val": bool(profile.get("val") is not None and float(data["close"].iloc[-1]) < profile["val"]),
            "volume_delta": cvd.get("volume_delta"),
            "cvd": cvd.get("cvd"),
            "delta_zscore": cvd.get("delta_zscore"),
            "absorption_score": absorption.get("absorption_score"),
            "possible_absorption": absorption.get("possible_absorption"),
            "cvd_bullish_divergence": False,
            "cvd_bearish_divergence": False,
            "divergence_strength": 0.0,
            "asian_high": float(data["high"].max()),
            "asian_low": float(data["low"].min()),
            "asian_range": float(data["high"].max()) - float(data["low"].min()),
            "distance_to_asian_high": float(data["close"].iloc[-1]) - float(data["high"].max()),
            "distance_to_asian_low": float(data["close"].iloc[-1]) - float(data["low"].min()),
            "sweep_above_high": sweep.get("direction") == "SHORT",
            "sweep_below_low": sweep.get("direction") == "LONG",
            "sweep_distance": abs(float(data["close"].iloc[-1]) - float(data["high"].max())) if sweep.get("direction") == "SHORT" else abs(float(data["close"].iloc[-1]) - float(data["low"].min())),
            "return_inside_range": True,
            "bullish_fvg": fvg.get("bullish_fvg"),
            "bearish_fvg": fvg.get("bearish_fvg"),
            "fvg_size": fvg.get("fvg_size"),
            "fvg_fill_percentage": fvg.get("fvg_fill_percentage"),
            "regime": regime.get("regime"),
            "realized_volatility": regime.get("realized_volatility"),
            "atr": regime.get("atr"),
            "atr_percentile": regime.get("atr_percentile"),
            "trend_strength": regime.get("trend_strength"),
            "volume_quality": profile.get("volume_quality", "INSUFFICIENT_DATA"),
            "order_flow_quality": cvd.get("cvd_method", "APPROXIMATE"),
            "data_age_seconds": 0.0,
            "data_quality_score": 1.0 if profile.get("poc") is not None else 0.0,
            "feature_quality": "SUFFICIENT" if profile.get("poc") is not None else "INSUFFICIENT_DATA",
        }
        return feature_vector
