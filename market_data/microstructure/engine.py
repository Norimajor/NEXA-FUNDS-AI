from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .features import MicrostructureFeatureEngine
from .fvg import detect_fvg
from .liquidity_sweeps import detect_liquidity_sweep_fvg_setup
from .order_flow import compute_absorption_features, compute_cvd
from .regime import compute_market_regime
from .volume_profile import compute_volume_profile, detect_volume_profile_reversion


class MicrostructureEngine:
    def __init__(self, value_area_percentage: float = 0.7):
        self.value_area_percentage = value_area_percentage
        self.feature_engine = MicrostructureFeatureEngine(value_area_percentage=value_area_percentage)

    def analyze(self, df: pd.DataFrame, symbol: str = "XAUUSD") -> dict[str, Any]:
        if df.empty:
            return {
                "symbol": symbol,
                "regime": "REGIME_UNKNOWN",
                "feature_vector": {"symbol": symbol, "timestamp": datetime.now(timezone.utc).isoformat(), "feature_quality": "INSUFFICIENT_DATA"},
                "setup": {"strategy": "NONE", "direction": "NONE"},
                "value_area": {"poc": None, "vah": None, "val": None},
                "cvd": {"cvd_method": "APPROXIMATE", "cvd": None},
                "quality": {"data_quality_score": 0.0},
            }

        data = df.copy().sort_values("timestamp").reset_index(drop=True)
        profile = compute_volume_profile(data, value_area_percentage=self.value_area_percentage)
        cvd = compute_cvd(data)
        absorption = compute_absorption_features(data, cvd)
        value_setup = detect_volume_profile_reversion(data, profile)
        fvg = detect_fvg(data)
        regime = compute_market_regime(data)
        liquidity = detect_liquidity_sweep_fvg_setup(data)
        feature_vector = self.feature_engine.build(data, symbol=symbol, timestamp=data["timestamp"].iloc[-1])

        setup = value_setup
        if setup.get("direction") == "NONE":
            setup = liquidity
        if setup.get("direction") == "NONE" and (fvg.get("bullish_fvg") or fvg.get("bearish_fvg")):
            setup = {"strategy": "FVG", "direction": "LONG" if fvg.get("bullish_fvg") else "SHORT", "setup_strength": 0.4, "reason": "FVG"}

        return {
            "symbol": symbol,
            "timestamp": data["timestamp"].iloc[-1].isoformat(),
            "regime": regime.get("regime"),
            "feature_vector": {**feature_vector, "symbol": symbol},
            "setup": setup,
            "value_area": profile,
            "cvd": {"cvd_method": cvd.get("cvd_method"), "cvd": cvd.get("cvd"), "delta_zscore": cvd.get("delta_zscore")},
            "absorption": absorption,
            "fvg": fvg,
            "quality": {"data_quality_score": feature_vector.get("data_quality_score", 0.0), "order_flow_quality": cvd.get("cvd_method", "APPROXIMATE")},
        }
