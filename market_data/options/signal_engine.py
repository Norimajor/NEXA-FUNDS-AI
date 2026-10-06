from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from market_data.models import OptionContract
from .features import OptionsFeatureEngine


class OptionsSignalEngine:
    """Deterministic quantitative signal engine using only validated option data."""

    def __init__(self, feature_engine: OptionsFeatureEngine | None = None):
        self.feature_engine = feature_engine or OptionsFeatureEngine()

    def generate(self, contracts: Iterable[OptionContract], asset: str = "BTC", reference_price: float | None = None) -> dict[str, Any]:
        contract_list = [contract for contract in contracts if contract is not None]
        if not contract_list:
            return {
                "asset": asset.upper(),
                "signal": "HOLD",
                "score": 0.0,
                "confidence": 0.0,
                "regime": "NO_DATA",
                "signal_id": "no-data",
                "signal_timestamp": datetime.now(timezone.utc).isoformat(),
                "signal_expiry": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
                "entry_reference": reference_price,
                "stop_loss": None,
                "take_profit": None,
                "features": {},
                "reasons": ["No option contracts available."],
                "data_quality": {"sufficient": False, "missing_fields": ["contracts"]},
            }

        features = self.feature_engine.build(contract_list, asset=asset, reference_price=reference_price)
        score = self._score(features)
        signal, confidence, reasons = self._convert_score_to_signal(score, features)

        return {
            "asset": features.asset,
            "signal": signal,
            "score": round(score, 6),
            "confidence": round(confidence, 6),
            "regime": self._regime(features),
            "signal_id": f"{features.asset.lower()}-{int(datetime.now(timezone.utc).timestamp() * 1000)}",
            "signal_timestamp": datetime.now(timezone.utc).isoformat(),
            "signal_expiry": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
            "entry_reference": reference_price if reference_price is not None else features.market_positioning_features.get("call_oi"),
            "stop_loss": None,
            "take_profit": None,
            "features": features.to_dict(),
            "reasons": reasons,
            "data_quality": features.data_quality,
        }

    def _score(self, features) -> float:
        score = 0.0
        oi_ratio = features.put_call_oi_ratio
        if oi_ratio is not None:
            # bullish if calls dominate, bearish if puts dominate
            score += (oi_ratio - 1.0) * 0.5
        gamma = features.gamma_exposure or 0.0
        if gamma != 0:
            score += (gamma / max(abs(gamma), 1.0)) * 0.25
        delta = features.delta_exposure or 0.0
        if delta != 0:
            score += (delta / max(abs(delta), 1.0)) * 0.25
        if features.distance_to_major_oi_strike is not None:
            score -= min(features.distance_to_major_oi_strike / max(float(features.distance_to_major_oi_strike or 1.0), 1.0), 1.0) * 0.10
        return max(-1.0, min(1.0, score))

    def _convert_score_to_signal(self, score: float, features) -> tuple[str, float, list[str]]:
        reasons: list[str] = []
        if not features.data_quality.get("sufficient", False):
            return "HOLD", 0.0, ["Insufficient option data quality for a trade signal."]
        if score >= 0.35:
            reasons.append("Call-side open interest and gamma exposure favor upside.")
            return "BUY", min(0.95, 0.55 + score * 0.5), reasons
        if score <= -0.35:
            reasons.append("Put-side open interest and gamma exposure favor downside.")
            return "SELL", min(0.95, 0.55 + abs(score) * 0.5), reasons
        reasons.append("Option positioning is balanced; no directional edge established.")
        return "HOLD", max(0.0, 0.5 - abs(score) * 0.5), reasons

    def _regime(self, features) -> str:
        if features.put_call_oi_ratio is None:
            return "UNKNOWN"
        if features.put_call_oi_ratio > 1.05:
            return "BEARISH"
        if features.put_call_oi_ratio < 0.95:
            return "BULLISH"
        return "BALANCED"
