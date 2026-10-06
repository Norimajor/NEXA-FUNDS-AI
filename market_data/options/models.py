from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class OptionsFeatureVector:
    asset: str
    timestamp: datetime
    total_call_oi: float | None = None
    total_put_oi: float | None = None
    put_call_oi_ratio: float | None = None
    call_oi_concentration: float | None = None
    put_oi_concentration: float | None = None
    gamma_exposure: float | None = None
    vega_exposure: float | None = None
    theta_exposure: float | None = None
    delta_exposure: float | None = None
    gamma_concentration_by_strike: dict[float, float] | None = None
    gamma_concentration_by_expiration: dict[str, float] | None = None
    oi_concentration_by_strike: dict[float, float] | None = None
    oi_concentration_by_expiration: dict[str, float] | None = None
    distance_to_major_oi_strike: float | None = None
    distance_to_major_gamma_strike: float | None = None
    expiration_concentration: dict[str, float] | None = None
    near_term_oi_concentration: float | None = None
    longer_term_oi_concentration: float | None = None
    call_put_greek_imbalance: float | None = None
    atm_near_atm_greek_exposure: float | None = None
    market_positioning_features: dict[str, float | None] = field(default_factory=dict)
    data_quality: dict[str, Any] = field(default_factory=dict)
    missing_fields: list[str] = field(default_factory=list)
    volume: float | None = None
    iv: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset": self.asset,
            "timestamp": self.timestamp.isoformat(),
            "total_call_oi": self.total_call_oi,
            "total_put_oi": self.total_put_oi,
            "put_call_oi_ratio": self.put_call_oi_ratio,
            "call_oi_concentration": self.call_oi_concentration,
            "put_oi_concentration": self.put_oi_concentration,
            "gamma_exposure": self.gamma_exposure,
            "vega_exposure": self.vega_exposure,
            "theta_exposure": self.theta_exposure,
            "delta_exposure": self.delta_exposure,
            "gamma_concentration_by_strike": self.gamma_concentration_by_strike,
            "gamma_concentration_by_expiration": self.gamma_concentration_by_expiration,
            "oi_concentration_by_strike": self.oi_concentration_by_strike,
            "oi_concentration_by_expiration": self.oi_concentration_by_expiration,
            "distance_to_major_oi_strike": self.distance_to_major_oi_strike,
            "distance_to_major_gamma_strike": self.distance_to_major_gamma_strike,
            "expiration_concentration": self.expiration_concentration,
            "near_term_oi_concentration": self.near_term_oi_concentration,
            "longer_term_oi_concentration": self.longer_term_oi_concentration,
            "call_put_greek_imbalance": self.call_put_greek_imbalance,
            "atm_near_atm_greek_exposure": self.atm_near_atm_greek_exposure,
            "market_positioning_features": self.market_positioning_features,
            "data_quality": self.data_quality,
            "missing_fields": self.missing_fields,
            "volume": self.volume,
            "iv": self.iv,
        }


@dataclass
class SignalRecord:
    asset: str
    signal: str
    score: float
    confidence: float
    regime: str
    timestamp: datetime
    signal_id: str
    expires_at: datetime
    features: dict[str, Any]
    reasons: list[str]
    data_quality: dict[str, Any]
