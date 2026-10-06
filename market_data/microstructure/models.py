from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class MicrostructureBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    buy_volume: float | None = None
    sell_volume: float | None = None
    volume_delta: float | None = None
    bid: float | None = None
    ask: float | None = None
    tick_volume: float | None = None
    order_flow_quality: str = "APPROXIMATE"
    source: str = "ohlcv"


@dataclass
class MicrostructureSetup:
    strategy: str
    direction: str
    entry_reference: float | None = None
    stop_reference: float | None = None
    target_reference: float | None = None
    setup_strength: float | None = None
    data_quality: dict[str, Any] = field(default_factory=dict)
    regime: str | None = None
    timestamp: datetime | None = None


@dataclass
class MicrostructureFeatureVector:
    symbol: str
    timestamp: datetime
    poc: float | None = None
    vah: float | None = None
    val: float | None = None
    value_area_width: float | None = None
    distance_to_poc: float | None = None
    distance_to_vah: float | None = None
    distance_to_val: float | None = None
    inside_value_area: bool | None = None
    above_vah: bool | None = None
    below_val: bool | None = None
    volume_delta: float | None = None
    cvd: float | None = None
    delta_zscore: float | None = None
    absorption_score: float | None = None
    possible_absorption: bool | None = None
    cvd_bullish_divergence: bool | None = None
    cvd_bearish_divergence: bool | None = None
    divergence_strength: float | None = None
    asian_high: float | None = None
    asian_low: float | None = None
    asian_range: float | None = None
    distance_to_asian_high: float | None = None
    distance_to_asian_low: float | None = None
    sweep_above_high: bool | None = None
    sweep_below_low: bool | None = None
    sweep_distance: float | None = None
    return_inside_range: bool | None = None
    bullish_fvg: bool | None = None
    bearish_fvg: bool | None = None
    fvg_size: float | None = None
    fvg_fill_percentage: float | None = None
    regime: str | None = None
    realized_volatility: float | None = None
    atr: float | None = None
    atr_percentile: float | None = None
    trend_strength: float | None = None
    volume_quality: str | None = None
    order_flow_quality: str | None = None
    data_age_seconds: float | None = None
    data_quality_score: float | None = None
    feature_quality: str = "UNKNOWN"

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "poc": self.poc,
            "vah": self.vah,
            "val": self.val,
            "value_area_width": self.value_area_width,
            "distance_to_poc": self.distance_to_poc,
            "distance_to_vah": self.distance_to_vah,
            "distance_to_val": self.distance_to_val,
            "inside_value_area": self.inside_value_area,
            "above_vah": self.above_vah,
            "below_val": self.below_val,
            "volume_delta": self.volume_delta,
            "cvd": self.cvd,
            "delta_zscore": self.delta_zscore,
            "absorption_score": self.absorption_score,
            "possible_absorption": self.possible_absorption,
            "cvd_bullish_divergence": self.cvd_bullish_divergence,
            "cvd_bearish_divergence": self.cvd_bearish_divergence,
            "divergence_strength": self.divergence_strength,
            "asian_high": self.asian_high,
            "asian_low": self.asian_low,
            "asian_range": self.asian_range,
            "distance_to_asian_high": self.distance_to_asian_high,
            "distance_to_asian_low": self.distance_to_asian_low,
            "sweep_above_high": self.sweep_above_high,
            "sweep_below_low": self.sweep_below_low,
            "sweep_distance": self.sweep_distance,
            "return_inside_range": self.return_inside_range,
            "bullish_fvg": self.bullish_fvg,
            "bearish_fvg": self.bearish_fvg,
            "fvg_size": self.fvg_size,
            "fvg_fill_percentage": self.fvg_fill_percentage,
            "regime": self.regime,
            "realized_volatility": self.realized_volatility,
            "atr": self.atr,
            "atr_percentile": self.atr_percentile,
            "trend_strength": self.trend_strength,
            "volume_quality": self.volume_quality,
            "order_flow_quality": self.order_flow_quality,
            "data_age_seconds": self.data_age_seconds,
            "data_quality_score": self.data_quality_score,
            "feature_quality": self.feature_quality,
        }
