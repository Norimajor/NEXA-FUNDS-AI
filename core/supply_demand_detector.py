from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class SupplyDemandDetector:
    """Detect demand/supply zones from impulse-origin candles.

    Zones are based on a base candle followed by statistically meaningful
    displacement. The result includes price coordinates so the frontend can
    draw zones directly.
    """
    lookback: int = 180
    displacement_atr: float = 1.25
    max_zones: int = 10

    def detect(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        start = max(1, len(df) - self.lookback)
        zones: list[dict[str, Any]] = []

        for i in range(start, len(df) - 3):
            row = df.iloc[i]
            impulse = df.iloc[i + 1]
            atr = float(row["atr_14"]) if pd.notna(row["atr_14"]) else 0.0
            if atr <= 0:
                continue

            displacement = abs(float(impulse["close"]) - float(impulse["open"]))
            if displacement < atr * self.displacement_atr:
                continue

            base_open = float(row["open"])
            base_close = float(row["close"])
            base_high = float(row["high"])
            base_low = float(row["low"])

            # Demand: bearish/base candle followed by strong bullish expansion.
            if float(impulse["close"]) > float(impulse["open"]) and base_close <= base_open:
                zone_low = base_low
                zone_high = max(base_open, base_close)
                kind = "demand"

            # Supply: bullish/base candle followed by strong bearish expansion.
            elif float(impulse["close"]) < float(impulse["open"]) and base_close >= base_open:
                zone_low = min(base_open, base_close)
                zone_high = base_high
                kind = "supply"
            else:
                continue

            strength = displacement / atr
            current = float(df.iloc[-1]["close"])

            # Count subsequent closes that invalidate the zone.
            future = df.iloc[i + 2:]
            if kind == "demand":
                invalidated = bool((future["close"] < zone_low).any())
            else:
                invalidated = bool((future["close"] > zone_high).any())

            distance = 0.0
            if current < zone_low:
                distance = zone_low - current
            elif current > zone_high:
                distance = current - zone_high

            zones.append({
                "timestamp": row["timestamp"].isoformat(),
                "type": kind,
                "low": round(zone_low, 8),
                "high": round(zone_high, 8),
                "mid": round((zone_low + zone_high) / 2, 8),
                "strength": round(strength, 2),
                "invalidated": invalidated,
                "distance": round(distance, 8),
                "source_index": i,
            })

        # Keep the newest valid zones and annotate whether current price is
        # inside/retesting/away from the zone.
        current = float(df.iloc[-1]["close"])
        valid = [z for z in zones if not z["invalidated"]]
        valid.sort(key=lambda z: z["source_index"], reverse=True)

        for z in valid:
            if z["low"] <= current <= z["high"]:
                z["status"] = "retesting"
            elif z["type"] == "demand" and current > z["high"]:
                z["status"] = "below_zone"
            elif z["type"] == "supply" and current < z["low"]:
                z["status"] = "above_zone"
            else:
                z["status"] = "away"

        return valid[: self.max_zones]

    def nearest(self, zones: list[dict[str, Any]], price: float, zone_type: str | None = None):
        candidates = [z for z in zones if zone_type is None or z["type"] == zone_type]
        if not candidates:
            return None
        return min(candidates, key=lambda z: abs(float(z["mid"]) - price))
