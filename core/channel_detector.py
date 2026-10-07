from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ChannelDetector:
    """Detect statistically supported price channels.

    The detector fits parallel regression lines to confirmed swing highs/lows.
    A channel is accepted only when both boundaries have meaningful touches,
    reasonable width relative to ATR, and a usable fit quality.

    Coordinates are returned in timestamp/price form for frontend rendering.
    """

    lookback: int = 220
    min_touches: int = 2
    max_channels: int = 4
    min_r2: float = 0.25

    def detect(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        work = df.tail(self.lookback).reset_index(drop=True)
        if len(work) < 80:
            return []

        highs = self._pivots(work, "swing_high", "swing_high_price" if "swing_high_price" in work else "high")
        lows = self._pivots(work, "swing_low", "swing_low_price" if "swing_low_price" in work else "low")
        if len(highs) < 2 or len(lows) < 2:
            return []

        atr = float(work.iloc[-1]["atr_14"])
        if not np.isfinite(atr) or atr <= 0:
            return []

        candidates = []
        # Highs define the upper boundary. For a channel, the lower boundary
        # uses the same slope and is positioned to contain the observed lows.
        for p1 in range(len(highs) - 1):
            for p2 in range(p1 + 1, len(highs)):
                h1, h2 = highs[p1], highs[p2]
                dx = h2["x"] - h1["x"]
                if dx < 10:
                    continue

                slope = (h2["price"] - h1["price"]) / dx
                intercept_high = h1["price"] - slope * h1["x"]

                low_intercepts = [p["price"] - slope * p["x"] for p in lows]
                if not low_intercepts:
                    continue

                # Place lower boundary just below the lowest projected low.
                intercept_low = min(low_intercepts)

                # Reject excessively wide/narrow channels.
                width = intercept_high - intercept_low
                if width <= 0 or width < atr * 0.8 or width > atr * 12:
                    continue

                upper_touches = self._touches(highs, slope, intercept_high, atr)
                lower_touches = self._touches(lows, slope, intercept_low, atr)
                if upper_touches < self.min_touches or lower_touches < self.min_touches:
                    continue

                r2_high = self._r2(highs, slope, intercept_high)
                r2_low = self._r2(lows, slope, intercept_low)
                score = (
                    min(r2_high, 1) * 30
                    + min(r2_low, 1) * 30
                    + min(upper_touches, 5) * 5
                    + min(lower_touches, 5) * 5
                )

                candidates.append(
                    self._make_channel(
                        work, slope, intercept_low, intercept_high,
                        upper_touches, lower_touches, r2_high, r2_low, score, atr
                    )
                )

        # Deduplicate overlapping channels and retain strongest.
        candidates.sort(key=lambda c: c["score"], reverse=True)
        selected = []
        for channel in candidates:
            if any(self._similar(channel, x, atr) for x in selected):
                continue
            selected.append(channel)
            if len(selected) >= self.max_channels:
                break
        return selected

    @staticmethod
    def _pivots(df, flag, price_col):
        points = []
        for i, row in df.iterrows():
            if bool(row[flag]) and pd.notna(row[price_col]):
                points.append({"x": i, "price": float(row[price_col])})
        return points

    @staticmethod
    def _touches(points, slope, intercept, atr):
        tolerance = max(atr * 0.35, 1e-12)
        return sum(abs(p["price"] - (slope * p["x"] + intercept)) <= tolerance for p in points)

    @staticmethod
    def _r2(points, slope, intercept):
        if len(points) < 2:
            return 0.0
        y = np.array([p["price"] for p in points])
        predicted = np.array([slope * p["x"] + intercept for p in points])
        ss_res = float(np.sum((y - predicted) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return 0.0 if ss_tot == 0 else max(0.0, 1 - ss_res / ss_tot)

    def _make_channel(self, df, slope, low_b, high_b, ut, lt, r2h, r2l, score, atr):
        last_x = len(df) - 1
        first_x = 0

        lower_now = slope * last_x + low_b
        upper_now = slope * last_x + high_b
        mid_now = (lower_now + upper_now) / 2
        price = float(df.iloc[-1]["close"])

        if abs(slope) < atr * 0.005:
            direction = "sideways"
        elif slope > 0:
            direction = "ascending"
        else:
            direction = "descending"

        if price > upper_now:
            location = "above"
        elif price < lower_now:
            location = "below"
        elif price >= mid_now:
            location = "upper_half"
        else:
            location = "lower_half"

        # Directional preference for the EA: only trend channels get a
        # directional bias. Sideways channels produce RANGE.
        if direction == "ascending":
            bias = "BUY"
        elif direction == "descending":
            bias = "SELL"
        else:
            bias = "RANGE"

        return {
            "id": f"channel_{first_x}_{last_x}_{round(slope, 10)}",
            "direction": direction,
            "bias": bias,
            "score": round(float(score), 2),
            "r2_upper": round(float(r2h), 3),
            "r2_lower": round(float(r2l), 3),
            "upper_touches": int(ut),
            "lower_touches": int(lt),
            "slope_per_candle": round(float(slope), 10),
            "width": round(float(high_b - low_b), 8),
            "location": location,
            "upper": {
                "start": {"timestamp": df.iloc[first_x]["timestamp"].isoformat(), "price": round(slope*first_x+high_b, 8)},
                "end": {"timestamp": df.iloc[last_x]["timestamp"].isoformat(), "price": round(upper_now, 8)},
            },
            "lower": {
                "start": {"timestamp": df.iloc[first_x]["timestamp"].isoformat(), "price": round(slope*first_x+low_b, 8)},
                "end": {"timestamp": df.iloc[last_x]["timestamp"].isoformat(), "price": round(lower_now, 8)},
            },
            "current": {
                "lower": round(lower_now, 8),
                "mid": round(mid_now, 8),
                "upper": round(upper_now, 8),
            },
        }

    @staticmethod
    def _similar(a, b, atr):
        return (
            a["direction"] == b["direction"]
            and abs(a["current"]["mid"] - b["current"]["mid"]) < atr * 0.8
            and abs(a["width"] - b["width"]) < atr * 2
        )
