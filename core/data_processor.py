from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}


class MarketDataProcessor:
    """Validate OHLCV candles and calculate analysis features."""

    def prepare(self, candles: Iterable[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
        df = candles.copy() if isinstance(candles, pd.DataFrame) else pd.DataFrame(list(candles))
        if df.empty:
            raise ValueError("No candle data was supplied.")

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        df = df[list(REQUIRED_COLUMNS)].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna().sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

        if len(df) < 80:
            raise ValueError("At least 80 valid candles are required.")

        if (df["high"] < df[["open", "close"]].max(axis=1)).any():
            raise ValueError("Invalid OHLC data: high is below open or close.")
        if (df["low"] > df[["open", "close"]].min(axis=1)).any():
            raise ValueError("Invalid OHLC data: low is above open or close.")
        if (df["high"] < df["low"]).any():
            raise ValueError("Invalid OHLC data: high is below low.")

        return self._features(df)

    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        close, high, low = df["close"], df["high"], df["low"]

        for span in (20, 50, 200):
            df[f"ema_{span}"] = close.ewm(span=span, adjust=False, min_periods=span).mean()

        delta = close.diff()
        gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
        ag = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        al = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        rs = ag / al.replace(0, np.nan)
        df["rsi_14"] = (100 - 100 / (1 + rs)).fillna(50)

        tr = pd.concat(
            [high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1
        ).max(axis=1)
        df["atr_14"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()

        e12 = close.ewm(span=12, adjust=False).mean()
        e26 = close.ewm(span=26, adjust=False).mean()
        df["macd"] = e12 - e26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        typical = (high + low + close) / 3
        df["vwap_20"] = (
            (typical * df["volume"]).rolling(20).sum()
            / df["volume"].rolling(20).sum().replace(0, np.nan)
        )

        df["volume_sma_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"].replace(0, np.nan)
        df["return_5"] = close.pct_change(5) * 100
        df["return_20"] = close.pct_change(20) * 100

        df["range"] = high - low
        df["body"] = (close - df["open"]).abs()
        df["body_ratio"] = df["body"] / df["range"].replace(0, np.nan)
        df["upper_wick"] = high - df[["open", "close"]].max(axis=1)
        df["lower_wick"] = df[["open", "close"]].min(axis=1) - low

        # Timestamp each fractal pivot at the candle where it becomes knowable.
        left = right = 2
        swing_high = (
            high.eq(high.rolling(left + right + 1, center=True).max())
            & high.shift(left).lt(high)
            & high.shift(-right).lt(high)
        )
        swing_low = (
            low.eq(low.rolling(left + right + 1, center=True).min())
            & low.shift(left).gt(low)
            & low.shift(-right).gt(low)
        )
        df["swing_high"] = swing_high.shift(right, fill_value=False).astype(bool)
        df["swing_low"] = swing_low.shift(right, fill_value=False).astype(bool)
        df["swing_high_price"] = high.where(swing_high).shift(right)
        df["swing_low_price"] = low.where(swing_low).shift(right)
        return df
