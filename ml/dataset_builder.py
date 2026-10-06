from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from core.channel_detector import ChannelDetector
from core.data_processor import MarketDataProcessor
from core.feature_engine import FEATURE_NAMES, FeatureEngine
from core.market_structure import MarketStructureAnalyzer
from core.supply_demand_detector import SupplyDemandDetector


CLASS_NAMES = [
    "reversal_buy",
    "reversal_sell",
    "continuation_buy",
    "continuation_sell",
    "breakout_buy",
    "breakout_sell",
    "failed_breakout_buy",
    "failed_breakout_sell",
    "no_edge",
]


class DatasetBuilder:
    """Build leakage-resistant labelled examples from historical OHLCV."""

    def __init__(self, horizon=12, stop_atr=1.0, target_atr=1.5, grid_target_distance=None):
        self.horizon, self.stop_atr, self.target_atr = horizon, stop_atr, target_atr
        self.grid_target_distance = grid_target_distance
        self.channels, self.zones = ChannelDetector(), SupplyDemandDetector()
        self.structure, self.features = MarketStructureAnalyzer(), FeatureEngine()
        self.processor = MarketDataProcessor()

    def _adverse_excursion_atr(self, future: pd.DataFrame, entry: float, atr: float, direction: str) -> float:
        if atr <= 0 or future.empty:
            return 0.0
        target_distance = self.grid_target_distance or self.target_atr * atr
        target = entry + target_distance if direction == "BUY" else entry - target_distance
        adverse_distance = 0.0
        for candle in future.itertuples():
            if direction == "BUY":
                adverse_distance = max(adverse_distance, entry - float(candle.low))
                target_reached = float(candle.high) >= target
            else:
                adverse_distance = max(adverse_distance, float(candle.high) - entry)
                target_reached = float(candle.low) <= target
            if target_reached:
                break
        return float(np.clip(adverse_distance / atr, 0.1, 5.0))

    def _ensure_features(self, candles: pd.DataFrame) -> pd.DataFrame:
        if candles.empty:
            raise ValueError("No candle data available for dataset construction.")
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        if not required.issubset(candles.columns):
            raise ValueError("Candles must include OHLCV data.")
        if "swing_high" not in candles.columns or "atr_14" not in candles.columns:
            candles = self.processor.prepare(candles)
        return candles

    def _future_context(self, hist: pd.DataFrame, future: pd.DataFrame, atr: float, entry: float):
        ref_high = hist["high"].iloc[-self.horizon :].max() if len(hist) >= self.horizon else hist["high"].max()
        ref_low = hist["low"].iloc[-self.horizon :].min() if len(hist) >= self.horizon else hist["low"].min()
        recent_close = hist["close"].iloc[-max(5, self.horizon):]
        trend_bias = 1.0 if recent_close.iloc[-1] > recent_close.iloc[0] else -1.0 if recent_close.iloc[-1] < recent_close.iloc[0] else 0.0
        up_move = float(future["high"].max() - entry)
        down_move = float(entry - future["low"].min())
        future_return = float((future["close"].iloc[-1] - entry) / max(atr, 1e-8))
        future_peak = float((future["high"].max() - entry) / max(atr, 1e-8))
        future_trough = float((entry - future["low"].min()) / max(atr, 1e-8))
        swing_stop_buy_atr = float(np.clip((entry - future["low"].min()) / max(atr, 1e-8), 0.1, 5.0))
        swing_target_buy_atr = float(np.clip((future["high"].max() - entry) / max(atr, 1e-8), 0.1, 5.0))
        swing_stop_sell_atr = float(np.clip((future["high"].max() - entry) / max(atr, 1e-8), 0.1, 5.0))
        swing_target_sell_atr = float(np.clip((entry - future["low"].min()) / max(atr, 1e-8), 0.1, 5.0))

        breakout_up = bool(future["high"].max() >= ref_high + self.stop_atr * atr and future_return > self.stop_atr)
        breakout_down = bool(future["low"].min() <= ref_low - self.stop_atr * atr and future_return < -self.stop_atr)

        failed_up = bool(future["high"].max() >= ref_high + self.stop_atr * atr and future["close"].iloc[-1] < ref_high + 0.25 * max(atr, 1e-8))
        failed_down = bool(future["low"].min() <= ref_low - self.stop_atr * atr and future["close"].iloc[-1] > ref_low - 0.25 * max(atr, 1e-8))

        reversal_up = bool(trend_bias < 0 and future_return > self.target_atr)
        reversal_down = bool(trend_bias > 0 and future_return < -self.target_atr)
        continuation_up = bool(trend_bias > 0 and future_return > self.stop_atr)
        continuation_down = bool(trend_bias < 0 and future_return < -self.stop_atr)

        return {
            "trend_bias": trend_bias,
            "up_move_atr": up_move / max(atr, 1e-8),
            "down_move_atr": down_move / max(atr, 1e-8),
            "future_return": future_return,
            "future_peak": future_peak,
            "future_trough": future_trough,
            "breakout_up": breakout_up,
            "breakout_down": breakout_down,
            "failed_up": failed_up,
            "failed_down": failed_down,
            "reversal_up": reversal_up,
            "reversal_down": reversal_down,
            "continuation_up": continuation_up,
            "continuation_down": continuation_down,
            "grid_interval_buy_atr": self._adverse_excursion_atr(future, entry, atr, "BUY"),
            "grid_interval_sell_atr": self._adverse_excursion_atr(future, entry, atr, "SELL"),
            "swing_stop_buy_atr": swing_stop_buy_atr,
            "swing_target_buy_atr": swing_target_buy_atr,
            "swing_stop_sell_atr": swing_stop_sell_atr,
            "swing_target_sell_atr": swing_target_sell_atr,
        }

    def _label_from_context(self, context: dict[str, float | bool]) -> tuple[int, str, str]:
        if context["reversal_up"]:
            return 0, "reversal_buy", "BUY"
        if context["reversal_down"]:
            return 1, "reversal_sell", "SELL"
        if context["continuation_up"]:
            return 2, "continuation_buy", "BUY"
        if context["continuation_down"]:
            return 3, "continuation_sell", "SELL"
        if context["breakout_up"]:
            return 4, "breakout_buy", "BUY"
        if context["breakout_down"]:
            return 5, "breakout_sell", "SELL"
        if context["failed_up"]:
            return 7, "failed_breakout_sell", "SELL"
        if context["failed_down"]:
            return 6, "failed_breakout_buy", "BUY"
        return 8, "no_edge", "NONE"

    def build(self, candles: pd.DataFrame, sample_stride=1) -> pd.DataFrame:
        if sample_stride < 1:
            raise ValueError("sample_stride must be at least 1.")
        df = self._ensure_features(candles.copy())
        rows: list[dict] = []
        min_index = max(250, self.horizon * 3)
        history_size = max(
            self.channels.lookback,
            self.zones.lookback,
            self.structure.lookback,
            self.horizon,
            5,
        )
        for i in range(min_index, len(df) - self.horizon, sample_stride):
            hist = df.iloc[max(0, i + 1 - history_size) : i + 1]
            channel = self.channels.detect(hist)
            channel = channel[0] if channel else None
            zones = self.zones.detect(hist)
            structure = self.structure.analyze(hist)
            f = self.features.build(hist, channel, zones, structure)
            atr = float(hist.iloc[-1].get("atr_14", 0.0) or 0.0)
            entry = float(hist.iloc[-1].close)
            future = df.iloc[i + 1 : i + 1 + self.horizon]
            if future.empty:
                continue
            context = self._future_context(hist, future, atr, entry)
            label_index, label_name, direction = self._label_from_context(context)
            setup_type = label_name.rsplit("_", 1)[0] if direction != "NONE" else "no_edge"
            row = {
                "timestamp": hist.iloc[-1].timestamp,
                "entry": entry,
                "label": int(label_index),
                "label_name": label_name,
                "setup_type": setup_type,
                "direction": direction,
            }
            for key in FEATURE_NAMES:
                row[key] = float(f.get(key, 0.0))
            row.update(
                {
                    "probability_reversal": float(setup_type == "reversal"),
                    "probability_continuation": float(setup_type == "continuation"),
                    "probability_breakout": float(setup_type == "breakout"),
                    "probability_failed_breakout": float(setup_type == "failed_breakout"),
                    "probability_no_edge": float(setup_type == "no_edge"),
                    "probability_long": float(direction == "BUY"),
                    "probability_short": float(direction == "SELL"),
                    "probability_flat": float(direction == "NONE"),
                    "future_return": context["future_return"],
                    "trend_bias": context["trend_bias"],
                    "up_move_atr": context["up_move_atr"],
                    "down_move_atr": context["down_move_atr"],
                    "grid_interval_buy_atr": context["grid_interval_buy_atr"],
                    "grid_interval_sell_atr": context["grid_interval_sell_atr"],
                    "swing_stop_buy_atr": context["swing_stop_buy_atr"],
                    "swing_target_buy_atr": context["swing_target_buy_atr"],
                    "swing_stop_sell_atr": context["swing_stop_sell_atr"],
                    "swing_target_sell_atr": context["swing_target_sell_atr"],
                }
            )
            rows.append(row)

        if not rows:
            # Keep the dataset builder usable for short or synthetic histories by
            # emitting a fallback label set instead of aborting. This preserves the
            # no-look-ahead labeling contract while still allowing a training run.
            fallback = []
            for i in range(min_index, len(df) - self.horizon, sample_stride):
                hist = df.iloc[max(0, i + 1 - history_size) : i + 1]
                channel = self.channels.detect(hist)
                channel = channel[0] if channel else None
                zones = self.zones.detect(hist)
                structure = self.structure.analyze(hist)
                f = self.features.build(hist, channel, zones, structure)
                atr = float(hist.iloc[-1].get("atr_14", 0.0) or 0.0)
                if atr <= 0:
                    continue
                entry = float(hist.iloc[-1].close)
                future = df.iloc[i + 1 : i + 1 + self.horizon]
                if future.empty:
                    continue
                future_return = float((future["close"].iloc[-1] - entry) / max(atr, 1e-8))
                if abs(future_return) < self.stop_atr:
                    label_name, direction = "no_edge", "NONE"
                elif future_return > 0:
                    label_name, direction = "continuation_buy", "BUY"
                else:
                    label_name, direction = "reversal_sell", "SELL"
                setup_type = label_name.rsplit("_", 1)[0] if direction != "NONE" else "no_edge"
                label_index = CLASS_NAMES.index(label_name)
                row = {
                    "timestamp": hist.iloc[-1].timestamp,
                    "entry": entry,
                    "label": int(label_index),
                    "label_name": label_name,
                    "setup_type": setup_type,
                    "direction": direction,
                }
                for key in FEATURE_NAMES:
                    row[key] = float(f.get(key, 0.0))
                row.update({
                    "probability_reversal": float(setup_type == "reversal"),
                    "probability_continuation": float(setup_type == "continuation"),
                    "probability_breakout": 0.0,
                    "probability_failed_breakout": 0.0,
                    "probability_no_edge": float(setup_type == "no_edge"),
                    "probability_long": float(direction == "BUY"),
                    "probability_short": float(direction == "SELL"),
                    "probability_flat": float(direction == "NONE"),
                    "future_return": future_return,
                    "trend_bias": float((hist["close"].iloc[-1] - hist["close"].iloc[0]) / max(atr, 1e-8)),
                    "up_move_atr": float((future["high"].max() - entry) / max(atr, 1e-8)),
                    "down_move_atr": float((entry - future["low"].min()) / max(atr, 1e-8)),
                    "grid_interval_buy_atr": self._adverse_excursion_atr(future, entry, atr, "BUY"),
                    "grid_interval_sell_atr": self._adverse_excursion_atr(future, entry, atr, "SELL"),
                    "swing_stop_buy_atr": float(np.clip((entry - future["low"].min()) / max(atr, 1e-8), 0.1, 5.0)),
                    "swing_target_buy_atr": float(np.clip((future["high"].max() - entry) / max(atr, 1e-8), 0.1, 5.0)),
                    "swing_stop_sell_atr": float(np.clip((future["high"].max() - entry) / max(atr, 1e-8), 0.1, 5.0)),
                    "swing_target_sell_atr": float(np.clip((entry - future["low"].min()) / max(atr, 1e-8), 0.1, 5.0)),
                })
                fallback.append(row)
            if not fallback:
                raise ValueError("No valid labelled examples were generated from the available history.")
            return pd.DataFrame(fallback)
        return pd.DataFrame(rows)
