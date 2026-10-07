import numpy as np
import pandas as pd


FEATURE_NAMES = [
    'ema_distance_20_50', 'ema_distance_50_200', 'price_distance_ema50',
    'rsi', 'macd_hist_atr', 'atr_pct', 'volume_ratio', 'return_5',
    'return_20', 'body_ratio', 'channel_score', 'channel_slope_atr',
    'channel_location', 'channel_bias_buy', 'channel_bias_sell',
    'demand_distance_atr', 'supply_distance_atr', 'demand_strength',
    'supply_strength', 'structure_bull', 'structure_bear',
    'pivot_high_distance_atr', 'pivot_low_distance_atr',
    'pivot_high_age', 'pivot_low_age', 'equal_high_cluster',
    'equal_low_cluster', 'double_top_similarity_atr',
    'double_bottom_similarity_atr', 'swing_high_change_atr',
    'swing_low_change_atr', 'upper_wick_ratio', 'lower_wick_ratio',
    'close_location', 'sweep_high', 'sweep_low',
    'bearish_rejection', 'bullish_rejection',
]

TRAINING_FEATURE_NAMES = [
    name for name in FEATURE_NAMES
    if name not in {
        'channel_score', 'channel_slope_atr', 'channel_location',
        'channel_bias_buy', 'channel_bias_sell', 'demand_distance_atr',
        'supply_distance_atr', 'demand_strength', 'supply_strength',
    }
]


class FeatureEngine:
    @staticmethod
    def _pivot_points(df, flag, price_column, fallback_column):
        flags = df[flag].fillna(False).to_numpy(dtype=bool) if flag in df else np.zeros(len(df), dtype=bool)
        prices = df[price_column] if price_column in df else df[fallback_column]
        return [
            (index, float(prices.iloc[index]))
            for index, confirmed in enumerate(flags)
            if confirmed and pd.notna(prices.iloc[index])
        ]

    def build(self, df, channel, zones, structure):
        r = df.iloc[-1]
        atr = max(float(r.atr_14), 1e-12)
        price = float(r.close)
        demand = min(
            (z for z in zones if z['type'] == 'demand'),
            key=lambda z: abs(z['mid'] - price),
            default=None,
        )
        supply = min(
            (z for z in zones if z['type'] == 'supply'),
            key=lambda z: abs(z['mid'] - price),
            default=None,
        )

        high_pivots = self._pivot_points(df, 'swing_high', 'swing_high_price', 'high')
        low_pivots = self._pivot_points(df, 'swing_low', 'swing_low_price', 'low')
        last_high = high_pivots[-1] if high_pivots else None
        last_low = low_pivots[-1] if low_pivots else None
        previous_highs = high_pivots[-7:-1]
        previous_lows = low_pivots[-7:-1]

        high_distance = np.clip((last_high[1] - price) / atr, -9, 9) if last_high else 9.0
        low_distance = np.clip((price - last_low[1]) / atr, -9, 9) if last_low else 9.0
        high_age = min(len(df) - 1 - last_high[0], 100) / 100 if last_high else 1.0
        low_age = min(len(df) - 1 - last_low[0], 100) / 100 if last_low else 1.0
        high_cluster = (
            sum(abs(level - last_high[1]) <= atr * 0.35 for _, level in high_pivots[-8:])
            if last_high else 0
        )
        low_cluster = (
            sum(abs(level - last_low[1]) <= atr * 0.35 for _, level in low_pivots[-8:])
            if last_low else 0
        )
        double_top = min(
            (abs(level - last_high[1]) / atr for _, level in previous_highs),
            default=5.0,
        ) if last_high else 5.0
        double_bottom = min(
            (abs(level - last_low[1]) / atr for _, level in previous_lows),
            default=5.0,
        ) if last_low else 5.0

        candle_range = max(float(r.high - r.low), 1e-12)
        upper_wick_ratio = float(r.upper_wick) / candle_range
        lower_wick_ratio = float(r.lower_wick) / candle_range
        close_location = float(np.clip((price - float(r.low)) / candle_range, 0, 1))
        prior_high = high_pivots[-1][1] if high_pivots else None
        prior_low = low_pivots[-1][1] if low_pivots else None
        sweep_high = prior_high is not None and float(r.high) > prior_high and price < prior_high
        sweep_low = prior_low is not None and float(r.low) < prior_low and price > prior_low

        return {
            'ema_distance_20_50': float((r.ema_20 - r.ema_50) / atr),
            'ema_distance_50_200': float((r.ema_50 - r.ema_200) / atr) if pd.notna(r.ema_200) else 0,
            'price_distance_ema50': float((price - r.ema_50) / atr),
            'rsi': float(r.rsi_14) / 100,
            'macd_hist_atr': float(r.macd_hist) / atr,
            'atr_pct': float(atr / price),
            'volume_ratio': float(np.clip(r.volume_ratio, 0, 5)),
            'return_5': float(r.return_5),
            'return_20': float(r.return_20),
            'body_ratio': float(np.clip(r.body_ratio, 0, 1)),
            'channel_score': float(channel['score'] / 100) if channel else 0,
            'channel_slope_atr': float(channel['slope_per_candle'] / atr) if channel else 0,
            'channel_location': {'above': 1.2, 'upper_half': .5, 'lower_half': -.5, 'below': -1.2}.get(channel['location'], 0) if channel else 0,
            'channel_bias_buy': 1.0 if channel and channel['bias'] == 'BUY' else 0,
            'channel_bias_sell': 1.0 if channel and channel['bias'] == 'SELL' else 0,
            'demand_distance_atr': float(abs(price - demand['mid']) / atr) if demand else 9,
            'supply_distance_atr': float(abs(price - supply['mid']) / atr) if supply else 9,
            'demand_strength': float(demand['strength']) if demand else 0,
            'supply_strength': float(supply['strength']) if supply else 0,
            'structure_bull': 1.0 if structure['structure'] == 'BULLISH' else 0,
            'structure_bear': 1.0 if structure['structure'] == 'BEARISH' else 0,
            'pivot_high_distance_atr': float(high_distance),
            'pivot_low_distance_atr': float(low_distance),
            'pivot_high_age': float(high_age),
            'pivot_low_age': float(low_age),
            'equal_high_cluster': float(min(max(high_cluster - 1, 0), 4) / 4),
            'equal_low_cluster': float(min(max(low_cluster - 1, 0), 4) / 4),
            'double_top_similarity_atr': float(min(double_top, 5.0)),
            'double_bottom_similarity_atr': float(min(double_bottom, 5.0)),
            'swing_high_change_atr': float(np.clip((last_high[1] - previous_highs[-1][1]) / atr, -5, 5)) if last_high and previous_highs else 0,
            'swing_low_change_atr': float(np.clip((last_low[1] - previous_lows[-1][1]) / atr, -5, 5)) if last_low and previous_lows else 0,
            'upper_wick_ratio': float(np.clip(upper_wick_ratio, 0, 1)),
            'lower_wick_ratio': float(np.clip(lower_wick_ratio, 0, 1)),
            'close_location': close_location,
            'sweep_high': float(sweep_high),
            'sweep_low': float(sweep_low),
            'bearish_rejection': float(upper_wick_ratio >= 0.45 and close_location <= 0.55),
            'bullish_rejection': float(lower_wick_ratio >= 0.45 and close_location >= 0.45),
        }

    def vector(self, features):
        return np.array([float(features.get(name, 0)) for name in FEATURE_NAMES], dtype=float)
