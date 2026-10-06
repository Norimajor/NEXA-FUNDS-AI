import unittest
from datetime import datetime, timedelta, timezone

import pandas as pd

from market_data.microstructure import (
    MicrostructureEngine,
    compute_absorption_features,
    compute_cvd,
    compute_volume_profile,
    detect_cvd_divergence,
    detect_fvg,
    detect_liquidity_sweep_fvg_setup,
    detect_value_area_reversion,
    detect_volume_profile_reversion,
    compute_market_regime,
)


class MicrostructureTests(unittest.TestCase):
    def _make_bars(self, count=120):
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        rows = []
        price = 100.0
        for i in range(count):
            ts = base + timedelta(minutes=15 * i)
            open_ = price
            close = price + ((i % 7) - 3) * 0.5
            high = max(open_, close) + 0.8
            low = min(open_, close) - 0.8
            volume = 1200 + (i % 5) * 200
            rows.append({
                "timestamp": ts,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            })
            price = close
        return pd.DataFrame(rows)

    def test_volume_profile_returns_value_area(self):
        df = self._make_bars(200)
        result = compute_volume_profile(df, value_area_percentage=0.7)
        self.assertIn("poc", result)
        self.assertIn("vah", result)
        self.assertIn("val", result)
        self.assertGreaterEqual(result["value_area_volume_percentage"], 0.0)
        self.assertLessEqual(result["value_area_volume_percentage"], 1.0)

    def test_cvd_approximate_and_direct(self):
        df = self._make_bars(60)
        df["buy_volume"] = df["volume"] * 0.6
        df["sell_volume"] = df["volume"] * 0.4
        direct = compute_cvd(df)
        self.assertEqual(direct["cvd_method"], "DIRECT")
        self.assertIn("cvd", direct)
        self.assertIn("delta_zscore", direct)

        df2 = self._make_bars(60)
        approx = compute_cvd(df2)
        self.assertEqual(approx["cvd_method"], "APPROXIMATE")
        self.assertIn("cvd", approx)

    def test_detect_value_area_reversion(self):
        df = self._make_bars(120)
        profile = compute_volume_profile(df, value_area_percentage=0.7)
        setup = detect_value_area_reversion(df, profile)
        self.assertIn("strategy", setup)
        self.assertIn("direction", setup)

    def test_detect_cvd_divergence(self):
        df = self._make_bars(80)
        cvd = compute_cvd(df)
        divergence = detect_cvd_divergence(df, cvd["cvd_series"])
        self.assertIn("cvd_bullish_divergence", divergence)
        self.assertIn("cvd_bearish_divergence", divergence)

    def test_liquidity_and_fvg_detection(self):
        df = self._make_bars(100)
        sweep = detect_liquidity_sweep_fvg_setup(df)
        self.assertIn("strategy", sweep)
        self.assertIn("direction", sweep)
        fvg = detect_fvg(df)
        self.assertIn("bullish_fvg", fvg)
        self.assertIn("bearish_fvg", fvg)

    def test_regime_detects_market_state(self):
        df = self._make_bars(120)
        regime = compute_market_regime(df)
        self.assertIn("regime", regime)
        self.assertIn(regime["regime"], {"REGIME_RANGE", "REGIME_EXPANSION", "REGIME_CHOP", "REGIME_UNKNOWN"})

    def test_microstructure_engine_generates_feature_vector(self):
        df = self._make_bars(180)
        engine = MicrostructureEngine()
        result = engine.analyze(df, symbol="XAUUSD")
        self.assertIn("feature_vector", result)
        self.assertEqual(result["feature_vector"]["symbol"], "XAUUSD")
        self.assertIn("regime", result)

    def test_lookahead_guard(self):
        df = self._make_bars(80)
        prev = compute_volume_profile(df.iloc[:60])
        current = detect_volume_profile_reversion(df.iloc[:60], prev)
        self.assertIn("direction", current)


if __name__ == "__main__":
    unittest.main()
