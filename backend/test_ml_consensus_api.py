from datetime import datetime, timedelta, timezone
import unittest
from copy import deepcopy
from unittest.mock import patch

from fastapi.testclient import TestClient

from api import ml_consensus
from backend.api import app


class MlConsensusApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        now = datetime.now(timezone.utc)
        candles = [
            {
                "timestamp": (now - timedelta(minutes=index)).isoformat(),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1000.0,
            }
            for index in range(260)
        ]
        self.payload = {
            "symbol": "XAUUSD",
            "h1_candles": candles,
            "m15_candles": candles,
        }
        self.h1_result = {
            "signal": "BUY",
            "confidence": 74.0,
            "setup_type": "continuation",
            "probabilities": {"buy": 74.0, "sell": 10.0},
            "model_version": "gold_h1",
        }
        self.m15_result = {
            "signal": "BUY",
            "direction": "BUY",
            "confidence": 68.0,
            "setup_type": "breakout",
            "probabilities": {"buy": 68.0, "sell": 12.0},
            "model_version": "gold_m15",
            "price": 100.0,
            "atr": 0.5,
            "grid_interval_buy_atr": 1.25,
            "grid_interval_sell_atr": 1.5,
            "candle_low": 99.0,
            "candle_high": 101.0,
            "levels": {
                "potential_bullish_reversal": {
                    "low": 99.5,
                    "high": 100.5,
                    "mid": 100.0,
                },
                "potential_bearish_reversal": {
                    "low": 100.5,
                    "high": 101.5,
                    "mid": 101.0,
                },
                "potential_bullish_reversal_source": "demand_zone",
                "potential_bearish_reversal_source": "supply_zone",
            },
        }

    def test_consensus_route_accepts_ea_payload_and_returns_consensus_shape(self):
        self.assertIn("/api/ml/predict/consensus", app.openapi()["paths"])
        self.assertIn(
            "post",
            app.openapi()["paths"]["/api/ml/predict/consensus"],
        )

        with (
            patch.object(ml_consensus, "INGEST_API_KEY", ""),
            patch.object(
                ml_consensus,
                "_prepare_and_analyze",
                side_effect=[self.h1_result, self.m15_result],
            ),
        ):
            response = self.client.post("/api/ml/predict/consensus", json=self.payload)

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["symbol"], "XAUUSD")
        self.assertEqual(result["timeframe"], "H1+M15")
        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["direction"], "BUY")
        self.assertEqual(result["recommended_mode"], "SWING")
        self.assertEqual(result["signal_reason"], "H1_M15_DIRECTION_CONFIRMED")
        self.assertEqual(result["trend_direction"], "BUY")
        self.assertEqual(result["confirmation_status"], "DIRECTION_CONFIRMED")
        self.assertEqual(result["confidence"], 68.0)
        self.assertEqual(result["setup_type"], "breakout")
        self.assertEqual(result["h1_model_version"], "gold_h1")
        self.assertEqual(result["m15_model_version"], "gold_m15")
        self.assertIn("buy_probability", result)
        self.assertIn("sell_probability", result)
        self.assertEqual(result["planned_direction"], "BUY")
        self.assertEqual(result["entry_direction"], "BUY")
        self.assertTrue(result["entry_triggered"])
        self.assertEqual(result["entry_status"], "ZONE_TOUCHED")
        self.assertEqual(result["entry_zone_low"], 99.5)
        self.assertEqual(result["entry_zone_high"], 100.5)
        self.assertEqual(result["poi_direction"], "BUY")
        self.assertIn("M15 top class breakout_buy", result["setup_reason"])
        self.assertIn("runner-up unavailable", result["setup_reason"])
        self.assertIn("demand_zone", result["poi_reason"])
        self.assertIn("H1 trend BUY", result["decision_explanation"])
        self.assertEqual(result["grid_interval_buy_atr"], 1.25)
        self.assertEqual(result["grid_interval_sell_atr"], 1.5)

    def test_consensus_accepts_a_learned_m15_reversal_against_h1_bias(self):
        h1_result = deepcopy(self.h1_result)
        m15_result = deepcopy(self.m15_result)
        h1_result.update(signal="BUY", direction="BUY", setup_type="continuation")
        m15_result.update(
            signal="SELL",
            direction="SELL",
            setup_type="reversal",
            confidence=66.0,
        )
        m15_result["candle_low"] = 99.0
        m15_result["candle_high"] = 100.0

        with (
            patch.object(ml_consensus, "INGEST_API_KEY", ""),
            patch.object(
                ml_consensus,
                "_prepare_and_analyze",
                side_effect=[h1_result, m15_result],
            ),
        ):
            response = self.client.post("/api/ml/predict/consensus", json=self.payload)

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["trend_direction"], "BUY")
        self.assertEqual(result["planned_direction"], "SELL")
        self.assertEqual(result["signal"], "SELL")
        self.assertEqual(result["confirmation_status"], "REVERSAL_CONFIRMED")
        self.assertEqual(result["signal_reason"], "M15_REVERSAL_CONFIRMED")
        self.assertEqual(result["entry_direction"], "WAIT")
        self.assertEqual(result["entry_status"], "WAIT_FOR_ZONE")

    def test_consensus_waits_for_m15_learned_confirmation(self):
        m15_result = deepcopy(self.m15_result)
        m15_result.update(signal="WAIT", direction="BUY", confidence=58.0)

        with (
            patch.object(ml_consensus, "INGEST_API_KEY", ""),
            patch.object(
                ml_consensus,
                "_prepare_and_analyze",
                side_effect=[self.h1_result, m15_result],
            ),
        ):
            response = self.client.post("/api/ml/predict/consensus", json=self.payload)

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["trend_direction"], "BUY")
        self.assertEqual(result["planned_direction"], "WAIT")
        self.assertEqual(result["confirmation_status"], "WAIT_M15_CONFIRMATION")
        self.assertEqual(result["entry_status"], "WAIT_FOR_M15_CONFIRMATION")
        self.assertEqual(result["entry_direction"], "WAIT")
        self.assertEqual(result["poi_direction"], "BUY")
        self.assertGreater(result["entry_zone_low"], 0)
        self.assertIn("no listed context cue", result["setup_context"])

    def test_consensus_uses_market_trend_when_h1_setup_classifier_returns_wait(self):
        h1_result = deepcopy(self.h1_result)
        m15_result = deepcopy(self.m15_result)
        h1_result.update(
            signal="WAIT",
            direction="NONE",
            setup_type="no_edge",
            trend_direction="SELL",
            trend_strength=0.86,
            trend_reason="PRICE_BELOW_BEARISH_EMA_STACK|LOWER_SWING_HIGHS_AND_LOWS",
        )
        m15_result.update(
            signal="SELL",
            direction="SELL",
            setup_type="continuation",
            confidence=67.0,
        )

        with (
            patch.object(ml_consensus, "INGEST_API_KEY", ""),
            patch.object(
                ml_consensus,
                "_prepare_and_analyze",
                side_effect=[h1_result, m15_result],
            ),
        ):
            response = self.client.post("/api/ml/predict/consensus", json=self.payload)

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["trend_direction"], "SELL")
        self.assertEqual(result["trend_strength"], 0.86)
        self.assertEqual(result["signal"], "SELL")
        self.assertEqual(result["signal_reason"], "H1_M15_DIRECTION_CONFIRMED")
        self.assertEqual(result["confidence"], 67.0)

    def test_consensus_waits_until_latest_candle_touches_entry_zone(self):
        m15_result = deepcopy(self.m15_result)
        m15_result["candle_low"] = 100.6
        m15_result["candle_high"] = 101.0

        with (
            patch.object(ml_consensus, "INGEST_API_KEY", ""),
            patch.object(
                ml_consensus,
                "_prepare_and_analyze",
                side_effect=[self.h1_result, m15_result],
            ),
        ):
            response = self.client.post("/api/ml/predict/consensus", json=self.payload)

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["planned_direction"], "BUY")
        self.assertEqual(result["entry_direction"], "WAIT")
        self.assertFalse(result["entry_triggered"])
        self.assertEqual(result["entry_status"], "WAIT_FOR_ZONE")

    def test_consensus_route_enforces_configured_api_key(self):
        with patch.object(ml_consensus, "INGEST_API_KEY", "configured-test-key"):
            unauthorized = self.client.post(
                "/api/ml/predict/consensus",
                json=self.payload,
            )
            rejected = self.client.post(
                "/api/ml/predict/consensus",
                json=self.payload,
                headers={"X-API-Key": "incorrect-key"},
            )
            with patch.object(
                ml_consensus,
                "_prepare_and_analyze",
                side_effect=[self.h1_result, self.m15_result],
            ):
                authorized = self.client.post(
                    "/api/ml/predict/consensus",
                    json=self.payload,
                    headers={"X-API-Key": "configured-test-key"},
                )

        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(rejected.status_code, 401)
        self.assertEqual(authorized.status_code, 200, authorized.text)
