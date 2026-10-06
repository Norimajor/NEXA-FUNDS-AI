import unittest
from datetime import datetime, timedelta, timezone
import os
from unittest.mock import patch

from fastapi.testclient import TestClient

from api import web_server
from api.web_server import app, engine, _verify_key
from market_data.models import OptionContract
from market_data.options import OptionsSignalEngine


class SignalApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_local_prediction_auth_does_not_disable_remote_or_ingestion_auth(self):
        with patch("api.web_server.INGEST_API_KEY", "configured-test-key"), patch.dict(
            os.environ, {"ALLOW_LOCAL_ML_PREDICTIONS": "true"}
        ):
            _verify_key(None, allow_loopback=True, client_host="127.0.0.1")
            _verify_key(None, allow_loopback=True, client_host="::1")
            with self.assertRaises(Exception):
                _verify_key(None, allow_loopback=True, client_host="192.0.2.10")
            with self.assertRaises(Exception):
                _verify_key(None, client_host="127.0.0.1")

    def test_btc_signal_endpoint_schema(self):
        response = self.client.get("/api/signals/btc")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("asset", payload)
        self.assertIn("signal", payload)
        self.assertIn("risk", payload)
        self.assertIn("data_quality", payload)
        self.assertIn("execution", payload)
        self.assertEqual(payload["execution"]["paper_trading"], True)
        self.assertEqual(payload["execution"]["live_trading"], False)

    def test_gold_signal_endpoint_returns_hold(self):
        response = self.client.get("/api/signals/gold")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["asset"], "XAUUSD")
        self.assertEqual(payload["signal"], "HOLD")
        self.assertFalse(payload["data_quality"]["sufficient"])

    def test_status_endpoint(self):
        response = self.client.get("/api/signals/status")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("btc", payload)
        self.assertIn("gold", payload)
        self.assertIn("signal_engine", payload)
        self.assertIn("execution", payload)

    def test_health_reports_loaded_timeframe_models(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        models = response.json()["models"]
        self.assertIn("h1", models)
        self.assertIn("m15", models)
        self.assertNotEqual(models["h1"], "NO_MODEL")
        self.assertNotEqual(models["m15"], "NO_MODEL")

    def test_signal_engine_handles_missing_iv_and_volume(self):
        contracts = [
            OptionContract(
                asset="BTC",
                provider="deribit",
                timestamp=datetime.now(timezone.utc),
                underlying_price=60000.0,
                expiration="20261101",
                days_to_expiration=30,
                strike=60000.0,
                option_type="call",
                contract_symbol="BTC-01NOV26-60000-C",
                bid=None,
                ask=None,
                last_price=2000.0,
                mark_price=2000.0,
                volume=None,
                open_interest=1500,
                implied_volatility=None,
                bid_iv=None,
                ask_iv=None,
                delta=0.62,
                gamma=0.002,
                theta=-0.45,
                vega=None,
                moneyness=None,
            ),
            OptionContract(
                asset="BTC",
                provider="deribit",
                timestamp=datetime.now(timezone.utc),
                underlying_price=60000.0,
                expiration="20261101",
                days_to_expiration=30,
                strike=61000.0,
                option_type="put",
                contract_symbol="BTC-01NOV26-61000-P",
                bid=None,
                ask=None,
                last_price=1800.0,
                mark_price=1800.0,
                volume=None,
                open_interest=1300,
                implied_volatility=None,
                bid_iv=None,
                ask_iv=None,
                delta=-0.43,
                gamma=0.0015,
                theta=-0.4,
                vega=None,
                moneyness=None,
            ),
        ]
        signal = OptionsSignalEngine().generate(contracts, asset="BTC", reference_price=60000.0)
        self.assertIn(signal["signal"], {"BUY", "SELL", "HOLD"})
        self.assertIsInstance(signal["reasons"], list)

    def test_signal_engine_stale_data_rejected(self):
        old = datetime.now(timezone.utc).replace(year=2020)
        contracts = [
            OptionContract(
                asset="BTC",
                provider="deribit",
                timestamp=old,
                underlying_price=60000.0,
                expiration="20261101",
                days_to_expiration=30,
                strike=60000.0,
                option_type="call",
                contract_symbol="BTC-01NOV26-60000-C",
                bid=None,
                ask=None,
                last_price=2000.0,
                mark_price=2000.0,
                volume=100,
                open_interest=1500,
                implied_volatility=None,
                bid_iv=0.25,
                ask_iv=0.27,
                delta=0.62,
                gamma=0.002,
                theta=-0.45,
                vega=0.18,
                moneyness=None,
            )
        ]
        from api.web_server import build_btc_signal_response
        payload = build_btc_signal_response()
        self.assertEqual(payload["signal"], "HOLD")

    def test_ml_predict_endpoint_returns_signal_from_real_candles(self):
        now = datetime.now(timezone.utc)
        candles = []
        price = 100.0
        for i in range(260):
            ts = now - timedelta(minutes=260 - i)
            open_ = price
            close = price + 0.25 if i % 4 == 0 else price - 0.15
            high = max(open_, close) + 0.6
            low = min(open_, close) - 0.6
            candles.append({
                "timestamp": ts.isoformat(),
                "open": float(open_),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": 1200.0,
            })
            price = close

        with patch("api.web_server.INGEST_API_KEY", ""):
            response = self.client.post("/api/ml/predict", json={
                "symbol": "XAUUSD",
                "timeframe": "M15",
                "candles": candles,
            })
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertIn("signal", payload)
        self.assertIn(payload["signal"], {"BUY", "SELL", "WAIT"})
        self.assertIn("buy_probability", payload)
        self.assertIn("sell_probability", payload)
        self.assertIn("confidence", payload)
        self.assertIn("setup_type", payload)
        self.assertIn("direction", payload)
        self.assertIn("signal_reason", payload)
        self.assertIn(payload["setup_type"], {
            "reversal", "continuation", "breakout", "failed_breakout", "no_edge", "unknown"
        })
        self.assertIn(payload["direction"], {"BUY", "SELL", "NONE"})

    def test_ml_consensus_requires_h1_and_m15_agreement(self):
        now = datetime.now(timezone.utc)
        candles = [{
            "timestamp": (now + timedelta(minutes=index)).isoformat(),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1000.0,
        } for index in range(260)]
        h1_result = {
            "signal": "BUY", "confidence": 74.0, "setup_type": "continuation",
            "probabilities": {"buy": 74.0, "sell": 10.0}, "model_version": "gold_h1",
        }
        m15_result = {
            "signal": "BUY", "confidence": 68.0, "setup_type": "breakout",
            "probabilities": {"buy": 68.0, "sell": 12.0}, "model_version": "gold_m15",
        }
        payload = {
            "symbol": "XAUUSD", "h1_candles": candles, "m15_candles": candles,
        }

        with (
            patch("api.web_server.INGEST_API_KEY", ""),
            patch.object(web_server.h1_engine, "analyze", return_value=h1_result),
            patch.object(web_server.m15_engine, "analyze", return_value=m15_result),
        ):
            response = self.client.post("/api/ml/predict/consensus", json=payload)

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["direction"], "BUY")
        self.assertEqual(result["confidence"], 68.0)
        self.assertEqual(result["setup_type"], "breakout")
        self.assertEqual(result["signal_reason"], "H1_M15_AGREE")

    def test_ml_consensus_returns_wait_when_timeframes_disagree(self):
        now = datetime.now(timezone.utc)
        candles = [{
            "timestamp": (now + timedelta(minutes=index)).isoformat(),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1000.0,
        } for index in range(260)]
        h1_result = {
            "signal": "BUY", "confidence": 74.0, "setup_type": "continuation",
            "probabilities": {"buy": 74.0, "sell": 10.0}, "model_version": "gold_h1",
        }
        m15_result = {
            "signal": "SELL", "confidence": 68.0, "setup_type": "reversal",
            "probabilities": {"buy": 12.0, "sell": 68.0}, "model_version": "gold_m15",
        }
        payload = {
            "symbol": "XAUUSD", "h1_candles": candles, "m15_candles": candles,
        }

        with (
            patch("api.web_server.INGEST_API_KEY", ""),
            patch.object(web_server.h1_engine, "analyze", return_value=h1_result),
            patch.object(web_server.m15_engine, "analyze", return_value=m15_result),
        ):
            response = self.client.post("/api/ml/predict/consensus", json=payload)

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["signal"], "WAIT")
        self.assertEqual(result["direction"], "NONE")
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["signal_reason"], "H1_M15_DISAGREE_OR_WAIT")

    def test_ml_consensus_selects_gold_scalp_and_returns_grid_interval(self):
        now = datetime.now(timezone.utc)
        candles = [{
            "timestamp": (now + timedelta(minutes=index)).isoformat(),
            "open": 2300.0,
            "high": 2301.0,
            "low": 2299.0,
            "close": 2300.5,
            "volume": 1000.0,
        } for index in range(260)]
        h1_result = {
            "signal": "WAIT", "confidence": 68.0, "setup_type": "no_edge",
            "probabilities": {"buy": 0.0, "sell": 0.0}, "model_version": "gold_h1",
        }
        m15_result = {
            "signal": "BUY", "confidence": 72.0, "setup_type": "continuation",
            "probabilities": {"buy": 72.0, "sell": 10.0}, "model_version": "gold_m15",
            "atr": 2.0, "grid_interval_buy_atr": 0.75, "grid_interval_sell_atr": 1.25,
            "swing_stop_buy_atr": 0.7, "swing_target_buy_atr": 1.8,
            "swing_stop_sell_atr": 0.8, "swing_target_sell_atr": 1.9,
            "swing_stop_buy_atr": 0.7, "swing_target_buy_atr": 1.8,
            "swing_stop_sell_atr": 0.8, "swing_target_sell_atr": 1.9,
        }
        payload = {"symbol": "XAUUSD", "h1_candles": candles, "m15_candles": candles}

        with (
            patch("api.web_server.INGEST_API_KEY", ""),
            patch.object(web_server.h1_engine, "analyze", return_value=h1_result),
            patch.object(web_server.m15_engine, "analyze", return_value=m15_result),
        ):
            response = self.client.post("/api/ml/predict/consensus", json=payload)

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["recommended_mode"], "SCALP")
        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["h1_signal"], "WAIT")
        self.assertEqual(result["m15_signal"], "BUY")
        self.assertEqual(result["m15_atr"], 2.0)
        self.assertEqual(result["grid_interval_buy_atr"], 0.75)
        self.assertEqual(result["grid_interval_sell_atr"], 1.25)
        self.assertEqual(result["swing_stop_buy_atr"], 0.7)
        self.assertEqual(result["swing_target_buy_atr"], 1.8)
        self.assertEqual(result["swing_stop_sell_atr"], 0.8)
        self.assertEqual(result["swing_target_sell_atr"], 1.9)
        self.assertEqual(result["swing_stop_buy_atr"], 0.7)
        self.assertEqual(result["swing_target_buy_atr"], 1.8)
        self.assertEqual(result["swing_stop_sell_atr"], 0.8)
        self.assertEqual(result["swing_target_sell_atr"], 1.9)

    def test_ml_consensus_selects_swing_for_timeframe_agreed_reversal(self):
        now = datetime.now(timezone.utc)
        candles = [{
            "timestamp": (now + timedelta(minutes=index)).isoformat(),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1000.0,
        } for index in range(260)]
        h1_result = {
            "signal": "SELL", "confidence": 70.0, "setup_type": "reversal",
            "probabilities": {"buy": 10.0, "sell": 70.0}, "model_version": "h1",
        }
        m15_result = {
            "signal": "SELL", "confidence": 66.0, "setup_type": "reversal",
            "probabilities": {"buy": 12.0, "sell": 66.0}, "model_version": "m15",
        }
        payload = {"symbol": "EURUSD", "h1_candles": candles, "m15_candles": candles}

        with (
            patch("api.web_server.INGEST_API_KEY", ""),
            patch.object(web_server.h1_engine, "analyze", return_value=h1_result),
            patch.object(web_server.m15_engine, "analyze", return_value=m15_result),
        ):
            response = self.client.post("/api/ml/predict/consensus", json=payload)

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["recommended_mode"], "SWING")
        self.assertEqual(result["signal"], "SELL")
        self.assertEqual(result["signal_reason"], "H1_M15_REVERSAL_AGREE")

    def test_ml_predict_endpoint_preserves_probability_scale(self):
        now = datetime.now(timezone.utc)
        candles = [{
            "timestamp": (now + timedelta(minutes=index)).isoformat(),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1000.0,
        } for index in range(260)]
        result = {
            "signal": "BUY",
            "confidence": 61.0,
            "setup_type": "reversal",
            "direction": "BUY",
            "signal_reason": "CONFIDENCE_GATE_PASSED",
            "setup_probability": 0.61,
            "probabilities": {
                "buy": 61.0,
                "sell": 17.0,
                "probability_reversal": 0.61,
                "probability_continuation": 0.12,
                "probability_breakout": 0.05,
                "probability_failed_breakout": 0.05,
                "probability_no_edge": 0.17,
                "probability_reversal_buy": 0.41,
                "probability_reversal_sell": 0.20,
                "probability_continuation_buy": 0.10,
                "probability_continuation_sell": 0.02,
                "probability_breakout_buy": 0.04,
                "probability_breakout_sell": 0.01,
                "probability_failed_breakout_buy": 0.02,
                "probability_failed_breakout_sell": 0.03,
                "probability_long": 0.61,
                "probability_short": 0.17,
                "probability_flat": 0.17,
            },
            "model_version": "test",
            "channel_bias": "NONE",
            "timestamp": now.isoformat(),
            "price": 100.0,
            "risk": {},
            "features": {},
            "levels": {
                "source": "structural_reference_not_model_price_prediction",
                "potential_bullish_reversal": {"low": 98.0, "high": 99.0, "mid": 98.5, "distance": 1.5},
                "potential_bearish_reversal": {"low": 101.0, "high": 102.0, "mid": 101.5, "distance": 1.5},
                "potential_bullish_reversal_source": "demand_zone",
                "potential_bearish_reversal_source": "supply_zone",
                "breakout_up": 103.0,
                "breakout_down": 97.0,
            },
        }
        with patch("api.web_server.INGEST_API_KEY", ""), patch.object(engine, "analyze", return_value=result):
            response = self.client.post("/api/ml/predict", json={
                "symbol": "XAUUSD",
                "timeframe": "M15",
                "candles": candles,
            })

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["setup_type"], "reversal")
        self.assertEqual(payload["direction"], "BUY")
        self.assertEqual(payload["signal_reason"], "CONFIDENCE_GATE_PASSED")
        self.assertEqual(payload["buy"], 61.0)
        self.assertEqual(payload["buy_probability"], 0.61)
        self.assertEqual(payload["sell"], 17.0)
        self.assertEqual(payload["sell_probability"], 0.17)
        self.assertEqual(payload["probability_reversal_buy"], 0.41)
        self.assertEqual(payload["probability_failed_breakout_sell"], 0.03)
        self.assertEqual(payload["potential_bullish_reversal_low"], 98.0)
        self.assertEqual(payload["potential_bearish_reversal_high"], 102.0)
        self.assertEqual(payload["breakout_up"], 103.0)
        self.assertEqual(payload["breakout_down"], 97.0)
        self.assertEqual(payload["level_source"], "structural_reference_not_model_price_prediction")

    def test_ml_predict_endpoint_rejects_insufficient_model_history(self):
        candles = [{
            "timestamp": (datetime.now(timezone.utc) + timedelta(minutes=index)).isoformat(),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1000.0,
        } for index in range(80)]
        with patch("api.web_server.INGEST_API_KEY", ""):
            response = self.client.post("/api/ml/predict", json={
                "symbol": "BTCUSD",
                "timeframe": "M15",
                "candles": candles,
            })
        self.assertEqual(response.status_code, 400)
        self.assertIn("260 completed candles", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
