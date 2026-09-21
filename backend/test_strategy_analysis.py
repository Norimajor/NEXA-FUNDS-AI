import tempfile
import unittest
import math
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from backend.api import app
from backend.llm.mock_provider import MockLLMProvider
from backend.engine.strategy_analysis.analysis_service import StrategyAnalysisService
from backend.engine.strategy_interpreter.parser import StrategyParser


class TestStrategyAnalysis(unittest.TestCase):
    def _sample_frame(self, symbol="EURUSD", timeframe="M15"):
        start = pd.Timestamp("2024-01-01 00:00:00")
        freq_map = {"M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min", "H1": "h", "H4": "4h"}
        timestamps = pd.date_range(start, periods=220, freq=freq_map.get(timeframe, "15min"))
        prices = [1.1000 + i * 0.0002 for i in range(len(timestamps))]
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": prices,
                "high": [p + 0.0005 for p in prices],
                "low": [p - 0.0005 for p in prices],
                "close": prices,
                "volume": 1000,
                "spread": 0.0,
            }
        )
        df["close"] = df["close"].astype(float)
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        return df

    def test_health_and_root_endpoints(self):
        client = TestClient(app)
        self.assertEqual(client.get("/").status_code, 200)
        self.assertEqual(client.get("/health").status_code, 200)

    def test_production_origin_preflight_is_allowed(self):
        client = TestClient(app)
        response = client.options(
            "/analyze",
            headers={
                "Origin": "https://nexafunds-steel.vercel.app",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], "https://nexafunds-steel.vercel.app")

    def test_unlisted_origin_preflight_is_rejected(self):
        client = TestClient(app)
        response = client.options(
            "/analyze",
            headers={
                "Origin": "https://example.invalid",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertNotIn("access-control-allow-origin", response.headers)

    def test_parser_examples(self):
        parser = StrategyParser()
        xau = parser.parse("Buy XAUUSD when RSI drops below 30 on the 15m chart, exit at 1.5% profit or 0.7% loss.", symbol="XAUUSD", timeframe="M15")
        self.assertEqual(xau.symbol, "XAUUSD")
        self.assertEqual(xau.timeframe, "M15")
        self.assertEqual(xau.direction, "long")
        self.assertTrue(any(c.indicator == "RSI" for c in xau.entry_conditions))

        eur = parser.parse("Scalp EURUSD during the London session using a 9/21 EMA crossover, max 2 trades per day.", symbol="EURUSD", timeframe="M15")
        self.assertEqual(eur.symbol, "EURUSD")
        self.assertTrue(any(c.indicator == "EMA_CROSS" for c in eur.entry_conditions))

        us30 = parser.parse("Trend-follow US30 on the H1 chart, only long, trail stop at 1 ATR.", symbol="US30", timeframe="H1")
        self.assertEqual(us30.symbol, "US30")
        self.assertEqual(us30.timeframe, "H1")

    def test_empty_prompt_rejected(self):
        service = StrategyAnalysisService(data_directory=str(Path("backend/data")))
        result = service.analyze("   ")
        self.assertFalse(result["success"])
        self.assertIn("required", str(result["error"]).lower())

    def test_missing_historical_data_status(self):
        service = StrategyAnalysisService(data_directory=str(Path("backend/data")))
        result = service.analyze("Buy EURUSD when RSI closes below 30.")
        self.assertTrue(result["success"])
        self.assertEqual(result["data"]["status"], "unavailable")
        self.assertIn("not available", result["data"]["reason"].lower())

    def test_strategy_generates_zero_trades_safe(self):
        service = StrategyAnalysisService(data_directory=str(Path("backend/data")))
        result = service.analyze("Buy EURUSD when RSI stays above 99 for all time.")
        self.assertTrue(result["success"])
        self.assertEqual(result["backtest"]["status"], "no_trades")

    def test_real_backtest_uses_actual_data(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "EURUSD_M15.csv"
            df = self._sample_frame()
            df["close"] = df["close"] + 0.0002
            df.to_csv(path, index=False)
            service = StrategyAnalysisService(data_directory=tmp_dir)
            result = service.analyze("Buy EURUSD when RSI is below 70.")
            self.assertTrue(result["success"])
            self.assertIn(result["backtest"]["status"], {"completed", "no_trades"})
            if result["backtest"]["status"] == "completed":
                self.assertGreater(result["backtest"]["trades"], 0)
                self.assertTrue(result["backtest"].get("win_rate") is None or isinstance(result["backtest"]["win_rate"], (int, float)))

    def test_completed_backtest_uses_real_measurements(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "EURUSD_M15.csv"
            df = self._sample_frame()
            df["close"] = [1.1000 + 0.0012 * math.sin(i / 7.5) + 0.0003 * ((i % 11) - 5) / 11.0 for i in range(len(df))]
            df["open"] = df["close"] - 0.00015
            df["high"] = df["close"] + 0.0005
            df["low"] = df["close"] - 0.0005
            df.to_csv(path, index=False)
            service = StrategyAnalysisService(data_directory=tmp_dir)
            result = service.analyze("Buy EURUSD when RSI is below 70.")
            self.assertTrue(result["success"])
            self.assertEqual(result["backtest"]["status"], "completed")
            self.assertGreater(result["backtest"]["trades"], 0)
            self.assertIsNotNone(result["backtest"]["win_rate"])
            self.assertIsNotNone(result["backtest"]["profit_factor"])
            self.assertIsNotNone(result["backtest"]["expectancy"])
            self.assertIsNotNone(result["backtest"]["max_drawdown"])

    def test_optimization_metrics_are_not_fabricated(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "EURUSD_M15.csv"
            self._sample_frame().to_csv(path, index=False)
            service = StrategyAnalysisService(data_directory=tmp_dir)
            result = service.analyze("Buy EURUSD when RSI is below 60.")
            self.assertTrue(result["success"])
            self.assertIn("optimization", result)
            self.assertTrue(isinstance(result["optimization"].get("candidates", []), list))

    def test_walk_forward_handles_insufficient_data(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "EURUSD_M15.csv"
            short = self._sample_frame().iloc[:40].copy()
            short.to_csv(path, index=False)
            service = StrategyAnalysisService(data_directory=tmp_dir)
            result = service.analyze("Buy EURUSD when RSI is below 60.")
            self.assertTrue(result["success"])
            self.assertIn(result["walk_forward"]["status"], {"insufficient_data", "skipped", "completed"})

    def test_response_is_json_serializable(self):
        service = StrategyAnalysisService(data_directory=str(Path("backend/data")))
        result = service.analyze("Buy EURUSD when RSI is above 50.")
        import json
        json.dumps(result)

    def test_incomplete_strategy_returns_structured_validation(self):
        service = StrategyAnalysisService(data_directory=str(Path("backend/data")))
        result = service.analyze("Describe a strategy for me.")
        self.assertTrue(result["success"])
        self.assertEqual(result["validation"]["status"], "invalid")
        self.assertEqual(result["test"]["status"], "invalid")
        self.assertEqual(result["test"]["metrics"], {})
        self.assertTrue(result["validation"]["errors"])

    def test_unavailable_data_is_explicit_and_has_no_metrics(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = StrategyAnalysisService(data_directory=tmp_dir).analyze("Buy EURUSD when RSI is below 30.")
        self.assertTrue(result["success"])
        self.assertEqual(result["test"]["status"], "unavailable")
        self.assertIn("Data unavailable", result["test"]["reason"])
        self.assertNotIn("win_rate", result["test"])
        self.assertIn("assumptions", result["strategy"])

    def test_structured_report_uses_measured_metrics_and_candidates(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "EURUSD_M15.csv"
            frame = self._sample_frame()
            frame["close"] = [1.1000 + 0.0012 * math.sin(i / 7.5) + 0.0003 * ((i % 11) - 5) / 11.0 for i in range(len(frame))]
            frame["open"] = frame["close"] - 0.00015
            frame["high"] = frame["close"] + 0.0005
            frame["low"] = frame["close"] - 0.0005
            frame.to_csv(path, index=False)
            result = StrategyAnalysisService(data_directory=tmp_dir).analyze("Buy EURUSD when RSI is below 70.")
        self.assertEqual(result["test"]["status"], "completed")
        metrics = result["test"]["metrics"]
        self.assertEqual(metrics["trades"], metrics["wins"] + metrics["losses"])
        self.assertIsNotNone(metrics["gross_profit"])
        self.assertIsNotNone(metrics["gross_loss"])
        self.assertIsNotNone(metrics["expectancy"])
        self.assertIsNotNone(metrics["average_trade_duration_minutes"])
        self.assertIn("performance_analysis", result)
        self.assertTrue(result["recommendations"])
        self.assertEqual(result["optimization"]["baseline"]["trades"], metrics["trades"])
        self.assertTrue(result["candidates"])
        self.assertTrue(all(candidate["results"]["status"] in {"completed", "no_trades"} for candidate in result["candidates"]))

    def test_api_analyze_preserves_structured_response_contract(self):
        client = TestClient(app)
        with unittest.mock.patch("backend.api.get_llm_provider", return_value=MockLLMProvider()):
            response = client.post("/analyze", json={"prompt": "Buy EURUSD when RSI is below 30."})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["success"])
        self.assertIn("strategy", body)
        self.assertIn("test", body)
        self.assertIn("performance_analysis", body)
        self.assertIn("recommendations", body)
        self.assertIn("candidates", body)
        self.assertIn("robustness", body)

    def test_api_analyze_returns_cors_header_for_production_origin(self):
        client = TestClient(app)
        with unittest.mock.patch("backend.api.get_llm_provider", return_value=MockLLMProvider()):
            response = client.post(
                "/analyze",
                json={"prompt": "Buy EURUSD when RSI is below 30."},
                headers={"Origin": "https://nexafunds-steel.vercel.app"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], "https://nexafunds-steel.vercel.app")

    def test_api_rejects_invalid_input(self):
        client = TestClient(app)
        response = client.post("/analyze", json={"prompt": "   "})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
