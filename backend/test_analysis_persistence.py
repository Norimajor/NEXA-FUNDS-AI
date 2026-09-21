import tempfile
import unittest
import os
from unittest.mock import Mock, patch

from backend.analysis_persistence import AnalysisStore, strategy_fingerprint
from backend.combination_ranking import CombinationRankingService
from fastapi.testclient import TestClient
from backend.api import app


def _record(record_id, threshold, trades=20, status="completed"):
    strategy = {
        "symbol": "EURUSD",
        "timeframe": "M15",
        "direction": "long",
        "entry_conditions": [{"indicator": "RSI", "value": threshold}],
    }
    return {
        "id": record_id,
        "strategy": strategy,
        "strategy_fingerprint": strategy_fingerprint(strategy),
        "backtest": {
            "status": status,
            "trades": trades,
            "net_profit": 12.0,
            "expectancy": 0.6,
            "profit_factor": 1.8,
            "win_rate": 60.0,
            "max_drawdown": 4.0,
        },
        "optimization": {"status": "skipped", "candidates": []},
        "walk_forward": {"status": "completed"},
        "summary": "Measured.",
    }


class TestAnalysisPersistence(unittest.TestCase):
    def test_insert_and_recent_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            store = AnalysisStore(f"{directory}/analysis.sqlite3")
            result = _record(0, 30)
            saved = store.insert("buy EURUSD", result)
            self.assertEqual(saved["prompt"], "buy EURUSD")
            self.assertEqual(saved["strategy_fingerprint"], strategy_fingerprint(result["strategy"]))
            self.assertEqual(store.recent(1)[0]["backtest"]["trades"], 20)

    def test_ranking_uses_only_measured_safeguarded_candidates(self):
        ranked = CombinationRankingService(minimum_trades=10).rank(
            [_record(1, 30), _record(2, 40, trades=2), _record(3, 50, status="not_tested")]
        )
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["analysis_id"], 1)
        self.assertIn("score", ranked[0])

    def test_api_persists_analysis_and_exposes_history(self):
        with tempfile.TemporaryDirectory() as directory:
            strategy = Mock()
            strategy.symbol = "EURUSD"
            strategy.timeframe = "M15"
            strategy.direction = "long"
            result = _record(0, 30)
            result["success"] = True
            with patch.dict(os.environ, {"NEXA_FUNDS_ANALYSIS_DB": f"{directory}/api.sqlite3"}):
                with patch("backend.api.get_llm_provider") as provider:
                    provider.return_value.interpret_strategy.return_value = strategy
                    with patch("backend.api.StrategyAnalysisService") as service:
                        service.return_value.analyze.return_value = result
                        client = TestClient(app)
                        response = client.post("/analyze", json={"prompt": "buy EURUSD"})
                        self.assertEqual(response.status_code, 200)
                        history = client.get("/history")
                        self.assertEqual(history.status_code, 200)
                        self.assertEqual(len(history.json()["history"]), 1)
                        recommendations = client.post("/recommendations", json={})
                        self.assertEqual(recommendations.status_code, 200)
                        self.assertEqual(len(recommendations.json()["recommendations"]), 1)


if __name__ == "__main__":
    unittest.main()
