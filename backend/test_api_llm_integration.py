import os
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from backend.api import app
from backend.engine.strategy_interpreter.models import StrategyCondition, StrategyDefinition
from backend.llm.mock_provider import MockLLMProvider
from backend.llm.provider import LLMProviderError


class TestAnalyzeLLMIntegration(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.prompt = "Buy XAUUSD when RSI drops below 30 on M15. Take profit 1.5% and stop loss 0.7%."

    def _completed_stub(self, strategy, data_info):
        return {
            "backtest": {"status": "completed", "candles": data_info["candles"], "trades": 1},
            "trades": [],
            "test": {"status": "completed", "metrics": {"trades": 1}},
            "performance_analysis": {"strengths": [], "weaknesses": [], "limitations": []},
            "summary": "Deterministic test result.",
        }

    def test_mock_provider_reaches_deterministic_pipeline_with_percentage_exits(self):
        captured = {}

        def evaluate(service, strategy, data_info):
            captured["strategy"] = strategy
            return self._completed_stub(strategy, data_info)

        with patch.dict(os.environ, {"LLM_PROVIDER": "mock"}, clear=False):
            with patch("backend.engine.strategy_analysis.analysis_service.StrategyAnalysisService._evaluate_strategy", evaluate):
                response = self.client.post("/analyze", json={"prompt": self.prompt})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        strategy = captured["strategy"]
        self.assertEqual(strategy.symbol, "XAUUSD")
        self.assertEqual(strategy.timeframe, "M15")
        self.assertEqual(strategy.direction, "long")
        self.assertEqual(strategy.entry_conditions[0].operator, "<")
        self.assertEqual(strategy.entry_conditions[0].value, 30.0)
        self.assertEqual(strategy.take_profit, {"type": "percentage", "value": 1.5, "specified": True})
        self.assertEqual(strategy.stop_loss, {"type": "percentage", "value": 0.7, "specified": True})
        self.assertEqual(body["validation"]["status"], "valid")
        self.assertEqual(body["strategy"]["take_profit"], strategy.take_profit)
        self.assertEqual(body["strategy"]["stop_loss"], strategy.stop_loss)

    def test_invalid_provider_strategy_is_rejected_before_backtest(self):
        invalid = MockLLMProvider().interpret_strategy(self.prompt)
        invalid.entry_conditions[0].operator = "invalid"
        invalid.risk_percent = -1
        provider = Mock()
        provider.interpret_strategy.return_value = invalid
        evaluate = Mock()

        with patch("backend.api.get_llm_provider", return_value=provider):
            with patch("backend.engine.strategy_analysis.analysis_service.StrategyAnalysisService._evaluate_strategy", evaluate):
                response = self.client.post("/analyze", json={"prompt": self.prompt})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["validation"]["status"], "invalid")
        evaluate.assert_not_called()

    def test_provider_error_returns_safe_interpretation_error(self):
        provider = Mock()
        provider.interpret_strategy.side_effect = LLMProviderError("internal provider details")
        service = Mock()

        with patch("backend.api.get_llm_provider", return_value=provider):
            with patch("backend.api.StrategyAnalysisService", return_value=service):
                response = self.client.post("/analyze", json={"prompt": self.prompt})

        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json(), {"detail": "Strategy interpretation failed. Please try again."})
        service.analyze.assert_not_called()

    def test_mock_provider_does_not_construct_openai_provider(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "mock"}, clear=False):
            with patch("backend.llm.openai_provider.OpenAIProvider") as openai_provider:
                response = self.client.post("/analyze", json={"prompt": self.prompt})

        self.assertEqual(response.status_code, 200)
        openai_provider.assert_not_called()

    def test_chat_skips_web_search_for_normal_greeting(self):
        with patch("backend.api.web_search", create=True) as mocked_search:
            response = self.client.post("/chat", json={"message": "Hello, how are you?"})

        self.assertEqual(response.status_code, 200)
        mocked_search.assert_not_called()
        self.assertIn("message", response.json())

    def test_chat_uses_web_search_for_current_fed_rate(self):
        fake_results = [{
            "title": "Federal Reserve Policy",
            "url": "https://example.com/fed-rate",
            "snippet": "The Federal Reserve kept rates unchanged.",
            "source": "Federal Reserve",
            "published_date": "2026-09-25",
        }]

        with patch("backend.api.web_search", create=True, return_value={
            "query": "current Federal Funds Rate",
            "results": fake_results,
            "provider": "tavily",
        }) as mocked_search:
            response = self.client.post(
                "/chat",
                json={"message": "What is the current Federal Funds Rate?"},
            )

        self.assertEqual(response.status_code, 200)
        mocked_search.assert_called_once()
        body = response.json()
        self.assertEqual(body["intent"], "web_research")
        self.assertIn("Federal Reserve", body["research"]["sources"][0]["source"])

    def test_chat_avoids_web_search_for_backtest_requests(self):
        with patch("backend.api.web_search", create=True) as mocked_search:
            response = self.client.post(
                "/chat",
                json={"message": "Backtest RSI below 30 on XAUUSD M15."},
            )

        self.assertEqual(response.status_code, 200)
        mocked_search.assert_not_called()
        self.assertIn("strategy", response.json().get("intent", ""))


if __name__ == "__main__":
    unittest.main()
