import os
import tempfile
import unittest
from unittest.mock import Mock

from fastapi.testclient import TestClient

from backend.api import app
from backend.conversation_assistant import ConversationStore, ConversationalAssistant


class TestConversationAssistant(unittest.TestCase):
    def test_greeting_onboarding_uses_greeting_word_and_capabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            assistant = ConversationalAssistant(store=ConversationStore(os.path.join(directory, "chat.sqlite3")))
            greeting = assistant.respond("hey", "u1")
            introduced = assistant.respond("you're speaking with Obed", "u1")
            capabilities = assistant.respond("yes", "u1")
            self.assertEqual(greeting["message"], "Hey, I am NEXAFUNDS AI. Who am I speaking with please?")
            self.assertEqual(introduced["message"], "Welcome Obed! Wanna know how I can help you?")
            self.assertEqual(capabilities["intent"], "capabilities")
            self.assertIn("backtest", capabilities["message"])

    def test_greeting_extracts_and_remembers_name(self):
        with tempfile.TemporaryDirectory() as directory:
            assistant = ConversationalAssistant(store=ConversationStore(os.path.join(directory, "chat.sqlite3")))
            first = assistant.respond("how are you, my name is Obed", "u1")
            second = assistant.respond("hello again", "u1")
            self.assertEqual(first["intent"], "greeting")
            self.assertIn("Obed", second["message"])
            self.assertEqual(assistant.store.profile("u1")["name"], "Obed")

    def test_combination_search_returns_structured_clarification(self):
        with tempfile.TemporaryDirectory() as directory:
            assistant = ConversationalAssistant(store=ConversationStore(os.path.join(directory, "chat.sqlite3")))
            result = assistant.respond("I want the best profitable indicator combinations to build an EA", "u1")
            self.assertEqual(result["status"], "needs_clarification")
            self.assertTrue({item["field"] for item in result["questions"]} >=
                            {"symbol", "timeframe", "risk_percent", "platform", "data"})

    def test_completed_search_only_returns_measured_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            data = os.path.join(directory, "XAUUSD_M15.csv")
            with open(data, "w", encoding="utf-8") as output:
                output.write("timestamp,open,high,low,close,volume\n")
            service = Mock()
            service.analyze.return_value = {
                "backtest": {"status": "completed", "trades": 12, "net_profit": 20},
                "walk_forward": {"status": "completed"},
                "strategy": {"symbol": "XAUUSD", "timeframe": "M15"},
            }
            old = os.environ.get("NEXA_FUNDS_DATA_DIR")
            os.environ["NEXA_FUNDS_DATA_DIR"] = directory
            try:
                result = ConversationalAssistant(
                    store=ConversationStore(os.path.join(directory, "chat.sqlite3")),
                    analysis_service=service,
                ).respond("find indicator combinations for XAUUSD M15 risk 1% MT5", "u1")
            finally:
                if old is None:
                    os.environ.pop("NEXA_FUNDS_DATA_DIR", None)
                else:
                    os.environ["NEXA_FUNDS_DATA_DIR"] = old
            self.assertEqual(result["status"], "completed")
            self.assertEqual(len(result["candidates"]), 3)
            self.assertTrue(all(item["walk_forward"]["status"] == "completed" for item in result["candidates"]))

    def test_all_downloaded_symbols_runs_each_valid_dataset_and_reports_skips(self):
        with tempfile.TemporaryDirectory() as directory:
            for filename in ("XAUUSD_M15.csv", "EURUSD_H1.csv"):
                with open(os.path.join(directory, filename), "w", encoding="utf-8") as output:
                    output.write("timestamp,open,high,low,close,volume\n")
                    output.write("2024-01-01,1,2,0.5,1.5,10\n")
            with open(os.path.join(directory, "bad_M15.csv"), "w", encoding="utf-8") as output:
                output.write("timestamp,open,high,low,close\n")
            with open(os.path.join(directory, "GBPUSD_D1.csv"), "w", encoding="utf-8") as output:
                output.write("timestamp,open,high,low,close,volume\n")
            service = Mock()
            service.analyze.side_effect = lambda prompt: {
                "backtest": {"status": "completed", "trades": 12, "net_profit": 20,
                             "expectancy": 1, "profit_factor": 2, "win_rate": 55,
                             "max_drawdown": 2},
                "walk_forward": {"status": "completed"},
                "strategy": {"symbol": prompt.split()[1], "timeframe": prompt.split()[3],
                             "direction": "BUY"},
            }
            old = os.environ.get("NEXA_FUNDS_DATA_DIR")
            os.environ["NEXA_FUNDS_DATA_DIR"] = directory
            try:
                result = ConversationalAssistant(
                    store=ConversationStore(os.path.join(directory, "chat.sqlite3")),
                    analysis_service=service,
                ).respond("find the best indicator combinations across all downloaded symbols risk 1% MT5", "u1")
            finally:
                if old is None:
                    os.environ.pop("NEXA_FUNDS_DATA_DIR", None)
                else:
                    os.environ["NEXA_FUNDS_DATA_DIR"] = old
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["limits"]["executed"], 2)
            self.assertEqual(len(result["datasets"]), 2)
            self.assertEqual(len(result["datasets"][0]["candidates"]), 3)
            self.assertEqual(len(result["skipped_datasets"]), 2)
            self.assertTrue(any("Missing required" in item["reason"] for item in result["skipped_datasets"]))
            self.assertTrue(any("Unsupported timeframe" in item["reason"] for item in result["skipped_datasets"]))

    def test_follow_up_requests_resolve_previous_strategy(self):
        with tempfile.TemporaryDirectory() as directory:
            service = Mock()
            service.analyze.side_effect = [
                {
                    "success": True,
                    "strategy": {"symbol": "XAUUSD", "timeframe": "M15", "direction": "LONG"},
                    "summary": "Created and validated the strategy.",
                    "backtest": {"status": "completed", "trades": 12, "net_return": 3.2, "win_rate": 58, "max_drawdown": 6},
                    "validation": {"valid": True},
                },
                {
                    "success": True,
                    "strategy": {"symbol": "XAUUSD", "timeframe": "M15", "direction": "LONG"},
                    "summary": "Backtest completed.",
                    "backtest": {"status": "completed", "trades": 12, "net_return": 3.2, "win_rate": 58, "max_drawdown": 6},
                    "validation": {"valid": True},
                },
            ]
            assistant = ConversationalAssistant(store=ConversationStore(os.path.join(directory, "follow_up.sqlite3")), analysis_service=service)
            assistant.respond("Build an XAUUSD M15 strategy that goes long when RSI is below 30.", "u1")
            result = assistant.respond("Test it.", "u1")
            self.assertEqual(result["intent"], "backtest_request")
            self.assertEqual(service.analyze.call_count, 2)
            self.assertIn("XAUUSD", result["message"])

    def test_strategy_modification_and_backtest_analysis_use_recent_context(self):
        with tempfile.TemporaryDirectory() as directory:
            service = Mock()
            service.analyze.side_effect = [
                {
                    "success": True,
                    "strategy": {"symbol": "XAUUSD", "timeframe": "M15", "direction": "LONG"},
                    "summary": "Created and validated the strategy.",
                    "backtest": {"status": "completed", "trades": 8, "net_return": -2.1, "win_rate": 38, "max_drawdown": 14},
                    "validation": {"valid": True},
                },
                {
                    "success": True,
                    "strategy": {"symbol": "XAUUSD", "timeframe": "M15", "direction": "LONG"},
                    "summary": "Backtest completed.",
                    "backtest": {"status": "completed", "trades": 8, "net_return": -2.1, "win_rate": 38, "max_drawdown": 14},
                    "validation": {"valid": True},
                },
            ]
            assistant = ConversationalAssistant(store=ConversationStore(os.path.join(directory, "context.sqlite3")), analysis_service=service)
            assistant.respond("Build an XAUUSD M15 strategy that goes long when RSI is below 30.", "u1")
            update = assistant.respond("Change the EMA to 50.", "u1")
            self.assertEqual(update["intent"], "strategy_modification")
            self.assertIn("EMA 50", update["message"])
            assistant.respond("Test that.", "u1")
            analysis = assistant.respond("Why did it perform badly?", "u1")
            self.assertEqual(analysis["intent"], "backtest_analysis")
            self.assertIn("max drawdown", analysis["message"].lower())


if __name__ == "__main__":
    unittest.main()
