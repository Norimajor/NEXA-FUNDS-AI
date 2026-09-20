import unittest
from pathlib import Path

import pandas as pd

from backend.engine.backtest_engine import BacktestEngine
from backend.engine.mtf_data_engine import MTFDataEngine
from backend.engine.strategy_analysis.analysis_service import StrategyAnalysisService
from backend.engine.strategy_interpreter.parser import StrategyParser


class TestBacktestAudit(unittest.TestCase):
    DATA_PATH = Path("backend/data/XAUUSD_M15.csv")

    def test_parser_makes_defaults_explicit(self):
        strategy = StrategyParser().parse("Buy XAUUSD on M15 when RSI is below 70.")
        condition = strategy.entry_conditions[0]
        self.assertEqual(strategy.symbol, "XAUUSD")
        self.assertEqual(strategy.timeframe, "M15")
        self.assertEqual(strategy.direction, "long")
        self.assertEqual(condition.indicator, "RSI")
        self.assertEqual(condition.period, 14)
        self.assertEqual(condition.operator, "<")
        self.assertEqual(condition.entry_semantics, "condition")
        self.assertEqual(strategy.max_simultaneous_positions, 1)
        self.assertFalse(strategy.pyramiding)
        self.assertFalse(strategy.stop_loss["specified"])

    def test_parser_distinguishes_crossing(self):
        strategy = StrategyParser().parse("Buy XAUUSD on M15 when RSI crosses below 30.")
        condition = strategy.entry_conditions[0]
        self.assertEqual(condition.operator, "<")
        self.assertEqual(condition.entry_semantics, "cross")

    def test_real_data_quality_is_strict(self):
        report = MTFDataEngine("backend/data").validate_file("XAUUSD", "M15")
        self.assertEqual(report["status"], "valid")
        self.assertEqual(report["invalid_candles"], 0)
        self.assertEqual(report["duplicate_timestamps"], 0)
        self.assertGreater(report["candles"], 100000)

    def test_control_trade_reconciles_from_real_candle_slice(self):
        frame = pd.read_csv(self.DATA_PATH).iloc[:2].copy()
        frame["signal"] = ["BUY", "NO_SIGNAL"]
        entry = float(frame.iloc[0]["close"])
        exit_price = float(frame.iloc[1]["close"])
        stop_distance = 1000.0
        engine = BacktestEngine(initial_balance=10000.0, risk_percent=1.0)
        result = engine.run(frame, stop_distance=stop_distance, symbol="XAUUSD", timeframe="M15")
        trade = result["trades"].iloc[0]
        expected_size = 100.0 / stop_distance
        expected_pnl = (exit_price - entry) * expected_size
        self.assertAlmostEqual(float(trade["size"]), expected_size)
        self.assertAlmostEqual(float(trade["pnl"]), expected_pnl)
        self.assertAlmostEqual(float(trade["pnl"]), result["final_balance"] - result["initial_balance"])
        self.assertEqual(trade["exit_reason"], "END_OF_DATA")
        self.assertEqual(trade["symbol"], "XAUUSD")
        self.assertEqual(trade["timeframe"], "M15")
        self.assertEqual(int(trade["trade_id"]), 1)

    def test_persistent_condition_does_not_reenter_while_true(self):
        service = StrategyAnalysisService(data_directory="backend/data")
        strategy = StrategyParser().parse("Buy XAUUSD on M15 when RSI is below 70.")
        data = MTFDataEngine("backend/data").load("XAUUSD", "M15").iloc[:500].copy()
        indicators = service.indicator_engine.calculate_indicators(data.copy())
        signals = service._generate_signal_dataframe(strategy, {"M15": indicators}, data)
        signal_times = signals.loc[signals["signal"] == "BUY", "timestamp"]
        self.assertLessEqual(len(signal_times), int((indicators["RSI_14"] < 70).sum()))
        self.assertEqual(signal_times.duplicated().sum(), 0)

    def test_report_reconciles_trade_ledger(self):
        result = StrategyAnalysisService().analyze("Buy XAUUSD on M15 when RSI is below 70.")
        trades = pd.DataFrame(result["trades"])
        self.assertEqual(len(trades), result["backtest"]["trades"])
        self.assertEqual(int((trades["result"] == "WIN").sum()) + int((trades["result"] == "LOSS").sum()), len(trades))
        self.assertAlmostEqual(float(trades["pnl"].sum()), float(result["backtest"]["net_profit"]), places=6)
        self.assertAlmostEqual(float(trades["gross_pnl"].sum()), float(trades["pnl"].sum()), places=6)
        self.assertEqual(trades["trade_id"].nunique(), len(trades))
        self.assertTrue((trades["exit_time"] >= trades["entry_time"]).all())


if __name__ == "__main__":
    unittest.main()
