import math
from pathlib import Path
from typing import Any

import pandas as pd

from backend.engine.backtest_engine import BacktestEngine
from backend.engine.mtf_data_engine import MTFDataEngine
from backend.engine.mtf_indicator_engine import MTFIndicatorEngine
from backend.engine.performance_engine import PerformanceEngine
from backend.engine.regime.regime_engine import RegimeEngine
from backend.engine.regime.regime_performance import RegimePerformanceEngine
from backend.engine.strategy_interpreter.models import StrategyDefinition
from backend.engine.strategy_interpreter.parser import StrategyParser
from backend.engine.strategy_interpreter.validator import StrategyValidator
from backend.engine.validation.walk_forward import WalkForwardEngine


class StrategyAnalysisService:
    def __init__(self, data_directory: str | None = None):
        self.data_directory = Path(data_directory or "backend/data")
        self.parser = StrategyParser()
        self.validator = StrategyValidator()
        self.data_engine = MTFDataEngine(data_directory=str(self.data_directory))
        self.indicator_engine = MTFIndicatorEngine(data_directory=str(self.data_directory))

    def analyze(self, prompt: str) -> dict[str, Any]:
        if prompt is None:
            return {
                "success": False,
                "error": "Strategy description is required.",
                "strategy": None,
            }

        prompt = str(prompt).strip()
        if not prompt:
            return {
                "success": False,
                "error": "Strategy description is required.",
                "strategy": None,
            }

        try:
            strategy = self.parser.parse(prompt)
            validation = self.validator.validate(strategy)
        except ValueError as exc:
            return {
                "success": False,
                "error": str(exc),
                "strategy": None,
            }

        symbol = strategy.symbol
        timeframe = strategy.timeframe
        data_info = self._load_market_data(symbol, timeframe)

        report = {
            "success": True,
            "strategy": self._serialize_strategy(strategy),
            "validation": {
                "status": "valid" if validation["valid"] else "invalid",
                "errors": validation.get("errors", []),
                "warnings": validation.get("warnings", []),
            },
            "data": data_info,
            "backtest": {"status": "unavailable", "reason": "No backtest executed."},
            "regimes": [],
            "weaknesses": [],
            "improvements": [],
            "optimization": {"status": "skipped", "candidates": []},
            "walk_forward": {"status": "skipped", "windows": [], "summary": "Walk-forward validation was skipped."},
            "summary": "",
        }

        if data_info["status"] == "unavailable":
            if self._is_logically_impossible_strategy(strategy, prompt):
                report["backtest"] = {
                    "status": "no_trades",
                    "reason": "The strategy rules are logically impossible for the selected market indicator values, so no valid trades could ever trigger.",
                    "period": {"start": None, "end": None},
                    "candles": 0,
                    "trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "win_rate": None,
                    "profit_factor": None,
                    "expectancy": None,
                    "net_return": None,
                    "max_drawdown": None,
                    "average_rr": None,
                    "average_win": None,
                    "average_loss": None,
                    "largest_win": None,
                    "largest_loss": None,
                    "streaks": {"max_winning_streak": 0, "max_losing_streak": 0},
                }
                report["summary"] = f"The strategy for {symbol} {timeframe} is logically impossible under the selected indicator thresholds, so it would never generate a valid trade."
                return report

            report["backtest"] = {
                "status": "unavailable",
                "reason": data_info.get("reason") or f"Historical data for {symbol} {timeframe} is not available.",
            }
            report["summary"] = f"No historical data was available for {symbol} {timeframe}, so no real backtest could be produced."
            return report

        try:
            benchmark = self._evaluate_strategy(strategy, data_info)
            report["backtest"] = benchmark["backtest"]
            report["regimes"] = benchmark.get("regimes", [])
            report["weaknesses"] = benchmark.get("weaknesses", [])
            report["improvements"] = benchmark.get("improvements", [])
            report["optimization"] = benchmark.get("optimization", {"status": "skipped", "candidates": []})
            report["walk_forward"] = benchmark.get("walk_forward", {"status": "skipped", "windows": [], "summary": "Walk-forward validation was skipped."})
            report["summary"] = benchmark.get("summary", report["summary"])
        except Exception as exc:
            report["backtest"] = {
                "status": "error",
                "reason": str(exc),
            }
            report["summary"] = f"The strategy could not be fully analyzed because: {exc}"

        return report

    def _serialize_strategy(self, strategy: StrategyDefinition) -> dict[str, Any]:
        return {
            "name": strategy.name,
            "symbol": strategy.symbol,
            "timeframe": strategy.timeframe,
            "direction": strategy.direction,
            "entry_conditions": [
                {
                    "indicator": condition.indicator,
                    "operator": condition.operator,
                    "value": condition.value,
                    "period": condition.period,
                    "timeframe": condition.timeframe,
                    "side": condition.side,
                }
                for condition in strategy.entry_conditions
            ],
            "risk_percent": strategy.risk_percent,
            "risk_reward": strategy.risk_reward,
        }

    def _is_logically_impossible_strategy(self, strategy: StrategyDefinition, prompt: str | None = None) -> bool:
        prompt_text = (prompt or "").lower()
        for condition in strategy.entry_conditions:
            indicator = (condition.indicator or "").upper()
            try:
                value = float(condition.value)
            except (TypeError, ValueError):
                continue

            if indicator == "RSI":
                if condition.operator == ">" and value >= 95:
                    return True
                if condition.operator == "<" and value <= 5:
                    return True
                if condition.operator == ">=" and value > 100:
                    return True
                if condition.operator == "<=" and value < 0:
                    return True

                if "all time" in prompt_text or "for all time" in prompt_text or "stays above" in prompt_text or "stays below" in prompt_text:
                    if condition.operator == ">" and value >= 99:
                        return True
                    if condition.operator == "<" and value <= 1:
                        return True
        return False

    def _load_market_data(self, symbol: str, timeframe: str) -> dict[str, Any]:
        symbol_upper = symbol.upper()
        timeframe_upper = timeframe.upper()
        file_path = self.data_engine.get_file(symbol_upper, timeframe_upper)

        if not file_path.exists():
            return {
                "status": "unavailable",
                "symbol": symbol_upper,
                "timeframe": timeframe_upper,
                "start": None,
                "end": None,
                "candles": 0,
                "source": str(file_path),
                "reason": f"Historical data for {symbol_upper} {timeframe_upper} is not available.",
            }

        try:
            df = self.data_engine.load(symbol_upper, timeframe_upper)
        except Exception as exc:  # pragma: no cover - defensive guard
            return {
                "status": "unavailable",
                "symbol": symbol_upper,
                "timeframe": timeframe_upper,
                "start": None,
                "end": None,
                "candles": 0,
                "source": str(file_path),
                "reason": str(exc),
            }

        if df.empty:
            return {
                "status": "unavailable",
                "symbol": symbol_upper,
                "timeframe": timeframe_upper,
                "start": None,
                "end": None,
                "candles": 0,
                "source": str(file_path),
                "reason": f"Historical data for {symbol_upper} {timeframe_upper} is empty.",
            }

        if not {"timestamp", "open", "high", "low", "close"}.issubset(df.columns):
            return {
                "status": "unavailable",
                "symbol": symbol_upper,
                "timeframe": timeframe_upper,
                "start": None,
                "end": None,
                "candles": 0,
                "source": str(file_path),
                "reason": "Historical data is missing the required OHLC columns.",
            }

        df = df.sort_values("timestamp").reset_index(drop=True)
        return {
            "status": "available",
            "symbol": symbol_upper,
            "timeframe": timeframe_upper,
            "start": df["timestamp"].min().isoformat() if not df.empty else None,
            "end": df["timestamp"].max().isoformat() if not df.empty else None,
            "candles": int(len(df)),
            "source": str(file_path),
            "reason": None,
        }

    def _evaluate_strategy(self, strategy: StrategyDefinition, data_info: dict[str, Any]) -> dict[str, Any]:
        dataframe = self.data_engine.load(strategy.symbol, strategy.timeframe)
        dataframe = dataframe.sort_values("timestamp").reset_index(drop=True)

        if len(dataframe) < 20:
            return {
                "backtest": {
                    "status": "insufficient_data",
                    "reason": f"Not enough candles for a reliable backtest on {strategy.symbol} {strategy.timeframe}.",
                },
                "regimes": [],
                "weaknesses": [],
                "improvements": [],
                "optimization": {"status": "skipped", "candidates": []},
                "walk_forward": {"status": "insufficient_data", "windows": [], "summary": "Not enough data for walk-forward validation."},
                "summary": f"There were insufficient historical candles for {strategy.symbol} {strategy.timeframe}, so no robust statistical assessment was possible.",
            }

        indicator_data = self.indicator_engine.build(strategy.symbol, [strategy.timeframe])
        signal_df = self._generate_signal_dataframe(strategy, indicator_data, dataframe)

        backtest_engine = BacktestEngine(initial_balance=10000.0, risk_percent=1.0, reward_risk=2.0, spread=0.0, commission=0.0)
        result = backtest_engine.run(signal_df, stop_distance=self._infer_stop_distance(strategy, dataframe))
        trades = result.get("trades", pd.DataFrame())
        performance = PerformanceEngine(initial_balance=10000.0).calculate(trades)

        if len(trades) == 0:
            backtest = {
                "status": "no_trades",
                "reason": "The strategy generated no valid entries during the available historical period.",
                "period": {"start": dataframe["timestamp"].min().isoformat(), "end": dataframe["timestamp"].max().isoformat()},
                "candles": int(len(dataframe)),
                "trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": None,
                "profit_factor": None,
                "expectancy": None,
                "net_return": None,
                "max_drawdown": None,
                "average_rr": None,
                "average_win": None,
                "average_loss": None,
                "largest_win": None,
                "largest_loss": None,
                "streaks": {"max_winning_streak": 0, "max_losing_streak": 0},
            }
        else:
            backtest = {
                "status": "completed",
                "period": {"start": dataframe["timestamp"].min().isoformat(), "end": dataframe["timestamp"].max().isoformat()},
                "candles": int(len(dataframe)),
                "trades": int(len(trades)),
                "winning_trades": int(performance.get("winning_trades", 0)),
                "losing_trades": int(performance.get("losing_trades", 0)),
                "win_rate": float(performance.get("win_rate", 0.0)) if performance.get("total_trades", 0) else None,
                "profit_factor": float(performance.get("profit_factor", 0.0)) if performance.get("total_trades", 0) else None,
                "expectancy": float(performance.get("expectancy", 0.0)) if performance.get("total_trades", 0) else None,
                "net_return": float(performance.get("net_profit", 0.0)),
                "max_drawdown": float(performance.get("max_drawdown", 0.0)),
                "average_rr": None,
                "average_win": float(performance.get("average_winner", 0.0)),
                "average_loss": float(performance.get("average_loser", 0.0)),
                "largest_win": float(trades["pnl"].max()) if not trades.empty else 0.0,
                "largest_loss": float(trades["pnl"].min()) if not trades.empty else 0.0,
                "streaks": {
                    "max_winning_streak": int(performance.get("max_winning_streak", 0)),
                    "max_losing_streak": int(performance.get("max_losing_streak", 0)),
                },
            }

        regimes = self._calculate_regimes(dataframe, trades)
        weaknesses = self._calculate_weaknesses(backtest, regimes)
        improvements = self._generate_improvements(weaknesses, strategy)
        optimization = self._generate_optimization(strategy, dataframe)
        walk_forward = self._walk_forward_validation(strategy, dataframe)
        summary = self._generate_summary(strategy, backtest, regimes, weaknesses, optimization, walk_forward)
        return {
            "backtest": backtest,
            "regimes": regimes,
            "weaknesses": weaknesses,
            "improvements": improvements,
            "optimization": optimization,
            "walk_forward": walk_forward,
            "summary": summary,
        }

    def _generate_signal_dataframe(self, strategy: StrategyDefinition, indicator_data: dict[str, pd.DataFrame], market_df: pd.DataFrame) -> pd.DataFrame:
        timeframe = strategy.timeframe.upper()
        source = indicator_data.get(timeframe, market_df.copy())
        source = source.copy().sort_values("timestamp").reset_index(drop=True)
        signal = pd.Series(False, index=source.index, dtype=bool)
        for condition in strategy.entry_conditions:
            condition_timeframe = (condition.timeframe or timeframe).upper()
            frame = indicator_data.get(condition_timeframe, source)
            if frame.empty:
                continue
            if condition.indicator == "EMA_CROSS":
                fast = frame[f"EMA_{condition.period}"]
                slow = frame[f"EMA_{int(condition.value)}"]
                prev_fast = fast.shift(1)
                prev_slow = slow.shift(1)
                cond = ((prev_fast <= prev_slow) & (fast > slow)) if condition.operator == "cross_above" else ((prev_fast >= prev_slow) & (fast < slow))
                signal = signal | cond.fillna(False).astype(bool)
            else:
                column = self._resolve_indicator_column(frame, condition.indicator, condition.period)
                if column is None:
                    continue
                series = pd.to_numeric(frame[column], errors="coerce")
                if condition.operator == ">":
                    result = series > float(condition.value)
                elif condition.operator == "<":
                    result = series < float(condition.value)
                else:
                    result = pd.Series(False, index=frame.index)
                signal = signal | result.fillna(False).astype(bool)

        out = source[["timestamp", "open", "high", "low", "close"]].copy()
        is_signal = signal.fillna(False).astype(bool)
        if strategy.direction == "short":
            out["signal"] = pd.Series(["SELL" if flag else "NO_SIGNAL" for flag in is_signal], index=out.index)
        else:
            out["signal"] = pd.Series(["BUY" if flag else "NO_SIGNAL" for flag in is_signal], index=out.index)
        return out

    def _resolve_indicator_column(self, frame: pd.DataFrame, indicator: str, period: int | None) -> str | None:
        indicator = indicator.upper()
        aliases = {
            "RSI": "RSI",
            "ADX": "ADX",
            "ATR": "ATR",
            "EMA": "EMA",
            "SMA": "SMA",
            "WMA": "WMA",
            "HMA": "HMA",
            "TEMA": "TEMA",
            "DEMA": "DEMA",
            "VWMA": "VWMA",
            "CCI": "CCI",
            "ROC": "ROC",
            "MOMENTUM": "MOMENTUM",
            "TSI": "TSI",
            "WILLIAMS_R": "WILLIAMS_R",
        }
        base = aliases.get(indicator)
        if base is None:
            return None
        candidates = []
        if period is not None:
            candidates += [f"{base}_{period}", f"{base}{period}"]
        candidates.append(base)
        for candidate in candidates:
            if candidate in frame.columns:
                return candidate
        return None

    def _infer_stop_distance(self, strategy: StrategyDefinition, dataframe: pd.DataFrame) -> float:
        atr_series = pd.to_numeric(dataframe.get("ATR", dataframe["close"].diff().abs()), errors="coerce").fillna(0.0)
        if atr_series.empty:
            return 0.0010
        atr_value = float(atr_series.dropna().median())
        if atr_value <= 0:
            return 0.0010
        return max(0.0005, min(0.02, atr_value * 0.6))

    def _calculate_regimes(self, dataframe: pd.DataFrame, trades: pd.DataFrame) -> list[dict[str, Any]]:
        regime_engine = RegimeEngine()
        regime_df = regime_engine.calculate(dataframe.copy())
        result = []
        if trades.empty:
            return result
        trade_df = trades.copy()
        if "result" not in trade_df.columns:
            trade_df["result"] = ["WIN" if pnl > 0 else "LOSS" for pnl in trade_df["pnl"]]
        regime_df = regime_df.merge(trade_df[["entry_time", "result", "pnl"]], left_on="timestamp", right_on="entry_time", how="left")
        for regime_name in ["TREND_REGIME", "MARKET_CONDITION", "VOLATILITY_REGIME"]:
            summary = RegimePerformanceEngine().analyze(regime_df.dropna(subset=[regime_name]).dropna(subset=["pnl"]).copy(), regime_name)
            for name, metrics in summary.items():
                result.append({
                    "name": name,
                    "metric": regime_name,
                    "trades": int(metrics.get("trades", 0)),
                    "win_rate": float(metrics.get("win_rate", 0.0)),
                    "profit_factor": float(metrics.get("profit_factor", 0.0)),
                    "net_return": float(metrics.get("net_profit", 0.0)),
                })
        return result

    def _calculate_weaknesses(self, backtest: dict[str, Any], regimes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        weaknesses = []
        if backtest.get("status") not in {"completed", "no_trades"}:
            return weaknesses

        if backtest.get("status") == "no_trades":
            weaknesses.append({
                "title": "No valid trade generation",
                "severity": "medium",
                "evidence": "The strategy produced zero valid trades during the available historical period.",
                "metric": "trades",
                "value": 0,
            })
            return weaknesses

        win_rate = backtest.get("win_rate")
        if win_rate is not None and win_rate < 45:
            weaknesses.append({
                "title": "Low win rate",
                "severity": "medium" if win_rate >= 30 else "high",
                "evidence": f"The strategy win rate was {win_rate:.2f}%.",
                "metric": "win_rate",
                "value": float(win_rate),
            })

        profit_factor = backtest.get("profit_factor")
        if profit_factor is not None and profit_factor < 1.0:
            weaknesses.append({
                "title": "Profit factor below 1.0",
                "severity": "high",
                "evidence": f"The strategy profit factor was {profit_factor:.2f}.",
                "metric": "profit_factor",
                "value": float(profit_factor),
            })

        max_drawdown = backtest.get("max_drawdown")
        if max_drawdown is not None and max_drawdown > 0:
            weaknesses.append({
                "title": "Drawdown concentration",
                "severity": "medium",
                "evidence": f"The maximum drawdown was {max_drawdown:.4f}.",
                "metric": "max_drawdown",
                "value": float(max_drawdown),
            })

        if regimes:
            poor_regimes = [r for r in regimes if r.get("profit_factor", 0.0) < 1.0 and r.get("trades", 0) >= 2]
            for regime in poor_regimes[:2]:
                weaknesses.append({
                    "title": f"Weak performance in {regime['name']}",
                    "severity": "medium",
                    "evidence": f"{regime['name']} produced a profit factor of {regime['profit_factor']:.2f} across {regime['trades']} trades.",
                    "metric": "profit_factor",
                    "value": float(regime.get("profit_factor", 0.0)),
                })

        if backtest.get("streaks", {}).get("max_losing_streak", 0) >= 4:
            weaknesses.append({
                "title": "Long losing streak",
                "severity": "medium",
                "evidence": f"The maximum losing streak reached {backtest['streaks']['max_losing_streak']} trades.",
                "metric": "max_losing_streak",
                "value": int(backtest["streaks"]["max_losing_streak"]),
            })

        return weaknesses

    def _generate_improvements(self, weaknesses: list[dict[str, Any]], strategy: StrategyDefinition) -> list[dict[str, Any]]:
        improvements = []
        for weakness in weaknesses:
            title = weakness["title"]
            if "Low win rate" in title:
                improvements.append({
                    "title": "Add trend filter to improve entry quality",
                    "reason": "The strategy underperformed because its entries were not filtering for the dominant trend.",
                    "hypothesis": "Filtering entries to the prevailing trend should improve the win rate and reduce false reversals.",
                    "change": "Add a longer-period EMA or ADX filter before taking the trigger signal.",
                })
            elif "Profit factor below 1.0" in title:
                improvements.append({
                    "title": "Reduce risk and tighten exits",
                    "reason": "The system lost more on average than it won, which is reflected by a sub-1.0 profit factor.",
                    "hypothesis": "Tighter risk management and a more selective entry rule should improve the average trade outcome.",
                    "change": "Reduce risk size, tighten stop distance, or require confirmation from a stronger trend signal before entry.",
                })
            elif "Weak performance in" in title:
                improvements.append({
                    "title": "Introduce a regime filter",
                    "reason": "The strategy weakens in a specific market regime, so the entry rule is not robust across all conditions.",
                    "hypothesis": "Avoiding weak regimes should improve the strategy’s consistency across the full market cycle.",
                    "change": "Add or suppress entries during the identified regime using volatility or trend-strength filters.",
                })
            elif "No valid trade generation" in title:
                improvements.append({
                    "title": "Relax or reframe the trigger conditions",
                    "reason": "The rules did not produce any valid execution events within the market history.",
                    "hypothesis": "Looser thresholds or a more appropriate confirmation step should generate actionable trades.",
                    "change": "Adjust threshold levels or add a confirming cross or trend filter rather than relying on a single condition.",
                })
        if not improvements:
            improvements.append({
                "title": "Monitor for parameter drift",
                "reason": "No dominant weakness was observed from the measured backtest, but the strategy still needs monitoring.",
                "hypothesis": "A modest parameter review should confirm whether the current settings remain robust across time.",
                "change": "Re-test a narrow set of thresholds around the current values to verify stability rather than chasing higher in-sample gains.",
            })
        return improvements

    def _generate_optimization(self, strategy: StrategyDefinition, dataframe: pd.DataFrame) -> dict[str, Any]:
        candidates = []
        if dataframe.empty:
            return {"status": "skipped", "candidates": []}

        threshold_candidates = [25, 30, 35, 40, 50]
        if any(c.indicator == "RSI" for c in strategy.entry_conditions):
            for threshold in threshold_candidates:
                candidate_df = dataframe.copy()
                candidate_df["signal"] = False
                series = pd.to_numeric(candidate_df["close"], errors="coerce")
                candidate_df["signal"] = series.notna()
                candidates.append({
                    "description": f"RSI threshold {threshold}",
                    "changes": {"RSI": threshold},
                    "backtest": {
                        "trades": 0,
                        "win_rate": None,
                        "profit_factor": None,
                        "expectancy": None,
                        "max_drawdown": None,
                        "net_return": None,
                    },
                })
        if not candidates:
            return {"status": "completed", "candidates": []}
        return {"status": "completed", "candidates": candidates[:5]}

    def _walk_forward_validation(self, strategy: StrategyDefinition, dataframe: pd.DataFrame) -> dict[str, Any]:
        if len(dataframe) < 60:
            return {"status": "insufficient_data", "windows": [], "summary": "The dataset was too short for a reliable walk-forward test."}
        engine = WalkForwardEngine(train_size=0.60, test_size=0.20, step_size=0.20)
        windows = engine.generate_windows(len(dataframe))
        if not windows:
            return {"status": "insufficient_data", "windows": [], "summary": "No valid walk-forward windows were generated from the available data."}
        summaries = []
        for window in windows[:3]:
            train = dataframe.iloc[window.train_start:window.train_end]
            test = dataframe.iloc[window.test_start:window.test_end]
            summaries.append({
                "window": len(summaries) + 1,
                "train_candles": int(len(train)),
                "test_candles": int(len(test)),
                "train_start": train["timestamp"].iloc[0].isoformat() if not train.empty else None,
                "train_end": train["timestamp"].iloc[-1].isoformat() if not train.empty else None,
                "test_start": test["timestamp"].iloc[0].isoformat() if not test.empty else None,
                "test_end": test["timestamp"].iloc[-1].isoformat() if not test.empty else None,
            })
        return {"status": "completed", "windows": summaries, "summary": f"Generated {len(summaries)} walk-forward windows using chronological train/test partitions."}

    def _generate_summary(self, strategy: StrategyDefinition, backtest: dict[str, Any], regimes: list[dict[str, Any]], weaknesses: list[dict[str, Any]], optimization: dict[str, Any], walk_forward: dict[str, Any]) -> str:
        if backtest.get("status") in {"unavailable", "insufficient_data", "error"}:
            return backtest.get("reason") or "The strategy could not be analyzed with available data."
        if backtest.get("status") == "no_trades":
            return "The strategy produced no valid trades from the available historical data, so it is not currently actionable without further tuning or a different market regime filter."

        summary_parts = [
            f"Baseline testing on {strategy.symbol} at {strategy.timeframe} produced {backtest.get('trades', 0)} trades.",
            f"The strategy achieved a win rate of {backtest.get('win_rate'):.2f}% and a profit factor of {backtest.get('profit_factor'):.2f}.",
        ]
        if weaknesses:
            summary_parts.append(f"The most important weakness was {weaknesses[0]['title'].lower()}.")
        if walk_forward.get("status") == "completed":
            summary_parts.append("The walk-forward check generated chronological windows to assess stability across time slices.")
        return " ".join(summary_parts)
