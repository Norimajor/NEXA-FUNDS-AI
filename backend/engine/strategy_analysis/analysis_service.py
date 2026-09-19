import math
from copy import deepcopy
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
        assumptions = self._assumptions_for(prompt, strategy)

        report = {
            "success": True,
            "strategy": self._serialize_strategy(strategy, assumptions),
            "validation": {
                "status": "valid" if validation["valid"] else "invalid",
                "errors": validation.get("errors", []),
                "warnings": validation.get("warnings", []),
            },
            "data": data_info,
            "backtest": {"status": "unavailable", "reason": "No backtest executed."},
            "test": {"status": "unavailable", "reason": "No backtest executed."},
            "performance_analysis": self._empty_performance_analysis(),
            "regimes": [],
            "weaknesses": [],
            "improvements": [],
            "optimization": {"status": "skipped", "candidates": []},
            "recommendations": [],
            "candidates": [],
            "robustness": self._empty_robustness(),
            "walk_forward": {"status": "skipped", "windows": [], "summary": "Walk-forward validation was skipped."},
            "summary": "",
        }

        if not validation["valid"]:
            report["test"] = {
                "status": "invalid",
                "reason": "The strategy cannot be tested until the validation errors are resolved.",
                "metrics": {},
            }
            report["summary"] = "The strategy description was parsed, but critical rules are missing: " + "; ".join(validation["errors"])
            return report

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
                report["test"] = {
                    "status": "no_trades",
                    "reason": report["backtest"]["reason"],
                    "metrics": report["backtest"],
                }
                return report

            report["backtest"] = {
                "status": "unavailable",
                "reason": data_info.get("reason") or f"Historical data for {symbol} {timeframe} is not available.",
            }
            report["test"] = {
                "status": "unavailable",
                "reason": "Data unavailable for this test. " + report["backtest"]["reason"],
                "data_required": f"Historical OHLCV data for {symbol} {timeframe}.",
            }
            report["summary"] = f"No historical data was available for {symbol} {timeframe}, so no real backtest could be produced."
            return report

        try:
            benchmark = self._evaluate_strategy(strategy, data_info)
            report["backtest"] = benchmark["backtest"]
            report["test"] = benchmark["test"]
            report["performance_analysis"] = benchmark["performance_analysis"]
            report["regimes"] = benchmark.get("regimes", [])
            report["weaknesses"] = benchmark.get("weaknesses", [])
            report["improvements"] = benchmark.get("improvements", [])
            report["optimization"] = benchmark.get("optimization", {"status": "skipped", "candidates": []})
            report["recommendations"] = benchmark.get("recommendations", [])
            report["candidates"] = benchmark.get("candidates", [])
            report["robustness"] = benchmark.get("robustness", self._empty_robustness())
            report["walk_forward"] = benchmark.get("walk_forward", {"status": "skipped", "windows": [], "summary": "Walk-forward validation was skipped."})
            report["summary"] = benchmark.get("summary", report["summary"])
        except Exception as exc:
            report["backtest"] = {
                "status": "error",
                "reason": str(exc),
            }
            report["test"] = {
                "status": "error",
                "reason": "The strategy could not be tested because the analysis engine raised an error.",
                "details": str(exc),
            }
            report["summary"] = f"The strategy could not be fully analyzed because: {exc}"

        return report

    def _serialize_strategy(self, strategy: StrategyDefinition, assumptions: list[str] | None = None) -> dict[str, Any]:
        return {
            "name": strategy.name,
            "instrument": strategy.symbol,
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
            "exit_conditions": [self._serialize_condition(condition) for condition in strategy.exit_conditions],
            "filters": [self._serialize_condition(condition) for condition in strategy.filters],
            "risk_percent": strategy.risk_percent,
            "risk_reward": strategy.risk_reward,
            "assumptions": assumptions or [],
        }

    def _serialize_condition(self, condition) -> dict[str, Any]:
        return {
            "indicator": condition.indicator,
            "operator": condition.operator,
            "value": condition.value,
            "period": condition.period,
            "timeframe": condition.timeframe,
            "side": condition.side,
        }

    def _assumptions_for(self, prompt: str, strategy: StrategyDefinition) -> list[str]:
        assumptions = []
        if not any(symbol in prompt.upper() for symbol in ("XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "EURJPY", "GBPJPY", "AUDJPY", "NAS100", "US30", "BTCUSD", "ETHUSD")):
            assumptions.append(f"Instrument defaulted to {strategy.symbol} because no supported instrument was named.")
        if not any(token in prompt.upper() for token in ("M1", "M5", "M15", "M30", "H1", "H4", "1M", "5M", "15M", "30M", "1H", "4H")):
            assumptions.append(f"Timeframe defaulted to {strategy.timeframe} because no supported timeframe was named.")
        if not any(token in prompt.lower() for token in ("risk", "stop", "sl")):
            assumptions.append("Stop distance was inferred from the median available ATR, capped to the engine safety bounds.")
        if not any(token in prompt.lower() for token in ("risk reward", "risk/reward", "rr", "take profit", "target")):
            assumptions.append(f"Risk/reward defaulted to {strategy.risk_reward:.2f}:1 because no exit target was specified.")
        return assumptions

    def _empty_performance_analysis(self) -> dict[str, Any]:
        return {
            "strengths": [],
            "weaknesses": [],
            "profitable_conditions": [],
            "poor_conditions": [],
            "regime_analysis": {},
            "session_analysis": {},
            "limitations": [],
        }

    def _empty_robustness(self) -> dict[str, Any]:
        return {
            "status": "not_available",
            "walk_forward_available": False,
            "observations": [],
            "windows": [],
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
            reason = f"Not enough candles for a reliable backtest on {strategy.symbol} {strategy.timeframe}."
            return {
                "backtest": {
                    "status": "insufficient_data",
                    "reason": reason,
                },
                "test": {"status": "insufficient_data", "reason": reason, "metrics": {}},
                "performance_analysis": {
                    **self._empty_performance_analysis(),
                    "limitations": [reason],
                },
                "regimes": [],
                "weaknesses": [],
                "improvements": [],
                "optimization": {"status": "skipped", "candidates": []},
                "recommendations": [],
                "candidates": [],
                "robustness": self._empty_robustness(),
                "walk_forward": {"status": "insufficient_data", "windows": [], "summary": "Not enough data for walk-forward validation."},
                "summary": reason,
            }

        run = self._run_backtest(strategy, dataframe)
        trades = run["trades"]
        backtest = run["metrics"]

        regimes = self._calculate_regimes(dataframe, trades)
        weaknesses = self._calculate_weaknesses(backtest, regimes)
        improvements = self._generate_improvements(weaknesses, strategy)
        optimization = self._generate_optimization(strategy, dataframe)
        optimization["baseline"] = self._candidate_metrics(backtest)
        walk_forward = self._walk_forward_validation(strategy, dataframe)
        performance_analysis = self._build_performance_analysis(backtest, regimes, trades)
        recommendations = self._generate_recommendations(backtest, regimes, performance_analysis, strategy)
        summary = self._generate_summary(strategy, backtest, regimes, weaknesses, optimization, walk_forward)
        return {
            "backtest": backtest,
            "test": {
                "status": backtest["status"],
                "data_period": backtest.get("period"),
                "metrics": backtest,
                "reason": backtest.get("reason"),
            },
            "performance_analysis": performance_analysis,
            "regimes": regimes,
            "weaknesses": weaknesses,
            "improvements": improvements,
            "optimization": optimization,
            "recommendations": recommendations,
            "candidates": optimization.get("candidates", []),
            "robustness": walk_forward.get("robustness", self._empty_robustness()),
            "walk_forward": walk_forward,
            "summary": summary,
        }

    def _run_backtest(self, strategy: StrategyDefinition, dataframe: pd.DataFrame) -> dict[str, Any]:
        indicator_frame = self.indicator_engine.calculate_indicators(dataframe.copy())
        indicator_data = {strategy.timeframe: indicator_frame}
        signal_df = self._generate_signal_dataframe(strategy, indicator_data, dataframe)
        backtest_engine = BacktestEngine(
            initial_balance=10000.0,
            risk_percent=strategy.risk_percent,
            reward_risk=strategy.risk_reward,
            spread=0.0,
            commission=0.0,
        )
        result = backtest_engine.run(signal_df, stop_distance=self._infer_stop_distance(strategy, indicator_frame))
        trades = result.get("trades", pd.DataFrame())
        return {"trades": trades, "metrics": self._metrics_from_trades(trades, dataframe, strategy)}

    def _metrics_from_trades(self, trades: pd.DataFrame, dataframe: pd.DataFrame, strategy: StrategyDefinition) -> dict[str, Any]:
        period = {
            "start": dataframe["timestamp"].min().isoformat() if not dataframe.empty else None,
            "end": dataframe["timestamp"].max().isoformat() if not dataframe.empty else None,
        }
        base = {
            "status": "completed" if not trades.empty else "no_trades",
            "period": period,
            "candles": int(len(dataframe)),
            "trades": int(len(trades)),
            "wins": 0,
            "losses": 0,
            "winning_trades": 0,
            "losing_trades": 0,
        }
        if trades is None or trades.empty:
            base.update({
                "reason": "The strategy generated no valid entries during the available historical period.",
                "win_rate": None,
                "gross_profit": None,
                "gross_loss": None,
                "net_profit": None,
                "net_return": None,
                "profit_factor": None,
                "expectancy": None,
                "average_win": None,
                "average_loss": None,
                "average_rr": None,
                "max_drawdown": None,
                "average_drawdown": None,
                "largest_win": None,
                "largest_loss": None,
                "max_winning_streak": None,
                "max_losing_streak": None,
                "trade_frequency": None,
                "average_trade_duration_minutes": None,
            })
            return base

        performance = PerformanceEngine(initial_balance=10000.0).calculate(trades)
        pnl = pd.to_numeric(trades["pnl"], errors="coerce").fillna(0.0)
        balances = pd.concat([pd.Series([10000.0]), pd.to_numeric(trades["balance"], errors="coerce")], ignore_index=True)
        drawdowns = balances - balances.cummax()
        nonzero_drawdowns = drawdowns[drawdowns < 0].abs()
        durations = (pd.to_datetime(trades["exit_time"]) - pd.to_datetime(trades["entry_time"])).dt.total_seconds() / 60.0
        stop_distance = (pd.to_numeric(trades["entry"], errors="coerce") - pd.to_numeric(trades["stop_loss"], errors="coerce")).abs()
        reward_distance = (pd.to_numeric(trades["take_profit"], errors="coerce") - pd.to_numeric(trades["entry"], errors="coerce")).abs()
        period_days = max((dataframe["timestamp"].max() - dataframe["timestamp"].min()).total_seconds() / 86400.0, 0.0)
        profit_factor = self._finite_number(performance.get("profit_factor"))
        base.update({
            "reason": None,
            "wins": int(performance.get("winning_trades", 0)),
            "losses": int(performance.get("losing_trades", 0)),
            "winning_trades": int(performance.get("winning_trades", 0)),
            "losing_trades": int(performance.get("losing_trades", 0)),
            "win_rate": self._finite_number(performance.get("win_rate")),
            "gross_profit": self._finite_number(performance.get("gross_profit")),
            "gross_loss": self._finite_number(performance.get("gross_loss")),
            "net_profit": self._finite_number(performance.get("net_profit")),
            "net_return": self._finite_number(performance.get("net_profit")),
            "total_return_percent": self._finite_number((float(performance.get("net_profit", 0.0)) / 10000.0) * 100.0),
            "profit_factor": profit_factor,
            "expectancy": self._finite_number(performance.get("expectancy")),
            "average_win": self._finite_number(performance.get("average_winner")),
            "average_loss": self._finite_number(performance.get("average_loser")),
            "average_rr": self._finite_number((reward_distance / stop_distance.replace(0, pd.NA)).dropna().mean()),
            "max_drawdown": self._finite_number(performance.get("max_drawdown")),
            "average_drawdown": self._finite_number(nonzero_drawdowns.mean() if not nonzero_drawdowns.empty else 0.0),
            "largest_win": self._finite_number(pnl.max()),
            "largest_loss": self._finite_number(pnl.min()),
            "max_winning_streak": int(performance.get("max_winning_streak", 0)),
            "max_losing_streak": int(performance.get("max_losing_streak", 0)),
            "trade_frequency": self._finite_number(len(trades) / period_days) if period_days > 0 else None,
            "average_trade_duration_minutes": self._finite_number(durations.mean()),
            "risk_reward": strategy.risk_reward,
        })
        base["streaks"] = {
            "max_winning_streak": base["max_winning_streak"],
            "max_losing_streak": base["max_losing_streak"],
        }
        return base

    def _finite_number(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    def _build_performance_analysis(self, backtest: dict[str, Any], regimes: list[dict[str, Any]], trades: pd.DataFrame) -> dict[str, Any]:
        analysis = self._empty_performance_analysis()
        if backtest.get("status") not in {"completed", "no_trades"}:
            analysis["limitations"].append("Performance attribution requires sufficient historical data and completed trades.")
            return analysis
        if backtest.get("status") == "no_trades":
            analysis["limitations"].append("No condition, regime, or session performance can be attributed because no trades were generated.")
            return analysis

        if backtest.get("profit_factor") is not None and backtest["profit_factor"] > 1:
            analysis["strengths"].append({"finding": "Positive gross-profit to gross-loss ratio", "evidence": f"Measured profit factor was {backtest['profit_factor']:.2f} across {backtest['trades']} trades."})
        if backtest.get("expectancy") is not None and backtest["expectancy"] > 0:
            analysis["strengths"].append({"finding": "Positive average trade expectancy", "evidence": f"Measured expectancy was {backtest['expectancy']:.6f} per trade."})
        if backtest.get("max_losing_streak", 0) >= 4:
            analysis["weaknesses"].append({"finding": "Extended losing streak", "evidence": f"The longest measured losing streak was {backtest['max_losing_streak']} trades."})
        if backtest.get("max_drawdown") is not None and backtest["max_drawdown"] > 0:
            analysis["weaknesses"].append({"finding": "Observed drawdown", "evidence": f"Maximum measured drawdown was {backtest['max_drawdown']:.6f}."})

        for item in regimes:
            analysis["regime_analysis"].setdefault(item["metric"], {})[item["name"]] = {
                "trades": item["trades"],
                "win_rate": item["win_rate"],
                "profit_factor": item["profit_factor"],
                "net_profit": item.get("net_profit", item.get("net_return")),
            }
            target = analysis["profitable_conditions"] if item["profit_factor"] > 1 and item["trades"] > 0 else analysis["poor_conditions"] if item["profit_factor"] < 1 and item["trades"] > 0 else None
            if target is not None:
                target.append({
                    "condition": f"{item['metric']}={item['name']}",
                    "evidence": f"{item['trades']} trades, {item['win_rate']:.2f}% win rate, profit factor {item['profit_factor']:.2f}.",
                })

        if trades is not None and not trades.empty:
            session_groups = trades.copy()
            session_groups["entry_time"] = pd.to_datetime(session_groups["entry_time"])
            session_groups["session"] = session_groups["entry_time"].dt.hour.map(self._session_for_hour)
            for session, group in session_groups.groupby("session"):
                analysis["session_analysis"][session] = self._summarize_trade_group(group)
        analysis["limitations"].append("Condition-level attribution is unavailable because the backtest engine records trades, not the individual condition that triggered each entry.")
        return analysis

    def _summarize_trade_group(self, trades: pd.DataFrame) -> dict[str, Any]:
        pnl = pd.to_numeric(trades["pnl"], errors="coerce").fillna(0.0)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        gross_loss = abs(float(losses.sum()))
        profit_factor = float(wins.sum()) / gross_loss if gross_loss > 0 else None
        return {
            "trades": int(len(trades)),
            "wins": int(len(wins)),
            "losses": int(len(losses)),
            "win_rate": float(len(wins) / len(trades) * 100) if len(trades) else None,
            "gross_profit": float(wins.sum()),
            "gross_loss": gross_loss,
            "net_profit": float(pnl.sum()),
            "profit_factor": profit_factor,
            "expectancy": float(pnl.mean()) if len(pnl) else None,
        }

    def _session_for_hour(self, hour: int) -> str:
        if 0 <= hour < 7:
            return "Asia"
        if 7 <= hour < 13:
            return "London"
        if 13 <= hour < 21:
            return "New York"
        return "Off-hours"

    def _generate_recommendations(self, backtest: dict[str, Any], regimes: list[dict[str, Any]], performance_analysis: dict[str, Any], strategy: StrategyDefinition) -> list[dict[str, Any]]:
        recommendations = []
        if backtest.get("status") == "no_trades":
            return [{
                "recommendation": "Revisit the entry thresholds or add a confirmation rule before further testing.",
                "reason": "The selected rules produced no executable trades in the available history.",
                "evidence": "Measured trade count was 0; no profitability conclusion is possible.",
            }]
        for condition in performance_analysis.get("poor_conditions", [])[:3]:
            recommendations.append({
                "recommendation": f"Test filtering or excluding {condition['condition']}.",
                "reason": "This measured condition had below-1.0 profit factor.",
                "evidence": condition["evidence"],
            })
        if backtest.get("max_losing_streak", 0) >= 4:
            recommendations.append({
                "recommendation": "Test a lower risk percentage or a pause after a losing streak.",
                "reason": "The observed losing streak creates avoidable exposure concentration.",
                "evidence": f"The measured maximum losing streak was {backtest['max_losing_streak']} trades and maximum drawdown was {backtest.get('max_drawdown')}. ",
            })
        if backtest.get("profit_factor") is not None and backtest["profit_factor"] < 1:
            recommendations.append({
                "recommendation": "Test stronger confirmation or a different exit configuration.",
                "reason": "The baseline lost more gross profit than it generated.",
                "evidence": f"Measured gross profit was {backtest.get('gross_profit')} versus gross loss {backtest.get('gross_loss')}; profit factor was {backtest['profit_factor']:.2f}.",
            })
        if not recommendations:
            recommendations.append({
                "recommendation": "Test a narrow higher-timeframe trend filter using the supported EMA indicators.",
                "reason": "The baseline did not expose a dominant measured weakness requiring a specific change.",
                "evidence": f"The baseline completed with {backtest.get('trades', 0)} trades; this is a hypothesis for further testing, not a guaranteed improvement.",
            })
        return recommendations

    def _generate_signal_dataframe(self, strategy: StrategyDefinition, indicator_data: dict[str, pd.DataFrame], market_df: pd.DataFrame) -> pd.DataFrame:
        timeframe = strategy.timeframe.upper()
        source = indicator_data.get(timeframe, market_df.copy())
        source = source.copy().sort_values("timestamp").reset_index(drop=True)
        signal = pd.Series(True, index=source.index, dtype=bool)
        if not strategy.entry_conditions:
            signal = pd.Series(False, index=source.index, dtype=bool)
        for condition in strategy.entry_conditions:
            condition_timeframe = (condition.timeframe or timeframe).upper()
            frame = indicator_data.get(condition_timeframe, source)
            if frame.empty:
                signal &= False
                continue
            if condition.indicator == "EMA_CROSS":
                fast_column = f"EMA_{condition.period}"
                slow_column = f"EMA_{int(condition.value)}"
                if fast_column not in frame.columns or slow_column not in frame.columns:
                    signal &= False
                    continue
                fast = frame[fast_column]
                slow = frame[slow_column]
                prev_fast = fast.shift(1)
                prev_slow = slow.shift(1)
                cond = ((prev_fast <= prev_slow) & (fast > slow)) if condition.operator == "cross_above" else ((prev_fast >= prev_slow) & (fast < slow))
                signal &= cond.fillna(False).astype(bool).to_numpy()
            else:
                column = self._resolve_indicator_column(frame, condition.indicator, condition.period)
                if column is None:
                    signal &= False
                    continue
                series = pd.to_numeric(frame[column], errors="coerce")
                if condition.operator == ">":
                    result = series > float(condition.value)
                elif condition.operator == "<":
                    result = series < float(condition.value)
                else:
                    result = pd.Series(False, index=frame.index)
                signal &= result.fillna(False).astype(bool).to_numpy()

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
        if dataframe.empty:
            return {"status": "skipped", "candidates": []}

        rsi_conditions = [condition for condition in strategy.entry_conditions if condition.indicator == "RSI"]
        if not rsi_conditions:
            candidates = [
                {
                    "name": "Higher-timeframe EMA filter",
                    "change": "Add a supported EMA trend filter before the existing entry.",
                    "reason": "EMA indicators are available in the existing indicator engine, but this variation was not automatically tested because the current interpreter does not model filter composition separately.",
                    "test_status": "not_tested",
                    "results": None,
                },
                {
                    "name": "ATR volatility filter",
                    "change": "Require a minimum ATR condition before entry.",
                    "reason": "ATR is available in the existing indicator engine, but a threshold was not specified by the strategy and should not be invented.",
                    "test_status": "not_tested",
                    "results": None,
                },
            ]
            return {"status": "not_tested", "candidates": candidates}

        candidates = []
        for threshold in [25, 30, 35, 40, 50]:
            candidate = deepcopy(strategy)
            for condition in candidate.entry_conditions:
                if condition.indicator == "RSI":
                    condition.value = threshold
            result = self._run_backtest(candidate, dataframe)
            metrics = result["metrics"]
            candidates.append({
                "name": f"RSI threshold {threshold}",
                "description": f"Use RSI threshold {threshold} for the parsed RSI condition.",
                "change": {"RSI": threshold},
                "reason": "This candidate was re-tested on the same historical candles with only the RSI threshold changed.",
                "test_status": metrics["status"],
                "results": self._candidate_metrics(metrics),
                "backtest": self._candidate_metrics(metrics),
            })
        return {"status": "completed", "candidates": candidates}

    def _candidate_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": metrics.get("status"),
            "trades": metrics.get("trades"),
            "win_rate": metrics.get("win_rate"),
            "profit_factor": metrics.get("profit_factor"),
            "expectancy": metrics.get("expectancy"),
            "net_profit": metrics.get("net_profit"),
            "max_drawdown": metrics.get("max_drawdown"),
        }

    def _walk_forward_validation(self, strategy: StrategyDefinition, dataframe: pd.DataFrame) -> dict[str, Any]:
        if len(dataframe) < 60:
            return {
                "status": "insufficient_data",
                "windows": [],
                "summary": "The dataset was too short for a reliable walk-forward test.",
                "robustness": {
                    "status": "insufficient_data",
                    "walk_forward_available": False,
                    "observations": ["At least 60 candles are required for the configured chronological partitions."],
                    "windows": [],
                },
            }
        engine = WalkForwardEngine(train_size=0.60, test_size=0.20, step_size=0.20)
        windows = engine.generate_windows(len(dataframe))
        if not windows:
            return {
                "status": "insufficient_data",
                "windows": [],
                "summary": "No valid walk-forward windows were generated from the available data.",
                "robustness": self._empty_robustness(),
            }
        summaries = []
        for window in windows[:3]:
            train = dataframe.iloc[window.train_start:window.train_end]
            test = dataframe.iloc[window.test_start:window.test_end]
            train_metrics = self._run_backtest(strategy, train)["metrics"]
            test_metrics = self._run_backtest(strategy, test)["metrics"]
            summaries.append({
                "window": len(summaries) + 1,
                "train_candles": int(len(train)),
                "test_candles": int(len(test)),
                "train_start": train["timestamp"].iloc[0].isoformat() if not train.empty else None,
                "train_end": train["timestamp"].iloc[-1].isoformat() if not train.empty else None,
                "test_start": test["timestamp"].iloc[0].isoformat() if not test.empty else None,
                "test_end": test["timestamp"].iloc[-1].isoformat() if not test.empty else None,
                "train": self._candidate_metrics(train_metrics),
                "test": self._candidate_metrics(test_metrics),
            })
        profitable_test_windows = sum(1 for window in summaries if (window["test"].get("net_profit") or 0) > 0)
        test_trades = sum(window["test"].get("trades") or 0 for window in summaries)
        robustness_status = "inconclusive"
        observations = [f"Re-tested the frozen strategy across {len(summaries)} chronological train/test windows."]
        if profitable_test_windows / len(summaries) >= 0.6 and test_trades >= 20:
            robustness_status = "promising_but_unproven"
            observations.append(f"{profitable_test_windows} of {len(summaries)} test windows were profitable with {test_trades} out-of-sample trades.")
        else:
            observations.append(f"Only {profitable_test_windows} of {len(summaries)} test windows were profitable with {test_trades} out-of-sample trades; the evidence is limited.")
        robustness = {
            "status": robustness_status,
            "walk_forward_available": True,
            "observations": observations,
            "windows": summaries,
        }
        return {
            "status": "completed",
            "windows": summaries,
            "summary": f"Re-tested the frozen strategy across {len(summaries)} chronological train/test partitions.",
            "robustness": robustness,
        }

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
