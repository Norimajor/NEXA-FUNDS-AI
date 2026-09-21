"""Deterministic ranking of measured, compatible strategy candidates."""

from __future__ import annotations

import math
from typing import Any


class CombinationRankingService:
    def __init__(self, minimum_trades: int = 5):
        self.minimum_trades = minimum_trades

    def rank(self, records: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
        candidates = []
        for record in records:
            strategy = record.get("strategy") or {}
            baseline = record.get("backtest") or {}
            candidates.extend(self._measured(record, strategy, baseline, "baseline"))
            optimization = record.get("optimization") or {}
            for index, candidate in enumerate(optimization.get("candidates") or []):
                metrics = candidate.get("results") or candidate.get("backtest")
                if isinstance(metrics, dict):
                    candidates.extend(self._measured(record, strategy, metrics, f"candidate-{index}", candidate))

        compatible = [candidate for candidate in candidates if self._compatible(candidate, candidates)]
        ranked = sorted(compatible, key=lambda item: (-item["score"], item["strategy_fingerprint"] or "", item["source"]))
        return ranked[:limit]

    def _measured(self, record, strategy, metrics, source, details=None):
        status = metrics.get("status")
        trades = metrics.get("trades")
        if status != "completed" or not isinstance(trades, (int, float)) or trades < self.minimum_trades:
            return []
        numeric = ("net_profit", "expectancy", "profit_factor", "win_rate", "max_drawdown")
        if any(not isinstance(metrics.get(key), (int, float)) or not math.isfinite(float(metrics[key])) for key in numeric):
            return []
        robustness = record.get("walk_forward") or {}
        if robustness.get("status") != "completed":
            return []
        score = (float(metrics["expectancy"]) + float(metrics["net_profit"]) * 0.01
                 + min(float(metrics["profit_factor"]), 10.0) * 2.0
                 + float(metrics["win_rate"]) * 0.02
                 - abs(float(metrics["max_drawdown"])) * 0.01)
        return [{
            "source": source,
            "analysis_id": record.get("id"),
            "strategy_fingerprint": record.get("strategy_fingerprint"),
            "symbol": strategy.get("symbol") or strategy.get("instrument"),
            "timeframe": strategy.get("timeframe"),
            "direction": strategy.get("direction"),
            "metrics": metrics,
            "score": round(score, 10),
            "details": details or {},
        }]

    def _compatible(self, candidate, all_candidates):
        # Ranking is intentionally limited to the same market context. A
        # combination is only meaningful when every component has measured data.
        return bool(candidate.get("symbol") and candidate.get("timeframe") and candidate.get("direction"))
