from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import balanced_accuracy_score


EXECUTION_LEVEL_SCHEMA = 1
NO_TRADE = "NO_TRADE"
ACTION_CLASSES = (NO_TRADE, "BUY", "SELL")
LEVEL_NAMES = ("entry_offset_atr", "zone_width_atr", "stop_atr", "target_atr")
MODE_LIMITS = {
    "SCALP": {"max_stop_atr": 0.75, "min_reward_risk": 1.5},
    "SWING": {"max_stop_atr": 1.0, "min_reward_risk": 1.5},
}


def _simulate_plan(
    future_atr: np.ndarray,
    direction: str,
    entry_offset_atr: float,
    zone_width_atr: float,
    stop_atr: float,
    target_atr: float,
    cost_atr: float,
) -> tuple[float, bool]:
    half_width = zone_width_atr / 2.0
    zone_low = entry_offset_atr - half_width
    zone_high = entry_offset_atr + half_width
    entry_price = zone_high if direction == "BUY" else zone_low
    stop_price = entry_price - stop_atr if direction == "BUY" else entry_price + stop_atr
    target_price = entry_price + target_atr if direction == "BUY" else entry_price - target_atr
    filled = False

    for candle in future_atr:
        _, high, low, close = candle
        if not filled:
            if low <= zone_high and high >= zone_low:
                filled = True
            else:
                continue
        stop_hit = low <= stop_price if direction == "BUY" else high >= stop_price
        target_hit = high >= target_price if direction == "BUY" else low <= target_price
        if stop_hit:
            return -stop_atr - cost_atr, True
        if target_hit:
            return target_atr - cost_atr, True

    if not filled:
        return 0.0, False
    final_close = float(future_atr[-1, 3])
    gross = final_close - entry_price if direction == "BUY" else entry_price - final_close
    return gross - cost_atr, True


def label_execution_plan(
    future_atr: np.ndarray,
    mode: str,
    cost_atr: float,
) -> dict[str, float | str]:
    if mode not in MODE_LIMITS:
        raise ValueError(f"Unsupported execution mode: {mode}")
    candles = np.asarray(future_atr, dtype=float)
    if candles.ndim != 2 or candles.shape[1] != 4 or len(candles) == 0:
        raise ValueError("Future execution candles must be a non-empty OHLC array in ATR units.")
    if not np.isfinite(candles).all() or not np.isfinite(cost_atr) or cost_atr < 0:
        raise ValueError("Execution candles and cost must be finite; cost cannot be negative.")

    limits = MODE_LIMITS[mode]
    entry_offsets = (-0.5, -0.25, 0.0) if mode == "SCALP" else (-0.75, -0.375, 0.0)
    widths = (0.08, 0.16)
    stops = (0.35, 0.5, 0.75) if mode == "SCALP" else (0.5, 0.75, 1.0)
    targets = (0.75, 1.0, 1.25, 1.5, 2.0) if mode == "SCALP" else (1.0, 1.5, 2.0, 2.5, 3.0)
    best: dict[str, float | str] | None = None

    for direction in ("BUY", "SELL"):
        offsets = entry_offsets if direction == "BUY" else tuple(-value for value in entry_offsets)
        for offset in offsets:
            for width in widths:
                for stop in stops:
                    if stop > limits["max_stop_atr"]:
                        continue
                    for target in targets:
                        if target < stop * limits["min_reward_risk"]:
                            continue
                        net, filled = _simulate_plan(
                            candles, direction, offset, width, stop, target, cost_atr
                        )
                        if not filled or net <= 0:
                            continue
                        candidate: dict[str, float | str] = {
                            "action": direction,
                            "entry_offset_atr": float(offset),
                            "zone_width_atr": float(width),
                            "stop_atr": float(stop),
                            "target_atr": float(target),
                            "oracle_net_atr": float(net),
                        }
                        if best is None or net > float(best["oracle_net_atr"]):
                            best = candidate

    if best is not None:
        return best
    return {
        "action": NO_TRADE,
        "entry_offset_atr": 0.0,
        "zone_width_atr": 0.0,
        "stop_atr": 0.0,
        "target_atr": 0.0,
        "oracle_net_atr": 0.0,
    }


def _max_drawdown(returns: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    drawdown = 0.0
    for value in returns:
        equity += value
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return float(drawdown)


class ExecutionLevelTrainer:
    """Train and hold out a no-trade gate plus mode-specific POI/exit heads."""

    minimum_holdout_trades = 20
    minimum_confidence = 0.65

    @staticmethod
    def _classifier():
        return HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=300,
            max_leaf_nodes=15,
            min_samples_leaf=15,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=42,
        )

    @staticmethod
    def _regressor():
        return HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.04,
            max_iter=250,
            max_leaf_nodes=15,
            min_samples_leaf=15,
            l2_regularization=1.0,
            random_state=42,
        )

    def train(
        self,
        features: np.ndarray,
        actions: np.ndarray,
        levels: np.ndarray,
        future_candles_atr: np.ndarray,
        feature_names: list[str],
        mode: str,
        version: str,
        model_dir: str | Path = "models",
        purge: int = 12,
        cost_atr: float = 0.05,
    ) -> dict:
        X = np.asarray(features, dtype=float)
        y = np.asarray(actions, dtype=str)
        target = np.asarray(levels, dtype=float)
        future = np.asarray(future_candles_atr, dtype=float)
        if mode not in MODE_LIMITS:
            raise ValueError(f"Unsupported execution mode: {mode}")
        if (
            len(X) < 250
            or X.ndim != 2
            or y.shape != (len(X),)
            or target.shape != (len(X), len(LEVEL_NAMES))
            or future.ndim != 3
            or future.shape[0] != len(X)
            or future.shape[2] != 4
            or not np.isfinite(X).all()
            or not np.isfinite(target).all()
            or not np.isfinite(future).all()
            or not np.isfinite(cost_atr)
            or cost_atr < 0
        ):
            raise ValueError("Execution training requires 250+ aligned, finite features, labels, levels, and OHLC futures.")
        if not set(y).issubset(ACTION_CLASSES) or NO_TRADE not in set(y):
            raise ValueError("Execution actions must use BUY, SELL, and NO_TRADE labels.")
        test_start = int(len(X) * 0.8)
        train_end = test_start - max(purge, future.shape[1])
        if train_end < 100 or len(X) - test_start < self.minimum_holdout_trades:
            raise ValueError("Insufficient samples for the purged execution-model holdout.")
        train_actions = y[:train_end]
        for side in ("BUY", "SELL"):
            if int(np.sum(train_actions == side)) < 30:
                raise ValueError(f"Execution model needs at least 30 {side} training plans for {mode}.")

        classifier = self._classifier()
        classifier.fit(X[:train_end], train_actions)
        probabilities = classifier.predict_proba(X[test_start:])
        classes = np.asarray(classifier.classes_, dtype=str)
        predicted_indices = probabilities.argmax(axis=1)
        predicted_actions = classes[predicted_indices]
        confidence = probabilities[np.arange(len(probabilities)), predicted_indices]
        side_models: dict[str, dict[str, HistGradientBoostingRegressor]] = {}
        for side in ("BUY", "SELL"):
            mask = train_actions == side
            side_models[side] = {}
            for index, name in enumerate(LEVEL_NAMES):
                regressor = self._regressor()
                regressor.fit(X[:train_end][mask], target[:train_end][mask, index])
                side_models[side][name] = regressor

        executed_returns: list[float] = []
        holdout_plan_labels: list[str] = []
        for offset, predicted_side in enumerate(predicted_actions):
            sample_index = test_start + offset
            if (
                predicted_side not in {"BUY", "SELL"}
                or confidence[offset] < self.minimum_confidence
            ):
                continue
            predictions = side_models[str(predicted_side)]
            entry_offset = float(np.clip(
                predictions["entry_offset_atr"].predict(X[sample_index : sample_index + 1])[0],
                -0.75,
                0.75,
            ))
            width = float(np.clip(
                predictions["zone_width_atr"].predict(X[sample_index : sample_index + 1])[0],
                0.05,
                0.5,
            ))
            stop = float(np.clip(
                predictions["stop_atr"].predict(X[sample_index : sample_index + 1])[0],
                0.2,
                MODE_LIMITS[mode]["max_stop_atr"],
            ))
            target_distance = float(np.clip(
                predictions["target_atr"].predict(X[sample_index : sample_index + 1])[0],
                0.3,
                3.0,
            ))
            if target_distance < stop * MODE_LIMITS[mode]["min_reward_risk"]:
                continue
            net, filled = _simulate_plan(
                future[sample_index],
                str(predicted_side),
                entry_offset,
                width,
                stop,
                target_distance,
                cost_atr,
            )
            if filled:
                executed_returns.append(net)
                holdout_plan_labels.append(str(predicted_side))

        actual_actions = y[test_start:]
        metrics = {
            "schema_version": EXECUTION_LEVEL_SCHEMA,
            "mode": mode,
            "samples": int(len(X)),
            "training_samples": int(train_end),
            "purged_samples": int(test_start - train_end),
            "chronological_test_samples": int(len(X) - test_start),
            "classifier_balanced_accuracy": float(balanced_accuracy_score(
                actual_actions, predicted_actions
            )),
            "minimum_confidence": self.minimum_confidence,
            "modeled_cost_atr": float(cost_atr),
            "holdout_trades": len(executed_returns),
            "holdout_net_atr": float(sum(executed_returns)),
            "holdout_mean_net_atr": float(np.mean(executed_returns)) if executed_returns else 0.0,
            "holdout_win_rate": (
                float(np.mean(np.asarray(executed_returns) > 0))
                if executed_returns else 0.0
            ),
            "holdout_max_drawdown_atr": _max_drawdown(executed_returns),
            "holdout_trade_counts": {
                side: holdout_plan_labels.count(side) for side in ("BUY", "SELL")
            },
            "max_stop_atr": MODE_LIMITS[mode]["max_stop_atr"],
            "minimum_reward_risk": MODE_LIMITS[mode]["min_reward_risk"],
        }
        rejections = []
        if metrics["classifier_balanced_accuracy"] < 0.55:
            rejections.append("chronological execution-action balanced accuracy below 0.55")
        if metrics["holdout_trades"] < self.minimum_holdout_trades:
            rejections.append("fewer than 20 filled holdout executions")
        if metrics["holdout_net_atr"] <= 0:
            rejections.append("holdout net return after modeled costs is not positive")
        if metrics["holdout_win_rate"] < 0.45:
            rejections.append("holdout win rate below 0.45")
        if metrics["holdout_max_drawdown_atr"] > MODE_LIMITS[mode]["max_stop_atr"] * 3:
            rejections.append("holdout drawdown exceeds three maximum stop distances")
        metrics["validation_rejections"] = rejections
        metrics["validation_status"] = "passed" if not rejections else "rejected"

        if not rejections:
            classifier = self._classifier()
            classifier.fit(X, y)
            for side in ("BUY", "SELL"):
                mask = y == side
                side_models[side] = {}
                for index, name in enumerate(LEVEL_NAMES):
                    regressor = self._regressor()
                    regressor.fit(X[mask], target[mask, index])
                    side_models[side][name] = regressor
        bundle = {
            "schema_version": EXECUTION_LEVEL_SCHEMA,
            "mode": mode,
            "classifier": classifier,
            "classes": [str(label) for label in classifier.classes_],
            "side_models": side_models,
            "feature_names": list(feature_names),
            "minimum_confidence": self.minimum_confidence,
            "validation_status": metrics["validation_status"],
            "metrics": metrics,
            "version": version,
        }
        directory = Path(model_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{version}.joblib"
        joblib.dump(bundle, path)
        path.with_suffix(".json").write_text(json.dumps(metrics, indent=2))
        return {"path": str(path), "version": version, "metrics": metrics}
