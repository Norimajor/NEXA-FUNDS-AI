from dataclasses import dataclass
from typing import List

import pandas as pd

from .models import StrategyCondition, StrategyDefinition


@dataclass
class ConditionResult:
    condition: StrategyCondition
    passed: bool
    actual_value: float | None = None
    message: str = ""


@dataclass
class StrategyEvaluation:
    signal: str
    confidence: float
    passed: bool
    results: List[ConditionResult]


class StrategyEvaluator:

    SUPPORTED_TIMEFRAMES = {
        "H4",
        "H1",
        "M30",
        "M15",
        "M5",
        "M1",
    }

    # ========================================================
    # PUBLIC
    # ========================================================

    def evaluate(
        self,
        strategy: StrategyDefinition,
        indicator_data: dict[str, pd.DataFrame],
    ) -> StrategyEvaluation:

        results = []

        for condition in strategy.entry_conditions:

            result = self._evaluate_condition(
                condition,
                indicator_data,
            )

            results.append(result)

        if not results:

            return StrategyEvaluation(
                signal="NO_SIGNAL",
                confidence=0.0,
                passed=False,
                results=[],
            )

        passed_count = sum(
            1 for result in results
            if result.passed
        )

        total = len(results)

        confidence = (
            passed_count / total
        ) * 100.0

        all_passed = (
            passed_count == total
        )

        if all_passed:

            if strategy.direction == "long":

                signal = "BUY"

            elif strategy.direction == "short":

                signal = "SELL"

            else:

                signal = "SIGNAL"

        else:

            signal = "NO_SIGNAL"

        return StrategyEvaluation(
            signal=signal,
            confidence=confidence,
            passed=all_passed,
            results=results,
        )

    # ========================================================
    # CONDITION
    # ========================================================

    def _evaluate_condition(
        self,
        condition: StrategyCondition,
        indicator_data: dict[str, pd.DataFrame],
    ) -> ConditionResult:

        timeframe = (
            condition.timeframe
            if condition.timeframe
            else "M15"
        )

        timeframe = timeframe.upper()

        if timeframe not in self.SUPPORTED_TIMEFRAMES:

            return ConditionResult(
                condition=condition,
                passed=False,
                message=(
                    f"Unsupported timeframe: "
                    f"{timeframe}"
                ),
            )

        if timeframe not in indicator_data:

            return ConditionResult(
                condition=condition,
                passed=False,
                message=(
                    f"No indicator data for "
                    f"{timeframe}"
                ),
            )

        df = indicator_data[timeframe]

        if df.empty:

            return ConditionResult(
                condition=condition,
                passed=False,
                message=(
                    f"Empty indicator data: "
                    f"{timeframe}"
                ),
            )

        row = df.iloc[-1]

        indicator = (
            condition.indicator.upper()
        )

        # ====================================================
        # EMA CROSS
        # ====================================================

        if indicator == "EMA_CROSS":

            return self._evaluate_ema_cross(
                condition,
                df,
                timeframe,
            )

        # ====================================================
        # NORMAL INDICATOR
        # ====================================================

        column = self._find_indicator_column(
            df,
            indicator,
            condition.period,
        )

        if column is None:

            return ConditionResult(
                condition=condition,
                passed=False,
                message=(
                    f"Indicator {indicator} "
                    f"not found on {timeframe}"
                ),
            )

        actual = row[column]

        if pd.isna(actual):

            return ConditionResult(
                condition=condition,
                passed=False,
                message=(
                    f"{indicator} has no valid "
                    f"value on {timeframe}"
                ),
            )

        passed = self._compare(
            float(actual),
            condition.operator,
            float(condition.value),
        )

        return ConditionResult(
            condition=condition,
            passed=passed,
            actual_value=float(actual),
            message=(
                f"{timeframe} {indicator} "
                f"{actual:.5f} "
                f"{condition.operator} "
                f"{condition.value}"
            ),
        )

    # ========================================================
    # EMA CROSS
    # ========================================================

    def _evaluate_ema_cross(
        self,
        condition: StrategyCondition,
        df: pd.DataFrame,
        timeframe: str,
    ) -> ConditionResult:

        fast_period = condition.period

        slow_period = int(
            condition.value
        )

        fast_column = (
            f"EMA_{fast_period}"
        )

        slow_column = (
            f"EMA_{slow_period}"
        )

        if (
            fast_column not in df.columns
            or slow_column not in df.columns
        ):

            return ConditionResult(
                condition=condition,
                passed=False,
                message=(
                    f"EMA columns missing on "
                    f"{timeframe}: "
                    f"{fast_column}, "
                    f"{slow_column}"
                ),
            )

        if len(df) < 2:

            return ConditionResult(
                condition=condition,
                passed=False,
                message=(
                    f"Not enough data for "
                    f"EMA cross on {timeframe}"
                ),
            )

        previous = df.iloc[-2]
        current = df.iloc[-1]

        prev_fast = previous[fast_column]
        prev_slow = previous[slow_column]

        curr_fast = current[fast_column]
        curr_slow = current[slow_column]

        if any(
            pd.isna(x)
            for x in [
                prev_fast,
                prev_slow,
                curr_fast,
                curr_slow,
            ]
        ):

            return ConditionResult(
                condition=condition,
                passed=False,
                message=(
                    f"Invalid EMA values on "
                    f"{timeframe}"
                ),
            )

        operator = (
            condition.operator.lower()
        )

        if operator == "cross_above":

            passed = (
                prev_fast <= prev_slow
                and
                curr_fast > curr_slow
            )

        elif operator == "cross_below":

            passed = (
                prev_fast >= prev_slow
                and
                curr_fast < curr_slow
            )

        else:

            passed = False

        return ConditionResult(
            condition=condition,
            passed=passed,
            actual_value=float(curr_fast),
            message=(
                f"{timeframe} EMA "
                f"{fast_period}/{slow_period} "
                f"{operator}: "
                f"{prev_fast:.5f}/{prev_slow:.5f}"
                f" -> "
                f"{curr_fast:.5f}/{curr_slow:.5f}"
            ),
        )

    # ========================================================
    # FIND INDICATOR
    # ========================================================

    def _find_indicator_column(
        self,
        df: pd.DataFrame,
        indicator: str,
        period: int | None,
    ):

        # Standard indicators with periods.

        if period is not None:

            candidates = [
                f"{indicator}_{period}",
                f"{indicator}{period}",
            ]

            for column in candidates:

                if column in df.columns:

                    return column

        # Indicators without explicit period.

        if indicator in df.columns:

            return indicator

        # Try common default period.

        default_periods = {
            "RSI": 14,
            "ADX": 14,
            "CCI": 20,
            "ROC": 12,
            "MOMENTUM": 10,
            "ATR": 14,
            "TSI": 25,
            "WILLIAMS_R": 14,
            "MFI": 14,
            "CMF": 20,
            "RELATIVE_VOLUME": 20,
        }

        default_period = (
            default_periods.get(indicator)
        )

        if default_period:

            candidates = [
                f"{indicator}_{default_period}",
                f"{indicator}{default_period}",
            ]

            for column in candidates:

                if column in df.columns:

                    return column

        return None

    # ========================================================
    # COMPARISON
    # ========================================================

    def _compare(
        self,
        actual: float,
        operator: str,
        target: float,
    ) -> bool:

        operator = operator.lower()

        if operator in {">", "above", "greater_than"}:

            return actual > target

        if operator in {"<", "below", "less_than"}:

            return actual < target

        if operator in {"==", "=", "equals"}:

            return actual == target

        if operator in {">=", "greater_equal"}:

            return actual >= target

        if operator in {"<=", "less_equal"}:

            return actual <= target

        return False