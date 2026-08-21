from typing import Dict

import pandas as pd

from backend.engine.strategy_interpreter.evaluator.condition_evaluator import (
    ConditionEvaluator,
)


class MTFConditionEvaluator:
    """
    Evaluates strategy conditions across multiple timeframes.

    Example:

        H4 EMA_20 crosses above EMA_50
        H1 ADX > 20
        M30 RSI > 50
        M15 EMA_20 crosses above EMA_50
        M5 RSI > 50

    Each condition is evaluated against the dataframe
    belonging to its own timeframe.
    """

    SUPPORTED_TIMEFRAMES = {
        "H4",
        "H1",
        "M30",
        "M15",
        "M5",
        "M1",
    }

    DEFAULT_TIMEFRAME = "M15"

    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
    ):

        self.data = {}

        for timeframe, dataframe in data.items():

            timeframe = timeframe.upper()

            if timeframe not in self.SUPPORTED_TIMEFRAMES:
                continue

            if dataframe is None:
                continue

            if dataframe.empty:
                continue

            self.data[timeframe] = dataframe

    # ========================================================
    # SINGLE CONDITION
    # ========================================================

    def evaluate_condition(
        self,
        condition,
    ):

        timeframe = getattr(
            condition,
            "timeframe",
            None,
        )

        if timeframe is None:
            timeframe = self.DEFAULT_TIMEFRAME

        timeframe = timeframe.upper()

        if timeframe not in self.SUPPORTED_TIMEFRAMES:

            raise ValueError(
                f"Unsupported timeframe: "
                f"{timeframe}"
            )

        if timeframe not in self.data:

            raise ValueError(
                f"No data available for "
                f"timeframe {timeframe}"
            )

        dataframe = self.data[timeframe]

        evaluator = ConditionEvaluator(
            dataframe
        )

        return evaluator.evaluate_condition(
            condition
        )

    # ========================================================
    # FULL STRATEGY
    # ========================================================

    def evaluate_strategy(
        self,
        conditions,
    ):

        if not conditions:

            # Use the first available dataframe
            # to create an appropriately indexed
            # result.

            if not self.data:

                return pd.Series(
                    dtype=bool
                )

            dataframe = next(
                iter(self.data.values())
            )

            return pd.Series(
                False,
                index=dataframe.index,
            )

        results = []

        for condition in conditions:

            result = self.evaluate_condition(
                condition
            )

            # We only need the latest candle
            # when conditions belong to different
            # timeframes.

            results.append(
                bool(
                    result.iloc[-1]
                )
            )

        return all(results)

    # ========================================================
    # LATEST CONDITION RESULTS
    # ========================================================

    def evaluate_last(
        self,
        conditions,
    ):

        signals = {}

        all_true = True

        for condition in conditions:

            timeframe = getattr(
                condition,
                "timeframe",
                None,
            )

            if timeframe is None:
                timeframe = self.DEFAULT_TIMEFRAME

            timeframe = timeframe.upper()

            result = self.evaluate_condition(
                condition
            )

            latest_value = bool(
                result.iloc[-1]
            )

            name = self._condition_name(
                condition
            )

            signals[name] = {
                "timeframe": timeframe,
                "passed": latest_value,
            }

            if not latest_value:
                all_true = False

        signals["FINAL_SIGNAL"] = (
            "BUY"
            if all_true
            else "NONE"
        )

        return signals

    # ========================================================
    # CONDITION STATUS
    # ========================================================

    def condition_status(
        self,
        conditions,
    ):

        status = []

        for condition in conditions:

            timeframe = getattr(
                condition,
                "timeframe",
                None,
            )

            if timeframe is None:
                timeframe = self.DEFAULT_TIMEFRAME

            timeframe = timeframe.upper()

            result = self.evaluate_condition(
                condition
            )

            passed = bool(
                result.iloc[-1]
            )

            status.append({
                "timeframe": timeframe,
                "indicator": condition.indicator,
                "operator": condition.operator,
                "value": condition.value,
                "period": condition.period,
                "passed": passed,
            })

        return status

    # ========================================================
    # CONDITION NAME
    # ========================================================

    @staticmethod
    def _condition_name(
        condition,
    ):

        timeframe = getattr(
            condition,
            "timeframe",
            None,
        )

        if timeframe is None:
            timeframe = "M15"

        if condition.period is not None:

            return (
                f"{timeframe}_"
                f"{condition.indicator}_"
                f"{condition.period}_"
                f"{condition.operator}_"
                f"{condition.value}"
            )

        return (
            f"{timeframe}_"
            f"{condition.indicator}_"
            f"{condition.operator}_"
            f"{condition.value}"
        )