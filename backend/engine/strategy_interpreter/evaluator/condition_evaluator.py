import pandas as pd


class ConditionEvaluator:

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()

    def evaluate_condition(self, condition):

        indicator = condition.indicator.upper()
        operator = condition.operator
        value = condition.value
        period = condition.period

        if indicator == "EMA_CROSS":
            return self._evaluate_ema_cross(condition)

        column = self._resolve_column(
            indicator,
            period
        )

        if column is None:
            raise ValueError(
                f"Indicator '{indicator}' "
                f"with period '{period}' "
                f"not found in dataframe"
            )

        series = self.df[column]

        if operator == ">":
            return pd.to_numeric(
                series,
                errors="coerce"
            ) > value

        if operator == "<":
            return pd.to_numeric(
                series,
                errors="coerce"
            ) < value

        if operator == ">=":
            return pd.to_numeric(
                series,
                errors="coerce"
            ) >= value

        if operator == "<=":
            return pd.to_numeric(
                series,
                errors="coerce"
            ) <= value

        if operator == "==":
            return series == value

        if operator == "!=":
            return series != value

        if operator == "cross_above":
            return self._cross_above(
                pd.to_numeric(series, errors="coerce"),
                value
            )

        if operator == "cross_below":
            return self._cross_below(
                pd.to_numeric(series, errors="coerce"),
                value
            )

        if operator in {"true", "false"}:
            expected = operator == "true"

            if series.dtype == bool:
                return series == expected

            return (
                series.astype(str)
                .str.lower()
                .isin(
                    ["true", "1", "yes"]
                    if expected
                    else ["false", "0", "no"]
                )
            )

        if operator in {
            "bullish",
            "bearish",
            "buy_side",
            "sell_side",
            "premium",
            "discount",
        }:

            column_name = self._resolve_special_column(
                indicator,
                operator
            )

            if column_name is None:
                raise ValueError(
                    f"No column available for "
                    f"{indicator} {operator}"
                )

            return self.df[column_name].astype(bool)

        raise ValueError(
            f"Unsupported operator '{operator}'"
        )

    def _evaluate_ema_cross(self, condition):

        fast_period = condition.period
        slow_period = int(condition.value)

        fast_column = self._resolve_column(
            "EMA",
            fast_period
        )

        slow_column = self._resolve_column(
            "EMA",
            slow_period
        )

        if fast_column is None:
            raise ValueError(
                f"Fast EMA {fast_period} not found"
            )

        if slow_column is None:
            raise ValueError(
                f"Slow EMA {slow_period} not found"
            )

        fast = pd.to_numeric(
            self.df[fast_column],
            errors="coerce"
        )

        slow = pd.to_numeric(
            self.df[slow_column],
            errors="coerce"
        )

        previous_fast = fast.shift(1)
        previous_slow = slow.shift(1)

        if condition.operator == "cross_above":
            return (
                (previous_fast <= previous_slow)
                &
                (fast > slow)
            )

        if condition.operator == "cross_below":
            return (
                (previous_fast >= previous_slow)
                &
                (fast < slow)
            )

        raise ValueError(
            f"Unsupported EMA cross operator: "
            f"{condition.operator}"
        )

    def evaluate_strategy(self, conditions):

        if not conditions:
            return pd.Series(
                False,
                index=self.df.index
            )

        result = pd.Series(
            True,
            index=self.df.index
        )

        for condition in conditions:

            condition_result = (
                self.evaluate_condition(condition)
            )

            result &= (
                condition_result
                .fillna(False)
            )

        return result

    def evaluate_last(self, conditions):

        signals = {}
        all_true = True

        for condition in conditions:

            result = self.evaluate_condition(
                condition
            )

            name = self._condition_name(
                condition
            )

            value = bool(
                result.iloc[-1]
            )

            signals[name] = value

            if not value:
                all_true = False

        signals["FINAL_SIGNAL"] = (
            "BUY"
            if all_true
            else "NONE"
        )

        return signals

    def _resolve_column(
        self,
        indicator,
        period
    ):

        aliases = {
            "RSI": "RSI",
            "ADX": "ADX",
            "ATR": "ATR",
            "CCI": "CCI",
            "ROC": "ROC",
            "MOMENTUM": "MOMENTUM",
            "EMA": "EMA",
            "SMA": "SMA",
            "WMA": "WMA",
            "HMA": "HMA",
            "DEMA": "DEMA",
            "TEMA": "TEMA",
            "VWMA": "VWMA",
            "STOCHASTIC": "STOCH_K",
            "WILLIAMS_R": "WILLIAMS_R",
            "TSI": "TSI",
            "VWAP": "VWAP",
            "OBV": "OBV",
            "MFI": "MFI",
            "CMF": "CMF",
            "RELATIVE_VOLUME": "RELATIVE_VOLUME",
        }

        base = aliases.get(indicator)

        if base is None:
            return None

        candidates = []

        if period is not None:
            candidates.extend([
                f"{base}_{period}",
                f"{base}{period}",
            ])

        candidates.append(base)

        lower_columns = {
            str(column).lower(): column
            for column in self.df.columns
        }

        for candidate in candidates:

            if candidate in self.df.columns:
                return candidate

            if candidate.lower() in lower_columns:
                return lower_columns[
                    candidate.lower()
                ]

        return None

    def _resolve_special_column(
        self,
        indicator,
        operator
    ):

        candidates = [
            f"{operator}_{indicator}",
            f"{indicator}_{operator}",
            operator.upper(),
            f"{operator.upper()}_{indicator}",
            f"{indicator}_{operator.upper()}",
        ]

        lower_columns = {
            str(column).lower(): column
            for column in self.df.columns
        }

        for candidate in candidates:

            if candidate in self.df.columns:
                return candidate

            if candidate.lower() in lower_columns:
                return lower_columns[candidate.lower()]

        return None

    @staticmethod
    def _cross_above(series, level):

        previous = series.shift(1)

        return (
            (previous <= level)
            &
            (series > level)
        )

    @staticmethod
    def _cross_below(series, level):

        previous = series.shift(1)

        return (
            (previous >= level)
            &
            (series < level)
        )

    @staticmethod
    def _condition_name(condition):

        if condition.period is not None:
            return (
                f"{condition.indicator}_"
                f"{condition.period}_"
                f"{condition.operator}_"
                f"{condition.value}"
            )

        return (
            f"{condition.indicator}_"
            f"{condition.operator}_"
            f"{condition.value}"
        )
