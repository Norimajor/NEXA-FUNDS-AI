import numpy as np
import pandas as pd

from backend.engine.feature_engine import Feature


class Stochastic(Feature):

    name = "stochastic"

    def calculate(
        self,
        data: pd.DataFrame,
        parameters: dict | None = None,
    ) -> pd.DataFrame:

        parameters = parameters or {}

        period = int(parameters.get("period", 14))
        smooth_k = int(parameters.get("smooth_k", 3))
        smooth_d = int(parameters.get("smooth_d", 3))

        lowest_low = data["low"].rolling(period).min()
        highest_high = data["high"].rolling(period).max()

        denominator = (
            highest_high - lowest_low
        ).replace(0, np.nan)

        raw_k = (
            100
            * (data["close"] - lowest_low)
            / denominator
        )

        k = raw_k.rolling(smooth_k).mean()
        d = k.rolling(smooth_d).mean()

        return pd.DataFrame({
            "STOCH_K": k,
            "STOCH_D": d,
        })


class CCI(Feature):

    name = "cci"

    def calculate(
        self,
        data: pd.DataFrame,
        parameters: dict | None = None,
    ) -> pd.Series:

        parameters = parameters or {}

        period = int(parameters.get("period", 20))

        typical_price = (
            data["high"]
            + data["low"]
            + data["close"]
        ) / 3

        moving_average = (
            typical_price.rolling(period).mean()
        )

        mean_deviation = (
            typical_price
            .rolling(period)
            .apply(
                lambda values: np.mean(
                    np.abs(
                        values - values.mean()
                    )
                ),
                raw=True,
            )
        )

        denominator = (
            0.015 * mean_deviation
        ).replace(0, np.nan)

        return (
            typical_price - moving_average
        ) / denominator


class WilliamsR(Feature):

    name = "williams_r"

    def calculate(
        self,
        data: pd.DataFrame,
        parameters: dict | None = None,
    ) -> pd.Series:

        parameters = parameters or {}

        period = int(parameters.get("period", 14))

        highest_high = (
            data["high"]
            .rolling(period)
            .max()
        )

        lowest_low = (
            data["low"]
            .rolling(period)
            .min()
        )

        denominator = (
            highest_high - lowest_low
        ).replace(0, np.nan)

        return (
            -100
            * (highest_high - data["close"])
            / denominator
        )


class ROC(Feature):

    name = "roc"

    def calculate(
        self,
        data: pd.DataFrame,
        parameters: dict | None = None,
    ) -> pd.Series:

        parameters = parameters or {}

        period = int(parameters.get("period", 12))

        previous = data["close"].shift(period)

        return (
            (
                data["close"] - previous
            )
            / previous.replace(0, np.nan)
        ) * 100


class Momentum(Feature):

    name = "momentum"

    def calculate(
        self,
        data: pd.DataFrame,
        parameters: dict | None = None,
    ) -> pd.Series:

        parameters = parameters or {}

        period = int(parameters.get("period", 10))

        return (
            data["close"]
            - data["close"].shift(period)
        )


class TSI(Feature):

    name = "tsi"

    def calculate(
        self,
        data: pd.DataFrame,
        parameters: dict | None = None,
    ) -> pd.Series:

        parameters = parameters or {}

        long_period = int(
            parameters.get("long_period", 25)
        )

        short_period = int(
            parameters.get("short_period", 13)
        )

        momentum = data["close"].diff()

        abs_momentum = momentum.abs()

        smooth_momentum = (
            momentum
            .ewm(
                span=long_period,
                adjust=False,
            )
            .mean()
            .ewm(
                span=short_period,
                adjust=False,
            )
            .mean()
        )

        smooth_abs_momentum = (
            abs_momentum
            .ewm(
                span=long_period,
                adjust=False,
            )
            .mean()
            .ewm(
                span=short_period,
                adjust=False,
            )
            .mean()
        )

        return (
            100
            * smooth_momentum
            / smooth_abs_momentum.replace(
                0,
                np.nan,
            )
        )


class AwesomeOscillator(Feature):

    name = "awesome_oscillator"

    def calculate(
        self,
        data: pd.DataFrame,
        parameters: dict | None = None,
    ) -> pd.Series:

        parameters = parameters or {}

        fast_period = int(
            parameters.get("fast_period", 5)
        )

        slow_period = int(
            parameters.get("slow_period", 34)
        )

        median_price = (
            data["high"] + data["low"]
        ) / 2

        fast = (
            median_price
            .rolling(fast_period)
            .mean()
        )

        slow = (
            median_price
            .rolling(slow_period)
            .mean()
        )

        return fast - slow


class UltimateOscillator(Feature):

    name = "ultimate_oscillator"

    def calculate(
        self,
        data: pd.DataFrame,
        parameters: dict | None = None,
    ) -> pd.Series:

        parameters = parameters or {}

        p1 = int(parameters.get("period_1", 7))
        p2 = int(parameters.get("period_2", 14))
        p3 = int(parameters.get("period_3", 28))

        previous_close = data["close"].shift(1)

        buying_pressure = (
            data["close"]
            - pd.concat(
                [
                    data["low"],
                    previous_close,
                ],
                axis=1,
            ).min(axis=1)
        )

        true_range = pd.concat(
            [
                data["high"],
                previous_close,
            ],
            axis=1,
        ).max(axis=1) - pd.concat(
            [
                data["low"],
                previous_close,
            ],
            axis=1,
        ).min(axis=1)

        true_range = true_range.replace(
            0,
            np.nan,
        )

        avg1 = (
            buying_pressure.rolling(p1).sum()
            / true_range.rolling(p1).sum()
        )

        avg2 = (
            buying_pressure.rolling(p2).sum()
            / true_range.rolling(p2).sum()
        )

        avg3 = (
            buying_pressure.rolling(p3).sum()
            / true_range.rolling(p3).sum()
        )

        return 100 * (
            4 * avg1
            + 2 * avg2
            + avg3
        ) / 7


class PPO(Feature):

    name = "ppo"

    def calculate(
        self,
        data: pd.DataFrame,
        parameters: dict | None = None,
    ) -> pd.Series:

        parameters = parameters or {}

        fast_period = int(
            parameters.get("fast_period", 12)
        )

        slow_period = int(
            parameters.get("slow_period", 26)
        )

        fast = data["close"].ewm(
            span=fast_period,
            adjust=False,
        ).mean()

        slow = data["close"].ewm(
            span=slow_period,
            adjust=False,
        ).mean()

        return (
            100
            * (fast - slow)
            / slow.replace(0, np.nan)
        )
