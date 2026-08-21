import numpy as np
import pandas as pd

from backend.engine.feature_engine import Feature


class TrueRange(Feature):

    name = "true_range"

    def calculate(
        self,
        data: pd.DataFrame,
        parameters: dict | None = None,
    ) -> pd.Series:

        previous_close = data["close"].shift(1)

        return pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - previous_close).abs(),
                (data["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)


class BollingerBands(Feature):

    name = "bollinger_bands"

    def calculate(
        self,
        data: pd.DataFrame,
        parameters: dict | None = None,
    ) -> pd.DataFrame:

        parameters = parameters or {}

        period = int(
            parameters.get("period", 20)
        )

        standard_deviations = float(
            parameters.get("std_dev", 2.0)
        )

        close = data["close"].astype(float)

        middle = close.rolling(period).mean()

        std = close.rolling(period).std(
            ddof=0
        )

        upper = (
            middle
            + standard_deviations * std
        )

        lower = (
            middle
            - standard_deviations * std
        )

        bandwidth = (
            (upper - lower)
            / middle.replace(0, np.nan)
        )

        percent_b = (
            (close - lower)
            / (upper - lower).replace(
                0,
                np.nan,
            )
        )

        return pd.DataFrame({
            "BB_UPPER": upper,
            "BB_MIDDLE": middle,
            "BB_LOWER": lower,
            "BB_WIDTH": bandwidth,
            "BB_PERCENT_B": percent_b,
        })


class KeltnerChannels(Feature):

    name = "keltner_channels"

    def calculate(
        self,
        data: pd.DataFrame,
        parameters: dict | None = None,
    ) -> pd.DataFrame:

        parameters = parameters or {}

        ema_period = int(
            parameters.get("ema_period", 20)
        )

        atr_period = int(
            parameters.get("atr_period", 10)
        )

        multiplier = float(
            parameters.get("multiplier", 2.0)
        )

        close = data["close"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)

        previous_close = close.shift(1)

        true_range = pd.concat(
            [
                high - low,
                (high - previous_close).abs(),
                (low - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = true_range.ewm(
            alpha=1 / atr_period,
            adjust=False,
        ).mean()

        middle = close.ewm(
            span=ema_period,
            adjust=False,
        ).mean()

        upper = (
            middle + multiplier * atr
        )

        lower = (
            middle - multiplier * atr
        )

        return pd.DataFrame({
            "KC_UPPER": upper,
            "KC_MIDDLE": middle,
            "KC_LOWER": lower,
        })


class DonchianChannels(Feature):

    name = "donchian_channels"

    def calculate(
        self,
        data: pd.DataFrame,
        parameters: dict | None = None,
    ) -> pd.DataFrame:

        parameters = parameters or {}

        period = int(
            parameters.get("period", 20)
        )

        upper = (
            data["high"]
            .rolling(period)
            .max()
        )

        lower = (
            data["low"]
            .rolling(period)
            .min()
        )

        middle = (
            upper + lower
        ) / 2

        width = (
            upper - lower
        )

        return pd.DataFrame({
            "DONCHIAN_UPPER": upper,
            "DONCHIAN_MIDDLE": middle,
            "DONCHIAN_LOWER": lower,
            "DONCHIAN_WIDTH": width,
        })


class StandardDeviation(Feature):

    name = "standard_deviation"

    def calculate(
        self,
        data: pd.DataFrame,
        parameters: dict | None = None,
    ) -> pd.Series:

        parameters = parameters or {}

        period = int(
            parameters.get("period", 20)
        )

        return (
            data["close"]
            .astype(float)
            .rolling(period)
            .std(ddof=0)
        )


class HistoricalVolatility(Feature):

    name = "historical_volatility"

    def calculate(
        self,
        data: pd.DataFrame,
        parameters: dict | None = None,
    ) -> pd.Series:

        parameters = parameters or {}

        period = int(
            parameters.get("period", 20)
        )

        annualization = float(
            parameters.get(
                "annualization",
                252,
            )
        )

        returns = np.log(
            data["close"]
            / data["close"].shift(1)
        )

        return (
            returns
            .rolling(period)
            .std(ddof=0)
            * np.sqrt(annualization)
        )


class ATRPercentile(Feature):

    name = "atr_percentile"

    def calculate(
        self,
        data: pd.DataFrame,
        parameters: dict | None = None,
    ) -> pd.Series:

        parameters = parameters or {}

        atr_period = int(
            parameters.get("atr_period", 14)
        )

        percentile_period = int(
            parameters.get(
                "percentile_period",
                100,
            )
        )

        previous_close = data["close"].shift(1)

        true_range = pd.concat(
            [
                data["high"] - data["low"],
                (
                    data["high"]
                    - previous_close
                ).abs(),
                (
                    data["low"]
                    - previous_close
                ).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = true_range.ewm(
            alpha=1 / atr_period,
            adjust=False,
        ).mean()

        def percentile_rank(values):

            current = values[-1]

            return (
                100
                * np.mean(
                    values <= current
                )
            )

        return (
            atr
            .rolling(percentile_period)
            .apply(
                percentile_rank,
                raw=True,
            )
        )
