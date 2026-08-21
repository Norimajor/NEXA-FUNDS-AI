import numpy as np
import pandas as pd

from backend.engine.feature_engine import Feature


class Returns(Feature):

    name = "returns"

    def calculate(self, data, parameters=None):

        return data["close"].astype(float).pct_change()


class LogReturns(Feature):

    name = "log_returns"

    def calculate(self, data, parameters=None):

        close = data["close"].astype(float)

        return np.log(
            close / close.shift(1)
        )


class RollingMean(Feature):

    name = "rolling_mean"

    def calculate(self, data, parameters=None):

        parameters = parameters or {}

        period = int(
            parameters.get("period", 20)
        )

        return (
            data["close"]
            .astype(float)
            .rolling(period)
            .mean()
        )


class RollingMedian(Feature):

    name = "rolling_median"

    def calculate(self, data, parameters=None):

        parameters = parameters or {}

        period = int(
            parameters.get("period", 20)
        )

        return (
            data["close"]
            .astype(float)
            .rolling(period)
            .median()
        )


class RollingVariance(Feature):

    name = "rolling_variance"

    def calculate(self, data, parameters=None):

        parameters = parameters or {}

        period = int(
            parameters.get("period", 20)
        )

        return (
            data["close"]
            .astype(float)
            .rolling(period)
            .var()
        )


class RollingStd(Feature):

    name = "rolling_std"

    def calculate(self, data, parameters=None):

        parameters = parameters or {}

        period = int(
            parameters.get("period", 20)
        )

        return (
            data["close"]
            .astype(float)
            .rolling(period)
            .std()
        )


class RollingSkew(Feature):

    name = "rolling_skew"

    def calculate(self, data, parameters=None):

        parameters = parameters or {}

        period = int(
            parameters.get("period", 20)
        )

        return (
            data["close"]
            .astype(float)
            .rolling(period)
            .skew()
        )


class RollingKurtosis(Feature):

    name = "rolling_kurtosis"

    def calculate(self, data, parameters=None):

        parameters = parameters or {}

        period = int(
            parameters.get("period", 20)
        )

        return (
            data["close"]
            .astype(float)
            .rolling(period)
            .kurt()
        )


class ZScore(Feature):

    name = "zscore"

    def calculate(self, data, parameters=None):

        parameters = parameters or {}

        period = int(
            parameters.get("period", 20)
        )

        close = data["close"].astype(float)

        mean = (
            close
            .rolling(period)
            .mean()
        )

        std = (
            close
            .rolling(period)
            .std()
        )

        return (
            close - mean
        ) / std.replace(
            0,
            np.nan,
        )


class PricePercentile(Feature):

    name = "price_percentile"

    def calculate(self, data, parameters=None):

        parameters = parameters or {}

        period = int(
            parameters.get("period", 50)
        )

        close = data["close"].astype(float)

        def percentile(values):

            return (
                100
                * np.mean(
                    values <= values[-1]
                )
            )

        return (
            close
            .rolling(period)
            .apply(
                percentile,
                raw=True,
            )
        )


class ReturnPercentile(Feature):

    name = "return_percentile"

    def calculate(self, data, parameters=None):

        parameters = parameters or {}

        period = int(
            parameters.get("period", 50)
        )

        returns = (
            data["close"]
            .astype(float)
            .pct_change()
        )

        def percentile(values):

            return (
                100
                * np.mean(
                    values <= values[-1]
                )
            )

        return (
            returns
            .rolling(period)
            .apply(
                percentile,
                raw=True,
            )
        )


class Autocorrelation(Feature):

    name = "autocorrelation"

    def calculate(self, data, parameters=None):

        parameters = parameters or {}

        period = int(
            parameters.get("period", 20)
        )

        lag = int(
            parameters.get("lag", 1)
        )

        returns = (
            data["close"]
            .astype(float)
            .pct_change()
        )

        def autocorr(values):

            series = pd.Series(values)

            return series.autocorr(
                lag=lag
            )

        return (
            returns
            .rolling(period)
            .apply(
                autocorr,
                raw=False,
            )
        )


class RollingHigh(Feature):

    name = "rolling_high"

    def calculate(self, data, parameters=None):

        parameters = parameters or {}

        period = int(
            parameters.get("period", 20)
        )

        return (
            data["high"]
            .astype(float)
            .rolling(period)
            .max()
        )


class RollingLow(Feature):

    name = "rolling_low"

    def calculate(self, data, parameters=None):

        parameters = parameters or {}

        period = int(
            parameters.get("period", 20)
        )

        return (
            data["low"]
            .astype(float)
            .rolling(period)
            .min()
        )


class DistanceFromHigh(Feature):

    name = "distance_from_high"

    def calculate(self, data, parameters=None):

        parameters = parameters or {}

        period = int(
            parameters.get("period", 20)
        )

        rolling_high = (
            data["high"]
            .astype(float)
            .rolling(period)
            .max()
        )

        return (
            data["close"]
            - rolling_high
        )


class DistanceFromLow(Feature):

    name = "distance_from_low"

    def calculate(self, data, parameters=None):

        parameters = parameters or {}

        period = int(
            parameters.get("period", 20)
        )

        rolling_low = (
            data["low"]
            .astype(float)
            .rolling(period)
            .min()
        )

        return (
            data["close"]
            - rolling_low
        )
