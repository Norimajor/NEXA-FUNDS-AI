import numpy as np
import pandas as pd

from backend.engine.feature_engine import FeatureEngine

from backend.engine.indicators.statistics import (
    Returns,
    LogReturns,
    RollingMean,
    RollingMedian,
    RollingVariance,
    RollingStd,
    RollingSkew,
    RollingKurtosis,
    ZScore,
    PricePercentile,
    ReturnPercentile,
    Autocorrelation,
    RollingHigh,
    RollingLow,
    DistanceFromHigh,
    DistanceFromLow,
)


if __name__ == "__main__":

    np.random.seed(555)

    rows = 200

    returns = np.random.normal(
        0,
        0.001,
        rows,
    )

    close = (
        1.1000
        * np.exp(
            np.cumsum(returns)
        )
    )

    high = (
        close
        + np.random.uniform(
            0.0002,
            0.0010,
            rows,
        )
    )

    low = (
        close
        - np.random.uniform(
            0.0002,
            0.0010,
            rows,
        )
    )

    open_price = (
        close
        + np.random.normal(
            0,
            0.0003,
            rows,
        )
    )

    data = pd.DataFrame({

        "open": open_price,
        "high": high,
        "low": low,
        "close": close,

    })

    engine = FeatureEngine()

    engine.register(Returns())
    engine.register(LogReturns())
    engine.register(RollingMean())
    engine.register(RollingMedian())
    engine.register(RollingVariance())
    engine.register(RollingStd())
    engine.register(RollingSkew())
    engine.register(RollingKurtosis())
    engine.register(ZScore())
    engine.register(PricePercentile())
    engine.register(ReturnPercentile())
    engine.register(Autocorrelation())
    engine.register(RollingHigh())
    engine.register(RollingLow())
    engine.register(DistanceFromHigh())
    engine.register(DistanceFromLow())

    requests = [

        {
            "name": "returns",
            "output": "RETURNS",
        },

        {
            "name": "log_returns",
            "output": "LOG_RETURNS",
        },

        {
            "name": "rolling_mean",
            "parameters": {
                "period": 20,
            },
            "output": "MEAN_20",
        },

        {
            "name": "rolling_median",
            "parameters": {
                "period": 20,
            },
            "output": "MEDIAN_20",
        },

        {
            "name": "rolling_variance",
            "parameters": {
                "period": 20,
            },
            "output": "VARIANCE_20",
        },

        {
            "name": "rolling_std",
            "parameters": {
                "period": 20,
            },
            "output": "STD_20",
        },

        {
            "name": "rolling_skew",
            "parameters": {
                "period": 20,
            },
            "output": "SKEW_20",
        },

        {
            "name": "rolling_kurtosis",
            "parameters": {
                "period": 20,
            },
            "output": "KURTOSIS_20",
        },

        {
            "name": "zscore",
            "parameters": {
                "period": 20,
            },
            "output": "ZSCORE_20",
        },

        {
            "name": "price_percentile",
            "parameters": {
                "period": 50,
            },
            "output": "PRICE_PERCENTILE",
        },

        {
            "name": "return_percentile",
            "parameters": {
                "period": 50,
            },
            "output": "RETURN_PERCENTILE",
        },

        {
            "name": "autocorrelation",
            "parameters": {
                "period": 30,
                "lag": 1,
            },
            "output": "AUTOCORRELATION",
        },

        {
            "name": "rolling_high",
            "parameters": {
                "period": 20,
            },
            "output": "ROLLING_HIGH",
        },

        {
            "name": "rolling_low",
            "parameters": {
                "period": 20,
            },
            "output": "ROLLING_LOW",
        },

        {
            "name": "distance_from_high",
            "parameters": {
                "period": 20,
            },
            "output": "DISTANCE_FROM_HIGH",
        },

        {
            "name": "distance_from_low",
            "parameters": {
                "period": 20,
            },
            "output": "DISTANCE_FROM_LOW",
        },

    ]

    result = engine.calculate_all(
        data,
        requests,
    )

    print("========================================")
    print("       NEXA FUNDS AI")
    print("   STATISTICAL ENGINE TEST")
    print("========================================")
    print()

    print("Registered features:")
    print(engine.list_features())
    print()

    print("Generated columns:")

    print(
        [
            column
            for column in result.columns
            if column not in data.columns
        ]
    )

    print()

    print("Last 10 rows:")

    print(result.tail(10))

    print()

    print("NEXA FUNDS AI STATISTICAL ENGINE OK")
