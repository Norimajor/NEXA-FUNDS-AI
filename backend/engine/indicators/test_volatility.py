import numpy as np
import pandas as pd

from backend.engine.feature_engine import FeatureEngine

from backend.engine.indicators.volatility import (
    TrueRange,
    BollingerBands,
    KeltnerChannels,
    DonchianChannels,
    StandardDeviation,
    HistoricalVolatility,
    ATRPercentile,
)


if __name__ == "__main__":

    np.random.seed(42)

    rows = 150

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

    volume = np.random.randint(
        1000,
        5000,
        rows,
    )

    data = pd.DataFrame({

        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,

    })

    engine = FeatureEngine()

    engine.register(TrueRange())
    engine.register(BollingerBands())
    engine.register(KeltnerChannels())
    engine.register(DonchianChannels())
    engine.register(StandardDeviation())
    engine.register(HistoricalVolatility())
    engine.register(ATRPercentile())

    requests = [

        {
            "name": "true_range",
            "output": "TRUE_RANGE",
        },

        {
            "name": "bollinger_bands",
            "parameters": {
                "period": 20,
                "std_dev": 2,
            },
        },

        {
            "name": "keltner_channels",
            "parameters": {
                "ema_period": 20,
                "atr_period": 10,
                "multiplier": 2,
            },
        },

        {
            "name": "donchian_channels",
            "parameters": {
                "period": 20,
            },
        },

        {
            "name": "standard_deviation",
            "parameters": {
                "period": 20,
            },
            "output": "STDDEV_20",
        },

        {
            "name": "historical_volatility",
            "parameters": {
                "period": 20,
                "annualization": 252,
            },
            "output": "HIST_VOL_20",
        },

        {
            "name": "atr_percentile",
            "parameters": {
                "atr_period": 14,
                "percentile_period": 50,
            },
            "output": "ATR_PERCENTILE",
        },

    ]

    result = engine.calculate_all(
        data,
        requests,
    )

    print("========================================")
    print("       NEXA FUNDS AI")
    print("   VOLATILITY ENGINE TEST")
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

    print("NEXA FUNDS AI VOLATILITY ENGINE OK")
