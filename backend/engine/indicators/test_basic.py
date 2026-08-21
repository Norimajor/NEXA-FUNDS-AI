import pandas as pd

from backend.engine.feature_engine import FeatureEngine

from backend.engine.indicators.basic import (
    EMA,
    SMA,
    RSI,
    ATR,
)


if __name__ == "__main__":

    data = pd.DataFrame({

        "open": [
            1.1000,
            1.1008,
            1.1017,
            1.1022,
            1.1028,
        ],

        "high": [
            1.1010,
            1.1020,
            1.1025,
            1.1030,
            1.1035,
        ],

        "low": [
            1.0995,
            1.1005,
            1.1010,
            1.1015,
            1.1020,
        ],

        "close": [
            1.1008,
            1.1017,
            1.1022,
            1.1028,
            1.1032,
        ],

    })

    engine = FeatureEngine()

    engine.register(EMA())
    engine.register(SMA())
    engine.register(RSI())
    engine.register(ATR())

    print("========================================")
    print("       NEXA FUNDS AI")
    print("      FEATURE ENGINE TEST")
    print("========================================")
    print()

    print("Registered features:")
    print(engine.list_features())
    print()

    result = engine.calculate_all(

        data,

        [

            {
                "name": "ema",
                "parameters": {
                    "period": 3,
                },
                "output": "EMA_3",
            },

            {
                "name": "sma",
                "parameters": {
                    "period": 3,
                },
                "output": "SMA_3",
            },

            {
                "name": "rsi",
                "parameters": {
                    "period": 3,
                },
                "output": "RSI_3",
            },

            {
                "name": "atr",
                "parameters": {
                    "period": 3,
                },
                "output": "ATR_3",
            },

        ],
    )

    print("Generated columns:")
    print(
        [
            column
            for column in result.columns
            if column not in data.columns
        ]
    )

    print()

    print(result)

    print()

    print("NEXA FUNDS AI FEATURE ENGINE OK")
