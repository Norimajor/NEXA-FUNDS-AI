import numpy as np
import pandas as pd

from backend.engine.feature_engine import FeatureEngine

from backend.engine.indicators.price_action import (
    CandleBody,
    CandleRange,
    UpperWick,
    LowerWick,
    BodyRangeRatio,
    UpperWickRatio,
    LowerWickRatio,
    CandleDirection,
    InsideBar,
    OutsideBar,
    BullishEngulfing,
    BearishEngulfing,
    Doji,
    PinBar,
    RangeExpansion,
)


if __name__ == "__main__":

    np.random.seed(321)

    rows = 100

    close = (
        1.1000
        + np.cumsum(
            np.random.normal(
                0,
                0.001,
                rows,
            )
        )
    )

    open_price = (
        close
        + np.random.normal(
            0,
            0.0005,
            rows,
        )
    )

    high = np.maximum(
        open_price,
        close,
    ) + np.random.uniform(
        0.0001,
        0.0010,
        rows,
    )

    low = np.minimum(
        open_price,
        close,
    ) - np.random.uniform(
        0.0001,
        0.0010,
        rows,
    )

    data = pd.DataFrame({

        "open": open_price,
        "high": high,
        "low": low,
        "close": close,

    })

    engine = FeatureEngine()

    engine.register(CandleBody())
    engine.register(CandleRange())
    engine.register(UpperWick())
    engine.register(LowerWick())
    engine.register(BodyRangeRatio())
    engine.register(UpperWickRatio())
    engine.register(LowerWickRatio())
    engine.register(CandleDirection())
    engine.register(InsideBar())
    engine.register(OutsideBar())
    engine.register(BullishEngulfing())
    engine.register(BearishEngulfing())
    engine.register(Doji())
    engine.register(PinBar())
    engine.register(RangeExpansion())

    requests = [

        {
            "name": "candle_body",
            "output": "CANDLE_BODY",
        },

        {
            "name": "candle_range",
            "output": "CANDLE_RANGE",
        },

        {
            "name": "upper_wick",
            "output": "UPPER_WICK",
        },

        {
            "name": "lower_wick",
            "output": "LOWER_WICK",
        },

        {
            "name": "body_range_ratio",
            "output": "BODY_RANGE_RATIO",
        },

        {
            "name": "upper_wick_ratio",
            "output": "UPPER_WICK_RATIO",
        },

        {
            "name": "lower_wick_ratio",
            "output": "LOWER_WICK_RATIO",
        },

        {
            "name": "candle_direction",
            "output": "CANDLE_DIRECTION",
        },

        {
            "name": "inside_bar",
            "output": "INSIDE_BAR",
        },

        {
            "name": "outside_bar",
            "output": "OUTSIDE_BAR",
        },

        {
            "name": "bullish_engulfing",
            "output": "BULLISH_ENGULFING",
        },

        {
            "name": "bearish_engulfing",
            "output": "BEARISH_ENGULFING",
        },

        {
            "name": "doji",
            "parameters": {
                "threshold": 0.10,
            },
            "output": "DOJI",
        },

        {
            "name": "pin_bar",
            "parameters": {
                "wick_ratio": 2.0,
            },
            "output": "PIN_BAR",
        },

        {
            "name": "range_expansion",
            "parameters": {
                "period": 20,
            },
            "output": "RANGE_EXPANSION",
        },

    ]

    result = engine.calculate_all(
        data,
        requests,
    )

    print("========================================")
    print("       NEXA FUNDS AI")
    print("   PRICE ACTION ENGINE TEST")
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

    print("NEXA FUNDS AI PRICE ACTION ENGINE OK")
