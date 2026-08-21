import pandas as pd

from backend.engine.structure.liquidity import (
    LiquidityEngine,
)


if __name__ == "__main__":

    data = pd.DataFrame({

        "high": [
            1.1010,
            1.1030,
            1.1050,
            1.1030,
            1.1050,
            1.1040,
            1.1060,
            1.1050,
            1.1080,
            1.1070,
            1.1055,
            1.1090,
            1.1080,
            1.1060,
            1.1030,
        ],

        "low": [
            1.0990,
            1.1000,
            1.1020,
            1.1000,
            1.1020,
            1.1010,
            1.1030,
            1.1020,
            1.1050,
            1.1040,
            1.1020,
            1.1060,
            1.1050,
            1.1030,
            1.1000,
        ],

        "close": [
            1.1005,
            1.1020,
            1.1040,
            1.1020,
            1.1045,
            1.1030,
            1.1050,
            1.1040,
            1.1070,
            1.1060,
            1.1030,
            1.1080,
            1.1070,
            1.1040,
            1.1010,
        ],

    })


    engine = LiquidityEngine()

    result = engine.liquidity_sweeps(
        data,
        tolerance=0.0002,
        lookback=10,
    )


    print("========================================")
    print("       NEXA FUNDS AI")
    print("       LIQUIDITY ENGINE TEST")
    print("========================================")
    print()

    print("Generated columns:")

    print([
        "EQUAL_HIGH",
        "EQUAL_LOW",
        "BUY_SIDE_SWEEP",
        "SELL_SIDE_SWEEP",
        "SWEPT_HIGH",
        "SWEPT_LOW",
    ])

    print()

    print("Detected liquidity events:")

    events = result[
        result[
            [
                "EQUAL_HIGH",
                "EQUAL_LOW",
                "BUY_SIDE_SWEEP",
                "SELL_SIDE_SWEEP",
            ]
        ].any(axis=1)
    ]

    print(
        events[
            [
                "high",
                "low",
                "close",
                "EQUAL_HIGH",
                "EQUAL_LOW",
                "BUY_SIDE_SWEEP",
                "SELL_SIDE_SWEEP",
                "SWEPT_HIGH",
                "SWEPT_LOW",
            ]
        ]
    )

    print()

    print(
        "Equal highs:",
        int(
            result["EQUAL_HIGH"].sum()
        )
    )

    print(
        "Equal lows:",
        int(
            result["EQUAL_LOW"].sum()
        )
    )

    print(
        "Buy-side sweeps:",
        int(
            result["BUY_SIDE_SWEEP"].sum()
        )
    )

    print(
        "Sell-side sweeps:",
        int(
            result["SELL_SIDE_SWEEP"].sum()
        )
    )

    print()

    print(
        "NEXA FUNDS AI LIQUIDITY ENGINE OK"
    )
