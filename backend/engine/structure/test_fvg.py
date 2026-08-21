import pandas as pd

from backend.engine.structure.fvg import (
    FVGEngine,
)


if __name__ == "__main__":

    data = pd.DataFrame({

        "high": [
            1.1010,
            1.1040,
            1.1060,
            1.1080,
            1.1090,
            1.1050,
            1.1030,
            1.1010,
            1.0980,
            1.0960,
            1.0990,
            1.1020,
        ],

        "low": [
            1.0990,
            1.1020,
            1.1050,
            1.1070,
            1.1080,
            1.1030,
            1.1010,
            1.0990,
            1.0950,
            1.0940,
            1.0970,
            1.1000,
        ],

        "close": [
            1.1000,
            1.1030,
            1.1055,
            1.1075,
            1.1085,
            1.1040,
            1.1020,
            1.1000,
            1.0960,
            1.0950,
            1.0980,
            1.1010,
        ],

    })


    engine = FVGEngine()

    result = engine.calculate(
        data
    )


    print("========================================")
    print("       NEXA FUNDS AI")
    print("       FVG ENGINE TEST")
    print("========================================")
    print()

    print("Generated columns:")

    print([
        "BULLISH_FVG",
        "BEARISH_FVG",
        "FVG_LOW",
        "FVG_HIGH",
        "FVG_MID",
        "FVG_SIZE",
        "FVG_DIRECTION",
    ])

    print()

    events = result[
        result[
            [
                "BULLISH_FVG",
                "BEARISH_FVG",
            ]
        ].any(axis=1)
    ]

    print("Detected FVGs:")

    print(
        events[
            [
                "high",
                "low",
                "close",
                "BULLISH_FVG",
                "BEARISH_FVG",
                "FVG_LOW",
                "FVG_HIGH",
                "FVG_MID",
                "FVG_SIZE",
                "FVG_DIRECTION",
            ]
        ]
    )

    print()

    print(
        "Bullish FVGs:",
        int(
            result["BULLISH_FVG"].sum()
        )
    )

    print(
        "Bearish FVGs:",
        int(
            result["BEARISH_FVG"].sum()
        )
    )

    print()

    print(
        "NEXA FUNDS AI FVG ENGINE OK"
    )
