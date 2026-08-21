import pandas as pd

from backend.engine.structure.order_block import (
    OrderBlockEngine,
)


if __name__ == "__main__":

    data = pd.DataFrame({

        "open": [
            1.1000,
            1.1020,
            1.1010,
            1.1040,
            1.1060,
            1.1050,
            1.1080,
            1.1070,
            1.1040,
            1.1020,
            1.1000,
            1.0980,
        ],

        "high": [
            1.1010,
            1.1030,
            1.1020,
            1.1070,
            1.1070,
            1.1060,
            1.1100,
            1.1080,
            1.1050,
            1.1030,
            1.1010,
            1.0990,
        ],

        "low": [
            1.0990,
            1.1010,
            1.1000,
            1.1030,
            1.1050,
            1.1040,
            1.1070,
            1.1060,
            1.1030,
            1.1010,
            1.0990,
            1.0970,
        ],

        "close": [
            1.1005,
            1.1010,
            1.1015,
            1.1065,
            1.1055,
            1.1045,
            1.1095,
            1.1065,
            1.1040,
            1.1020,
            1.0995,
            1.0975,
        ],

    })


    engine = OrderBlockEngine()

    result = engine.calculate(
        data,
        displacement_multiplier=1.2,
        lookback=3,
    )


    print("========================================")
    print("       NEXA FUNDS AI")
    print("    ORDER BLOCK ENGINE TEST")
    print("========================================")
    print()

    print("Generated columns:")

    print([
        "BULLISH_ORDER_BLOCK",
        "BEARISH_ORDER_BLOCK",
        "OB_HIGH",
        "OB_LOW",
        "OB_MID",
        "OB_SIZE",
        "OB_DIRECTION",
        "OB_DISPLACEMENT",
    ])

    print()

    events = result[
        result[
            [
                "BULLISH_ORDER_BLOCK",
                "BEARISH_ORDER_BLOCK",
            ]
        ].any(axis=1)
    ]

    print("Detected Order Blocks:")

    print(
        events[
            [
                "open",
                "high",
                "low",
                "close",
                "BULLISH_ORDER_BLOCK",
                "BEARISH_ORDER_BLOCK",
                "OB_HIGH",
                "OB_LOW",
                "OB_MID",
                "OB_SIZE",
                "OB_DIRECTION",
                "OB_DISPLACEMENT",
            ]
        ]
    )

    print()

    print(
        "Bullish Order Blocks:",
        int(
            result[
                "BULLISH_ORDER_BLOCK"
            ].sum()
        )
    )

    print(
        "Bearish Order Blocks:",
        int(
            result[
                "BEARISH_ORDER_BLOCK"
            ].sum()
        )
    )

    print()

    print(
        "NEXA FUNDS AI ORDER BLOCK ENGINE OK"
    )
