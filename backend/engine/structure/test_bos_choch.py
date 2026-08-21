import pandas as pd

from backend.engine.structure.market_structure import (
    MarketStructureEngine,
)

from backend.engine.structure.bos_choch import (
    BOSCHOCHEngine,
)


if __name__ == "__main__":

    data = pd.DataFrame({

        "high": [
            1.101,
            1.102,
            1.105,
            1.103,
            1.101,
            1.104,
            1.107,
            1.105,
            1.103,
            1.108,
            1.111,
            1.109,
            1.106,
            1.103,
            1.100,
            1.098,
            1.101,
            1.104,
            1.107,
            1.110,
        ],

        "low": [
            1.099,
            1.100,
            1.102,
            1.100,
            1.098,
            1.101,
            1.104,
            1.102,
            1.100,
            1.105,
            1.108,
            1.106,
            1.103,
            1.100,
            1.097,
            1.095,
            1.098,
            1.101,
            1.104,
            1.107,
        ],

        "close": [
            1.100,
            1.101,
            1.104,
            1.101,
            1.099,
            1.103,
            1.106,
            1.103,
            1.101,
            1.107,
            1.110,
            1.107,
            1.104,
            1.101,
            1.098,
            1.096,
            1.100,
            1.103,
            1.106,
            1.109,
        ],

    })


    structure_engine = (
        MarketStructureEngine()
    )

    data = (
        structure_engine
        .calculate_structure(
            data,
            left=2,
            right=2,
        )
    )


    bos_engine = BOSCHOCHEngine()

    result = bos_engine.calculate(
        data
    )


    print("========================================")
    print("       NEXA FUNDS AI")
    print("      BOS / CHOCH ENGINE TEST")
    print("========================================")
    print()

    print("Detected events:")

    events = result[
        result[
            [
                "BULLISH_BOS",
                "BEARISH_BOS",
                "BULLISH_CHOCH",
                "BEARISH_CHOCH",
            ]
        ].any(axis=1)
    ]

    print(
        events[
            [
                "high",
                "low",
                "close",
                "SWING_HIGH",
                "SWING_LOW",
                "BULLISH_BOS",
                "BEARISH_BOS",
                "BULLISH_CHOCH",
                "BEARISH_CHOCH",
                "BROKEN_SWING_HIGH",
                "BROKEN_SWING_LOW",
            ]
        ]
    )

    print()

    print("Bullish BOS count:",
          int(result["BULLISH_BOS"].sum()))

    print("Bearish BOS count:",
          int(result["BEARISH_BOS"].sum()))

    print("Bullish CHOCH count:",
          int(result["BULLISH_CHOCH"].sum()))

    print("Bearish CHOCH count:",
          int(result["BEARISH_CHOCH"].sum()))

    print()

    print(
        "NEXA FUNDS AI BOS / CHOCH ENGINE OK"
    )
