import numpy as np
import pandas as pd

from backend.engine.structure.market_structure import (
    MarketStructureEngine,
)


if __name__ == "__main__":

    data = pd.DataFrame({

        "open": [
            1.1000,
            1.1010,
            1.1020,
            1.1010,
            1.1000,
            1.1010,
            1.1030,
            1.1020,
            1.1010,
            1.1040,
            1.1060,
            1.1050,
            1.1070,
            1.1060,
            1.1080,
        ],

        "high": [
            1.1010,
            1.1020,
            1.1040,
            1.1020,
            1.1010,
            1.1030,
            1.1050,
            1.1040,
            1.1020,
            1.1060,
            1.1080,
            1.1070,
            1.1090,
            1.1080,
            1.1100,
        ],

        "low": [
            1.0990,
            1.1000,
            1.1010,
            1.1000,
            1.0990,
            1.1000,
            1.1020,
            1.1010,
            1.1000,
            1.1030,
            1.1050,
            1.1040,
            1.1060,
            1.1050,
            1.1070,
        ],

        "close": [
            1.1005,
            1.1015,
            1.1030,
            1.1010,
            1.1005,
            1.1025,
            1.1040,
            1.1020,
            1.1015,
            1.1050,
            1.1070,
            1.1055,
            1.1080,
            1.1070,
            1.1090,
        ],

    })


    engine = MarketStructureEngine()

    result = engine.calculate_structure(
        data,
        left=2,
        right=2,
    )

    result = engine.classify_swings(
        result,
    )

    result = engine.market_bias(
        result,
    )


    print("========================================")
    print("       NEXA FUNDS AI")
    print("  MARKET STRUCTURE ENGINE TEST")
    print("========================================")
    print()

    print("Detected columns:")

    print([
        "SWING_HIGH",
        "SWING_LOW",
        "HIGH_STRUCTURE",
        "LOW_STRUCTURE",
        "MARKET_BIAS",
    ])

    print()

    print(result[
        [
            "high",
            "low",
            "close",
            "SWING_HIGH",
            "SWING_LOW",
            "HIGH_STRUCTURE",
            "LOW_STRUCTURE",
            "MARKET_BIAS",
        ]
    ])

    print()

    print("Swing highs:")

    print(
        list(
            result.index[
                result["SWING_HIGH"]
            ]
        )
    )

    print()

    print("Swing lows:")

    print(
        list(
            result.index[
                result["SWING_LOW"]
            ]
        )
    )

    print()

    print("NEXA FUNDS AI MARKET STRUCTURE ENGINE OK")
