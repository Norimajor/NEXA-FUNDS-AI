import pandas as pd

from backend.engine.strategies.smc_strategy import (
    SMCStrategyEngine,
)


if __name__ == "__main__":

    data = pd.DataFrame({

        "BUY_SIDE_SWEEP": [
            False,
            False,
            True,
            False,
            False,
        ],

        "SELL_SIDE_SWEEP": [
            False,
            True,
            False,
            True,
            False,
        ],

        "BULLISH_BOS": [
            False,
            False,
            False,
            True,
            False,
        ],

        "BEARISH_BOS": [
            False,
            False,
            True,
            False,
            False,
        ],

        "BULLISH_CHOCH": [
            False,
            True,
            False,
            False,
            False,
        ],

        "BEARISH_CHOCH": [
            False,
            False,
            True,
            False,
            False,
        ],

        "BULLISH_FVG": [
            False,
            True,
            False,
            True,
            False,
        ],

        "BEARISH_FVG": [
            False,
            False,
            True,
            False,
            False,
        ],

        "BULLISH_ORDER_BLOCK": [
            False,
            True,
            False,
            True,
            False,
        ],

        "BEARISH_ORDER_BLOCK": [
            False,
            False,
            True,
            False,
            False,
        ],

        "IN_PREMIUM": [
            False,
            False,
            True,
            False,
            True,
        ],

        "IN_DISCOUNT": [
            False,
            True,
            False,
            True,
            False,
        ],

    })


    engine = SMCStrategyEngine()

    result = engine.calculate(data)


    print("========================================")
    print("       NEXA FUNDS AI")
    print("      SMC STRATEGY TEST")
    print("========================================")
    print()

    print(
        result[
            [
                "SELL_SIDE_SWEEP",
                "BUY_SIDE_SWEEP",
                "BULLISH_CHOCH",
                "BEARISH_CHOCH",
                "BULLISH_FVG",
                "BEARISH_FVG",
                "BULLISH_ORDER_BLOCK",
                "BEARISH_ORDER_BLOCK",
                "IN_DISCOUNT",
                "IN_PREMIUM",
                "BULLISH_SMC_SCORE",
                "BEARISH_SMC_SCORE",
                "BULLISH_SMC_SETUP",
                "BEARISH_SMC_SETUP",
                "SMC_SIGNAL",
            ]
        ]
    )

    print()

    print(
        "BUY signals:",
        int(
            (
                result["SMC_SIGNAL"]
                == "BUY"
            ).sum()
        )
    )

    print(
        "SELL signals:",
        int(
            (
                result["SMC_SIGNAL"]
                == "SELL"
            ).sum()
        )
    )

    print()

    print(
        "NEXA FUNDS AI SMC STRATEGY ENGINE OK"
    )
