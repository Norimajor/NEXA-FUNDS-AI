import pandas as pd

from backend.engine.structure.premium_discount import (
    PremiumDiscountEngine,
)


if __name__ == "__main__":

    data = pd.DataFrame({

        "high": [
            1.101,
            1.103,
            1.105,
            1.107,
            1.110,
            1.109,
            1.108,
            1.106,
            1.105,
            1.104,
            1.103,
            1.102,
        ],

        "low": [
            1.099,
            1.100,
            1.101,
            1.103,
            1.106,
            1.105,
            1.104,
            1.102,
            1.101,
            1.100,
            1.099,
            1.098,
        ],

        "close": [
            1.100,
            1.102,
            1.104,
            1.106,
            1.109,
            1.107,
            1.105,
            1.103,
            1.102,
            1.101,
            1.100,
            1.099,
        ],

    })


    engine = PremiumDiscountEngine()

    result = engine.calculate(
        data,
        lookback=5,
    )


    print("========================================")
    print("       NEXA FUNDS AI")
    print(" PREMIUM / DISCOUNT ENGINE TEST")
    print("========================================")
    print()

    print("Generated columns:")

    print([
        "RANGE_HIGH",
        "RANGE_LOW",
        "EQUILIBRIUM",
        "PREMIUM_75",
        "DISCOUNT_25",
        "PRICE_POSITION",
        "ZONE",
        "DISTANCE_FROM_EQUILIBRIUM",
        "IN_PREMIUM",
        "IN_DISCOUNT",
    ])

    print()

    print(result.tail(10).to_string())

    print()

    print(
        "Premium candles:",
        int(
            result["IN_PREMIUM"].sum()
        )
    )

    print(
        "Discount candles:",
        int(
            result["IN_DISCOUNT"].sum()
        )
    )

    print()

    print(
        "NEXA FUNDS AI PREMIUM / DISCOUNT ENGINE OK"
    )
