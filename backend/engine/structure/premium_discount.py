import numpy as np
import pandas as pd


class PremiumDiscountEngine:

    def __init__(self):

        self.name = "NEXA FUNDS AI Premium Discount Engine"


    def validate_data(self, data):

        required = [
            "high",
            "low",
            "close",
        ]

        missing = [
            column
            for column in required
            if column not in data.columns
        ]

        if missing:

            raise ValueError(
                f"Missing required columns: {missing}"
            )


    def calculate(
        self,
        data,
        lookback=20,
    ):

        self.validate_data(data)

        result = data.copy()

        result["RANGE_HIGH"] = (
            result["high"]
            .rolling(
                lookback,
                min_periods=1,
            )
            .max()
        )

        result["RANGE_LOW"] = (
            result["low"]
            .rolling(
                lookback,
                min_periods=1,
            )
            .min()
        )

        result["EQUILIBRIUM"] = (
            result["RANGE_HIGH"]
            +
            result["RANGE_LOW"]
        ) / 2


        range_size = (
            result["RANGE_HIGH"]
            -
            result["RANGE_LOW"]
        )

        result["PREMIUM_75"] = (
            result["RANGE_LOW"]
            +
            range_size * 0.75
        )

        result["DISCOUNT_25"] = (
            result["RANGE_LOW"]
            +
            range_size * 0.25
        )


        result["PRICE_POSITION"] = (
            (
                result["close"]
                -
                result["RANGE_LOW"]
            )
            /
            range_size.replace(
                0,
                np.nan,
            )
        ) * 100


        result["ZONE"] = np.select(

            [
                result["PRICE_POSITION"] >= 75,

                result["PRICE_POSITION"] >= 50,

                result["PRICE_POSITION"] >= 25,

                result["PRICE_POSITION"] < 25,
            ],

            [
                "PREMIUM_EXTREME",

                "PREMIUM",

                "DISCOUNT",

                "DISCOUNT_EXTREME",
            ],

            default="EQUILIBRIUM",
        )


        result["DISTANCE_FROM_EQUILIBRIUM"] = (
            result["close"]
            -
            result["EQUILIBRIUM"]
        )


        result["IN_PREMIUM"] = (
            result["PRICE_POSITION"] > 50
        )

        result["IN_DISCOUNT"] = (
            result["PRICE_POSITION"] < 50
        )


        return result
