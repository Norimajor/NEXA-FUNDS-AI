import pandas as pd


class SMCStrategyEngine:

    def __init__(self):

        self.name = "NEXA FUNDS AI SMC Strategy Engine"


    def validate_data(self, data):

        required = [
            "BUY_SIDE_SWEEP",
            "SELL_SIDE_SWEEP",
            "BULLISH_BOS",
            "BEARISH_BOS",
            "BULLISH_CHOCH",
            "BEARISH_CHOCH",
            "BULLISH_FVG",
            "BEARISH_FVG",
            "BULLISH_ORDER_BLOCK",
            "BEARISH_ORDER_BLOCK",
            "IN_PREMIUM",
            "IN_DISCOUNT",
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


    def calculate(self, data):

        self.validate_data(data)

        result = data.copy()


        # =====================================
        # BULLISH SMC SETUP
        # =====================================

        result["BULLISH_SMC_SETUP"] = (

            result["SELL_SIDE_SWEEP"]

            &

            (
                result["BULLISH_CHOCH"]
                |
                result["BULLISH_BOS"]
            )

            &

            result["BULLISH_FVG"]

            &

            result["BULLISH_ORDER_BLOCK"]

            &

            result["IN_DISCOUNT"]

        )


        # =====================================
        # BEARISH SMC SETUP
        # =====================================

        result["BEARISH_SMC_SETUP"] = (

            result["BUY_SIDE_SWEEP"]

            &

            (
                result["BEARISH_CHOCH"]
                |
                result["BEARISH_BOS"]
            )

            &

            result["BEARISH_FVG"]

            &

            result["BEARISH_ORDER_BLOCK"]

            &

            result["IN_PREMIUM"]

        )


        # =====================================
        # PARTIAL SETUPS
        # =====================================

        result["BULLISH_SMC_SCORE"] = (

            result["SELL_SIDE_SWEEP"].astype(int)

            +

            result[
                [
                    "BULLISH_CHOCH",
                    "BULLISH_BOS",
                ]
            ].any(axis=1).astype(int)

            +

            result["BULLISH_FVG"].astype(int)

            +

            result[
                "BULLISH_ORDER_BLOCK"
            ].astype(int)

            +

            result["IN_DISCOUNT"].astype(int)

        )


        result["BEARISH_SMC_SCORE"] = (

            result["BUY_SIDE_SWEEP"].astype(int)

            +

            result[
                [
                    "BEARISH_CHOCH",
                    "BEARISH_BOS",
                ]
            ].any(axis=1).astype(int)

            +

            result["BEARISH_FVG"].astype(int)

            +

            result[
                "BEARISH_ORDER_BLOCK"
            ].astype(int)

            +

            result["IN_PREMIUM"].astype(int)

        )


        # =====================================
        # FINAL SIGNAL
        # =====================================

        result["SMC_SIGNAL"] = "NONE"


        bullish = (
            result["BULLISH_SMC_SETUP"]
        )

        bearish = (
            result["BEARISH_SMC_SETUP"]
        )


        result.loc[
            bullish,
            "SMC_SIGNAL"
        ] = "BUY"


        result.loc[
            bearish,
            "SMC_SIGNAL"
        ] = "SELL"


        return result
