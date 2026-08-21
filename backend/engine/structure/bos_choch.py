import pandas as pd


class BOSCHOCHEngine:

    def __init__(self):

        self.name = "NEXA FUNDS AI BOS CHOCH Engine"


    def validate_data(self, data):

        required = [
            "high",
            "low",
            "close",
            "SWING_HIGH",
            "SWING_LOW",
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

        result["BULLISH_BOS"] = False
        result["BEARISH_BOS"] = False

        result["BULLISH_CHOCH"] = False
        result["BEARISH_CHOCH"] = False

        result["BROKEN_SWING_HIGH"] = None
        result["BROKEN_SWING_LOW"] = None

        last_swing_high = None
        last_swing_low = None

        previous_bias = "NEUTRAL"


        for i in range(len(result)):

            row = result.iloc[i]

            close = float(row["close"])


            # Register confirmed swing levels
            if bool(row["SWING_HIGH"]):

                last_swing_high = float(
                    row["high"]
                )


            if bool(row["SWING_LOW"]):

                last_swing_low = float(
                    row["low"]
                )


            # Bullish break
            if (
                last_swing_high is not None
                and close > last_swing_high
            ):

                result.iloc[
                    i,
                    result.columns.get_loc(
                        "BULLISH_BOS"
                    )
                ] = True

                result.iloc[
                    i,
                    result.columns.get_loc(
                        "BROKEN_SWING_HIGH"
                    )
                ] = last_swing_high


                if previous_bias == "BEARISH":

                    result.iloc[
                        i,
                        result.columns.get_loc(
                            "BULLISH_CHOCH"
                        )
                    ] = True


                previous_bias = "BULLISH"

                last_swing_high = None


            # Bearish break
            if (
                last_swing_low is not None
                and close < last_swing_low
            ):

                result.iloc[
                    i,
                    result.columns.get_loc(
                        "BEARISH_BOS"
                    )
                ] = True

                result.iloc[
                    i,
                    result.columns.get_loc(
                        "BROKEN_SWING_LOW"
                    )
                ] = last_swing_low


                if previous_bias == "BULLISH":

                    result.iloc[
                        i,
                        result.columns.get_loc(
                            "BEARISH_CHOCH"
                        )
                    ] = True


                previous_bias = "BEARISH"

                last_swing_low = None


        return result
