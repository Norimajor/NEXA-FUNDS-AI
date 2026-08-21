import numpy as np
import pandas as pd


class FVGEngine:

    def __init__(self):

        self.name = "NEXA FUNDS AI Fair Value Gap Engine"


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


    def calculate(self, data):

        self.validate_data(data)

        result = data.copy()

        result["BULLISH_FVG"] = False
        result["BEARISH_FVG"] = False

        result["FVG_LOW"] = np.nan
        result["FVG_HIGH"] = np.nan

        result["FVG_MID"] = np.nan
        result["FVG_SIZE"] = np.nan

        result["FVG_DIRECTION"] = None


        for i in range(2, len(result)):

            candle_1_high = float(
                result["high"].iloc[i - 2]
            )

            candle_1_low = float(
                result["low"].iloc[i - 2]
            )

            candle_3_high = float(
                result["high"].iloc[i]
            )

            candle_3_low = float(
                result["low"].iloc[i]
            )


            # -------------------------
            # BULLISH FVG
            # Candle 3 low > Candle 1 high
            # -------------------------

            if candle_3_low > candle_1_high:

                fvg_low = candle_1_high
                fvg_high = candle_3_low

                result.iloc[
                    i,
                    result.columns.get_loc(
                        "BULLISH_FVG"
                    )
                ] = True

                result.iloc[
                    i,
                    result.columns.get_loc(
                        "FVG_LOW"
                    )
                ] = fvg_low

                result.iloc[
                    i,
                    result.columns.get_loc(
                        "FVG_HIGH"
                    )
                ] = fvg_high

                result.iloc[
                    i,
                    result.columns.get_loc(
                        "FVG_MID"
                    )
                ] = (
                    fvg_low + fvg_high
                ) / 2

                result.iloc[
                    i,
                    result.columns.get_loc(
                        "FVG_SIZE"
                    )
                ] = (
                    fvg_high - fvg_low
                )

                result.iloc[
                    i,
                    result.columns.get_loc(
                        "FVG_DIRECTION"
                    )
                ] = "BULLISH"


            # -------------------------
            # BEARISH FVG
            # Candle 3 high < Candle 1 low
            # -------------------------

            elif candle_3_high < candle_1_low:

                fvg_low = candle_3_high
                fvg_high = candle_1_low

                result.iloc[
                    i,
                    result.columns.get_loc(
                        "BEARISH_FVG"
                    )
                ] = True

                result.iloc[
                    i,
                    result.columns.get_loc(
                        "FVG_LOW"
                    )
                ] = fvg_low

                result.iloc[
                    i,
                    result.columns.get_loc(
                        "FVG_HIGH"
                    )
                ] = fvg_high

                result.iloc[
                    i,
                    result.columns.get_loc(
                        "FVG_MID"
                    )
                ] = (
                    fvg_low + fvg_high
                ) / 2

                result.iloc[
                    i,
                    result.columns.get_loc(
                        "FVG_SIZE"
                    )
                ] = (
                    fvg_high - fvg_low
                )

                result.iloc[
                    i,
                    result.columns.get_loc(
                        "FVG_DIRECTION"
                    )
                ] = "BEARISH"


        return result
