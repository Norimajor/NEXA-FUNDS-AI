import numpy as np
import pandas as pd


class OrderBlockEngine:

    def __init__(self):

        self.name = "NEXA FUNDS AI Order Block Engine"


    def validate_data(self, data):

        required = [
            "open",
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
        displacement_multiplier=1.5,
        lookback=5,
    ):

        self.validate_data(data)

        result = data.copy()

        # Basic candle range

        result["_RANGE"] = (
            result["high"].astype(float)
            -
            result["low"].astype(float)
        )

        result["_BODY"] = abs(
            result["close"].astype(float)
            -
            result["open"].astype(float)
        )

        result["_AVG_RANGE"] = (
            result["_RANGE"]
            .rolling(
                lookback,
                min_periods=1,
            )
            .mean()
        )


        result["BULLISH_ORDER_BLOCK"] = False
        result["BEARISH_ORDER_BLOCK"] = False

        result["OB_HIGH"] = np.nan
        result["OB_LOW"] = np.nan
        result["OB_MID"] = np.nan
        result["OB_SIZE"] = np.nan

        result["OB_DIRECTION"] = None

        result["OB_DISPLACEMENT"] = np.nan


        for i in range(1, len(result)):

            previous_open = float(
                result["open"].iloc[i - 1]
            )

            previous_close = float(
                result["close"].iloc[i - 1]
            )

            previous_high = float(
                result["high"].iloc[i - 1]
            )

            previous_low = float(
                result["low"].iloc[i - 1]
            )

            current_open = float(
                result["open"].iloc[i]
            )

            current_close = float(
                result["close"].iloc[i]
            )

            current_high = float(
                result["high"].iloc[i]
            )

            current_low = float(
                result["low"].iloc[i]
            )

            current_range = float(
                result["_RANGE"].iloc[i]
            )

            average_range = float(
                result["_AVG_RANGE"].iloc[i]
            )


            if average_range <= 0:
                continue


            displacement = (
                current_range
                /
                average_range
            )


            # ==================================
            # BULLISH ORDER BLOCK
            # ==================================
            #
            # Previous candle bearish
            # Current candle bullish
            # Current candle has significant range
            #

            previous_bearish = (
                previous_close
                <
                previous_open
            )

            current_bullish = (
                current_close
                >
                current_open
            )


            if (
                previous_bearish
                and
                current_bullish
                and
                displacement
                >=
                displacement_multiplier
            ):

                ob_high = previous_high
                ob_low = previous_low

                result.iloc[
                    i,
                    result.columns.get_loc(
                        "BULLISH_ORDER_BLOCK"
                    )
                ] = True

                result.iloc[
                    i,
                    result.columns.get_loc(
                        "OB_HIGH"
                    )
                ] = ob_high

                result.iloc[
                    i,
                    result.columns.get_loc(
                        "OB_LOW"
                    )
                ] = ob_low

                result.iloc[
                    i,
                    result.columns.get_loc(
                        "OB_MID"
                    )
                ] = (
                    ob_high + ob_low
                ) / 2

                result.iloc[
                    i,
                    result.columns.get_loc(
                        "OB_SIZE"
                    )
                ] = (
                    ob_high - ob_low
                )

                result.iloc[
                    i,
                    result.columns.get_loc(
                        "OB_DIRECTION"
                    )
                ] = "BULLISH"

                result.iloc[
                    i,
                    result.columns.get_loc(
                        "OB_DISPLACEMENT"
                    )
                ] = displacement


            # ==================================
            # BEARISH ORDER BLOCK
            # ==================================

            previous_bullish = (
                previous_close
                >
                previous_open
            )

            current_bearish = (
                current_close
                <
                current_open
            )


            if (
                previous_bullish
                and
                current_bearish
                and
                displacement
                >=
                displacement_multiplier
            ):

                ob_high = previous_high
                ob_low = previous_low

                result.iloc[
                    i,
                    result.columns.get_loc(
                        "BEARISH_ORDER_BLOCK"
                    )
                ] = True

                result.iloc[
                    i,
                    result.columns.get_loc(
                        "OB_HIGH"
                    )
                ] = ob_high

                result.iloc[
                    i,
                    result.columns.get_loc(
                        "OB_LOW"
                    )
                ] = ob_low

                result.iloc[
                    i,
                    result.columns.get_loc(
                        "OB_MID"
                    )
                ] = (
                    ob_high + ob_low
                ) / 2

                result.iloc[
                    i,
                    result.columns.get_loc(
                        "OB_SIZE"
                    )
                ] = (
                    ob_high - ob_low
                )

                result.iloc[
                    i,
                    result.columns.get_loc(
                        "OB_DIRECTION"
                    )
                ] = "BEARISH"

                result.iloc[
                    i,
                    result.columns.get_loc(
                        "OB_DISPLACEMENT"
                    )
                ] = displacement


        result.drop(
            columns=[
                "_RANGE",
                "_BODY",
                "_AVG_RANGE",
            ],
            inplace=True,
        )

        return result
