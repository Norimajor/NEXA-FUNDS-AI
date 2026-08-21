import numpy as np
import pandas as pd


class LiquidityEngine:

    def __init__(self):

        self.name = "NEXA FUNDS AI Liquidity Engine"


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


    def equal_highs(
        self,
        data,
        tolerance=0.0001,
        lookback=20,
    ):

        highs = data["high"].astype(float)

        result = pd.Series(
            False,
            index=data.index,
        )

        for i in range(len(data)):

            start = max(
                0,
                i - lookback,
            )

            previous = highs.iloc[
                start:i
            ]

            if len(previous) == 0:
                continue

            current = highs.iloc[i]

            distances = (
                abs(previous - current)
            )

            if (
                distances.min()
                <= tolerance
            ):

                result.iloc[i] = True

        return result


    def equal_lows(
        self,
        data,
        tolerance=0.0001,
        lookback=20,
    ):

        lows = data["low"].astype(float)

        result = pd.Series(
            False,
            index=data.index,
        )

        for i in range(len(data)):

            start = max(
                0,
                i - lookback,
            )

            previous = lows.iloc[
                start:i
            ]

            if len(previous) == 0:
                continue

            current = lows.iloc[i]

            distances = (
                abs(previous - current)
            )

            if (
                distances.min()
                <= tolerance
            ):

                result.iloc[i] = True

        return result


    def liquidity_levels(
        self,
        data,
        tolerance=0.0001,
        lookback=20,
    ):

        self.validate_data(data)

        result = data.copy()

        result["EQUAL_HIGH"] = (
            self.equal_highs(
                result,
                tolerance,
                lookback,
            )
        )

        result["EQUAL_LOW"] = (
            self.equal_lows(
                result,
                tolerance,
                lookback,
            )
        )

        return result


    def liquidity_sweeps(
        self,
        data,
        tolerance=0.0001,
        lookback=20,
    ):

        self.validate_data(data)

        result = self.liquidity_levels(
            data,
            tolerance,
            lookback,
        )

        result["BUY_SIDE_SWEEP"] = False
        result["SELL_SIDE_SWEEP"] = False

        result["SWEPT_HIGH"] = np.nan
        result["SWEPT_LOW"] = np.nan


        previous_highs = []

        previous_lows = []


        for i in range(len(result)):

            high = float(
                result["high"].iloc[i]
            )

            low = float(
                result["low"].iloc[i]
            )

            close = float(
                result["close"].iloc[i]
            )


            # -------------------------
            # BUY-SIDE LIQUIDITY
            # -------------------------

            if previous_highs:

                for level in previous_highs:

                    if (
                        high > level
                        and
                        close < level
                    ):

                        result.iloc[
                            i,
                            result.columns.get_loc(
                                "BUY_SIDE_SWEEP"
                            )
                        ] = True

                        result.iloc[
                            i,
                            result.columns.get_loc(
                                "SWEPT_HIGH"
                            )
                        ] = level

                        break


            # -------------------------
            # SELL-SIDE LIQUIDITY
            # -------------------------

            if previous_lows:

                for level in previous_lows:

                    if (
                        low < level
                        and
                        close > level
                    ):

                        result.iloc[
                            i,
                            result.columns.get_loc(
                                "SELL_SIDE_SWEEP"
                            )
                        ] = True

                        result.iloc[
                            i,
                            result.columns.get_loc(
                                "SWEPT_LOW"
                            )
                        ] = level

                        break


            # Store liquidity levels

            if bool(
                result["EQUAL_HIGH"].iloc[i]
            ):

                previous_highs.append(high)


            if bool(
                result["EQUAL_LOW"].iloc[i]
            ):

                previous_lows.append(low)


            # Prevent unlimited memory

            previous_highs = (
                previous_highs[-lookback:]
            )

            previous_lows = (
                previous_lows[-lookback:]
            )


        return result
