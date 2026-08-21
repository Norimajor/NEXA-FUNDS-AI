import numpy as np
import pandas as pd


class MarketStructureEngine:

    def __init__(self):
        self.name = "NEXA FUNDS AI Market Structure Engine"

    # ============================================================
    # VALIDATION
    # ============================================================

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

    # ============================================================
    # SWING HIGHS
    # ============================================================

    def swing_highs(
        self,
        data,
        left=2,
        right=2,
    ):

        high = data["high"].astype(float)

        result = pd.Series(
            False,
            index=data.index,
        )

        for i in range(
            left,
            len(data) - right,
        ):

            current = high.iloc[i]

            left_values = high.iloc[
                i - left:i
            ]

            right_values = high.iloc[
                i + 1:i + right + 1
            ]

            if (
                current > left_values.max()
                and
                current >= right_values.max()
            ):

                result.iloc[i] = True

        return result

    # ============================================================
    # SWING LOWS
    # ============================================================

    def swing_lows(
        self,
        data,
        left=2,
        right=2,
    ):

        low = data["low"].astype(float)

        result = pd.Series(
            False,
            index=data.index,
        )

        for i in range(
            left,
            len(data) - right,
        ):

            current = low.iloc[i]

            left_values = low.iloc[
                i - left:i
            ]

            right_values = low.iloc[
                i + 1:i + right + 1
            ]

            if (
                current < left_values.min()
                and
                current <= right_values.min()
            ):

                result.iloc[i] = True

        return result

    # ============================================================
    # STRUCTURE DETECTION
    # ============================================================

    def calculate_structure(
        self,
        data,
        left=2,
        right=2,
    ):

        self.validate_data(data)

        result = data.copy()

        result["SWING_HIGH"] = self.swing_highs(
            result,
            left,
            right,
        )

        result["SWING_LOW"] = self.swing_lows(
            result,
            left,
            right,
        )

        return result

    # ============================================================
    # CLASSIFY SWINGS
    # ============================================================

    def classify_swings(
        self,
        data,
    ):

        self.validate_data(data)

        result = data.copy()

        if "SWING_HIGH" not in result.columns:
            result["SWING_HIGH"] = self.swing_highs(result)

        if "SWING_LOW" not in result.columns:
            result["SWING_LOW"] = self.swing_lows(result)

        result["HIGH_STRUCTURE"] = None
        result["LOW_STRUCTURE"] = None

        # --------------------------------------------------------
        # HIGH STRUCTURE
        # --------------------------------------------------------

        previous_high = None

        for index, row in result[
            result["SWING_HIGH"]
        ].iterrows():

            current_high = float(
                row["high"]
            )

            if previous_high is not None:

                if current_high > previous_high:

                    result.loc[
                        index,
                        "HIGH_STRUCTURE",
                    ] = "HH"

                elif current_high < previous_high:

                    result.loc[
                        index,
                        "HIGH_STRUCTURE",
                    ] = "LH"

            previous_high = current_high

        # --------------------------------------------------------
        # LOW STRUCTURE
        # --------------------------------------------------------

        previous_low = None

        for index, row in result[
            result["SWING_LOW"]
        ].iterrows():

            current_low = float(
                row["low"]
            )

            if previous_low is not None:

                if current_low > previous_low:

                    result.loc[
                        index,
                        "LOW_STRUCTURE",
                    ] = "HL"

                elif current_low < previous_low:

                    result.loc[
                        index,
                        "LOW_STRUCTURE",
                    ] = "LL"

            previous_low = current_low

        return result

    # ============================================================
    # MARKET BIAS
    # ============================================================

    def market_bias(
        self,
        data,
    ):

        self.validate_data(data)

        result = data.copy()

        if "HIGH_STRUCTURE" not in result.columns:
            result = self.classify_swings(result)

        result["MARKET_BIAS"] = "NEUTRAL"

        latest_high_structure = None
        latest_low_structure = None

        for i in range(len(result)):

            high_structure = result[
                "HIGH_STRUCTURE"
            ].iloc[i]

            low_structure = result[
                "LOW_STRUCTURE"
            ].iloc[i]

            # ----------------------------------------------------
            # Remember latest confirmed structure
            # ----------------------------------------------------

            if high_structure in ("HH", "LH"):
                latest_high_structure = high_structure

            if low_structure in ("HL", "LL"):
                latest_low_structure = low_structure

            # ----------------------------------------------------
            # Determine directional bias
            # ----------------------------------------------------

            bullish_structure = (
                latest_high_structure == "HH"
                and
                latest_low_structure == "HL"
            )

            bearish_structure = (
                latest_high_structure == "LH"
                and
                latest_low_structure == "LL"
            )

            if bullish_structure:

                result.iloc[
                    i,
                    result.columns.get_loc(
                        "MARKET_BIAS"
                    )
                ] = "BULLISH"

            elif bearish_structure:

                result.iloc[
                    i,
                    result.columns.get_loc(
                        "MARKET_BIAS"
                    )
                ] = "BEARISH"

            else:

                result.iloc[
                    i,
                    result.columns.get_loc(
                        "MARKET_BIAS"
                    )
                ] = "NEUTRAL"

        return result

    # ============================================================
    # COMPLETE ENGINE
    # ============================================================

    def calculate(
        self,
        data,
        left=2,
        right=2,
    ):

        result = self.calculate_structure(
            data,
            left,
            right,
        )

        result = self.classify_swings(
            result,
        )

        result = self.market_bias(
            result,
        )

        return result