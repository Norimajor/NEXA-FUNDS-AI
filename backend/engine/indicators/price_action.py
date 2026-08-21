import numpy as np
import pandas as pd

from backend.engine.feature_engine import Feature


class CandleBody(Feature):

    name = "candle_body"

    def calculate(self, data, parameters=None):

        return (
            data["close"]
            - data["open"]
        )


class CandleRange(Feature):

    name = "candle_range"

    def calculate(self, data, parameters=None):

        return (
            data["high"]
            - data["low"]
        )


class UpperWick(Feature):

    name = "upper_wick"

    def calculate(self, data, parameters=None):

        body_high = np.maximum(
            data["open"],
            data["close"],
        )

        return (
            data["high"]
            - body_high
        )


class LowerWick(Feature):

    name = "lower_wick"

    def calculate(self, data, parameters=None):

        body_low = np.minimum(
            data["open"],
            data["close"],
        )

        return (
            body_low
            - data["low"]
        )


class BodyRangeRatio(Feature):

    name = "body_range_ratio"

    def calculate(self, data, parameters=None):

        body = (
            data["close"]
            - data["open"]
        ).abs()

        candle_range = (
            data["high"]
            - data["low"]
        ).replace(0, np.nan)

        return body / candle_range


class UpperWickRatio(Feature):

    name = "upper_wick_ratio"

    def calculate(self, data, parameters=None):

        body_high = np.maximum(
            data["open"],
            data["close"],
        )

        wick = (
            data["high"]
            - body_high
        )

        candle_range = (
            data["high"]
            - data["low"]
        ).replace(0, np.nan)

        return wick / candle_range


class LowerWickRatio(Feature):

    name = "lower_wick_ratio"

    def calculate(self, data, parameters=None):

        body_low = np.minimum(
            data["open"],
            data["close"],
        )

        wick = (
            body_low
            - data["low"]
        )

        candle_range = (
            data["high"]
            - data["low"]
        ).replace(0, np.nan)

        return wick / candle_range


class CandleDirection(Feature):

    name = "candle_direction"

    def calculate(self, data, parameters=None):

        return np.sign(
            data["close"]
            - data["open"]
        )


class InsideBar(Feature):

    name = "inside_bar"

    def calculate(self, data, parameters=None):

        previous_high = data["high"].shift(1)
        previous_low = data["low"].shift(1)

        return (
            (
                data["high"]
                <= previous_high
            )
            &
            (
                data["low"]
                >= previous_low
            )
        ).astype(int)


class OutsideBar(Feature):

    name = "outside_bar"

    def calculate(self, data, parameters=None):

        previous_high = data["high"].shift(1)
        previous_low = data["low"].shift(1)

        return (
            (
                data["high"]
                > previous_high
            )
            &
            (
                data["low"]
                < previous_low
            )
        ).astype(int)


class BullishEngulfing(Feature):

    name = "bullish_engulfing"

    def calculate(self, data, parameters=None):

        previous_open = data["open"].shift(1)
        previous_close = data["close"].shift(1)

        previous_bearish = (
            previous_close
            < previous_open
        )

        current_bullish = (
            data["close"]
            > data["open"]
        )

        engulfing = (
            data["open"]
            <= previous_close
        ) & (
            data["close"]
            >= previous_open
        )

        return (
            previous_bearish
            & current_bullish
            & engulfing
        ).astype(int)


class BearishEngulfing(Feature):

    name = "bearish_engulfing"

    def calculate(self, data, parameters=None):

        previous_open = data["open"].shift(1)
        previous_close = data["close"].shift(1)

        previous_bullish = (
            previous_close
            > previous_open
        )

        current_bearish = (
            data["close"]
            < data["open"]
        )

        engulfing = (
            data["open"]
            >= previous_close
        ) & (
            data["close"]
            <= previous_open
        )

        return (
            previous_bullish
            & current_bearish
            & engulfing
        ).astype(int)


class Doji(Feature):

    name = "doji"

    def calculate(self, data, parameters=None):

        parameters = parameters or {}

        threshold = float(
            parameters.get(
                "threshold",
                0.10,
            )
        )

        body = (
            data["close"]
            - data["open"]
        ).abs()

        candle_range = (
            data["high"]
            - data["low"]
        ).replace(0, np.nan)

        return (
            body / candle_range
            <= threshold
        ).astype(int)


class PinBar(Feature):

    name = "pin_bar"

    def calculate(self, data, parameters=None):

        parameters = parameters or {}

        wick_ratio = float(
            parameters.get(
                "wick_ratio",
                2.0,
            )
        )

        body = (
            data["close"]
            - data["open"]
        ).abs()

        upper = (
            data["high"]
            - np.maximum(
                data["open"],
                data["close"],
            )
        )

        lower = (
            np.minimum(
                data["open"],
                data["close"],
            )
            - data["low"]
        )

        bullish_pin = (
            lower >= body * wick_ratio
        )

        bearish_pin = (
            upper >= body * wick_ratio
        )

        return (
            bullish_pin
            | bearish_pin
        ).astype(int)


class RangeExpansion(Feature):

    name = "range_expansion"

    def calculate(self, data, parameters=None):

        parameters = parameters or {}

        period = int(
            parameters.get(
                "period",
                20,
            )
        )

        candle_range = (
            data["high"]
            - data["low"]
        )

        average_range = (
            candle_range
            .rolling(period)
            .mean()
        )

        return (
            candle_range
            / average_range.replace(
                0,
                np.nan,
            )
        )
