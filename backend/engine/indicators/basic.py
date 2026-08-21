import pandas as pd

from backend.engine.feature_engine import Feature


class EMA(Feature):

    name = "ema"

    def calculate(
        self,
        data: pd.DataFrame,
        parameters: dict | None = None,
    ) -> pd.Series:

        parameters = parameters or {}

        period = int(
            parameters.get("period", 20)
        )

        return data["close"].ewm(
            span=period,
            adjust=False,
        ).mean()


class SMA(Feature):

    name = "sma"

    def calculate(
        self,
        data: pd.DataFrame,
        parameters: dict | None = None,
    ) -> pd.Series:

        parameters = parameters or {}

        period = int(
            parameters.get("period", 20)
        )

        return data["close"].rolling(
            period
        ).mean()


class RSI(Feature):

    name = "rsi"

    def calculate(
        self,
        data: pd.DataFrame,
        parameters: dict | None = None,
    ) -> pd.Series:

        parameters = parameters or {}

        period = int(
            parameters.get("period", 14)
        )

        delta = data["close"].diff()

        gain = delta.clip(
            lower=0
        )

        loss = -delta.clip(
            upper=0
        )

        average_gain = gain.ewm(
            alpha=1 / period,
            adjust=False,
        ).mean()

        average_loss = loss.ewm(
            alpha=1 / period,
            adjust=False,
        ).mean()

        rs = average_gain / average_loss

        return 100 - (
            100 / (1 + rs)
        )


class ATR(Feature):

    name = "atr"

    def calculate(
        self,
        data: pd.DataFrame,
        parameters: dict | None = None,
    ) -> pd.Series:

        parameters = parameters or {}

        period = int(
            parameters.get("period", 14)
        )

        previous_close = data["close"].shift(1)

        high_low = (
            data["high"] - data["low"]
        )

        high_previous_close = (
            data["high"] - previous_close
        ).abs()

        low_previous_close = (
            data["low"] - previous_close
        ).abs()

        true_range = pd.concat(
            [
                high_low,
                high_previous_close,
                low_previous_close,
            ],
            axis=1,
        ).max(axis=1)

        return true_range.ewm(
            alpha=1 / period,
            adjust=False,
        ).mean()
