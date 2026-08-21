import numpy as np
import pandas as pd

from backend.engine.feature_engine import Feature


def _wma(series: pd.Series, period: int) -> pd.Series:

    weights = np.arange(
        1,
        period + 1,
        dtype=float,
    )

    weight_sum = weights.sum()

    return series.rolling(
        period
    ).apply(
        lambda values: (
            values * weights
        ).sum() / weight_sum,
        raw=True,
    )


class WMA(Feature):

    name = "wma"

    def calculate(
        self,
        data: pd.DataFrame,
        parameters: dict | None = None,
    ) -> pd.Series:

        parameters = parameters or {}

        period = int(
            parameters.get("period", 20)
        )

        return _wma(
            data["close"].astype(float),
            period,
        )


class HMA(Feature):

    name = "hma"

    def calculate(
        self,
        data: pd.DataFrame,
        parameters: dict | None = None,
    ) -> pd.Series:

        parameters = parameters or {}

        period = int(
            parameters.get("period", 20)
        )

        half_period = max(
            1,
            period // 2,
        )

        sqrt_period = max(
            1,
            int(period ** 0.5),
        )

        half_wma = _wma(
            data["close"].astype(float),
            half_period,
        )

        full_wma = _wma(
            data["close"].astype(float),
            period,
        )

        raw_hma = (
            2 * half_wma
            - full_wma
        )

        return _wma(
            raw_hma,
            sqrt_period,
        )


class DEMA(Feature):

    name = "dema"

    def calculate(
        self,
        data: pd.DataFrame,
        parameters: dict | None = None,
    ) -> pd.Series:

        parameters = parameters or {}

        period = int(
            parameters.get("period", 20)
        )

        close = data["close"].astype(float)

        ema1 = close.ewm(
            span=period,
            adjust=False,
        ).mean()

        ema2 = ema1.ewm(
            span=period,
            adjust=False,
        ).mean()

        return (
            2 * ema1
            - ema2
        )


class TEMA(Feature):

    name = "tema"

    def calculate(
        self,
        data: pd.DataFrame,
        parameters: dict | None = None,
    ) -> pd.Series:

        parameters = parameters or {}

        period = int(
            parameters.get("period", 20)
        )

        close = data["close"].astype(float)

        ema1 = close.ewm(
            span=period,
            adjust=False,
        ).mean()

        ema2 = ema1.ewm(
            span=period,
            adjust=False,
        ).mean()

        ema3 = ema2.ewm(
            span=period,
            adjust=False,
        ).mean()

        return (
            3 * ema1
            - 3 * ema2
            + ema3
        )


class VWMA(Feature):

    name = "vwma"

    def calculate(
        self,
        data: pd.DataFrame,
        parameters: dict | None = None,
    ) -> pd.Series:

        parameters = parameters or {}

        period = int(
            parameters.get("period", 20)
        )

        if "volume" not in data.columns:
            raise ValueError(
                "VWMA requires a volume column."
            )

        close = data["close"].astype(float)

        volume = data["volume"].astype(float)

        price_volume = (
            close * volume
        )

        volume_sum = volume.rolling(
            period
        ).sum()

        return (
            price_volume.rolling(
                period
            ).sum()
            / volume_sum.replace(
                0,
                np.nan,
            )
        )


class ADX(Feature):

    name = "adx"

    def calculate(
        self,
        data: pd.DataFrame,
        parameters: dict | None = None,
    ) -> pd.Series:

        parameters = parameters or {}

        period = int(
            parameters.get("period", 14)
        )

        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)

        previous_high = high.shift(1)
        previous_low = low.shift(1)
        previous_close = close.shift(1)

        up_move = (
            high - previous_high
        )

        down_move = (
            previous_low - low
        )

        plus_dm = up_move.where(
            (up_move > down_move)
            & (up_move > 0),
            0.0,
        )

        minus_dm = down_move.where(
            (down_move > up_move)
            & (down_move > 0),
            0.0,
        )

        true_range = pd.concat(
            [
                high - low,
                (high - previous_close).abs(),
                (low - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = true_range.ewm(
            alpha=1 / period,
            adjust=False,
        ).mean()

        plus_dm_smoothed = plus_dm.ewm(
            alpha=1 / period,
            adjust=False,
        ).mean()

        minus_dm_smoothed = minus_dm.ewm(
            alpha=1 / period,
            adjust=False,
        ).mean()

        plus_di = (
            100
            * plus_dm_smoothed
            / atr.replace(
                0,
                np.nan,
            )
        )

        minus_di = (
            100
            * minus_dm_smoothed
            / atr.replace(
                0,
                np.nan,
            )
        )

        denominator = (
            plus_di + minus_di
        )

        dx = (
            100
            * (plus_di - minus_di).abs()
            / denominator.replace(
                0,
                np.nan,
            )
        )

        return dx.ewm(
            alpha=1 / period,
            adjust=False,
        ).mean()
