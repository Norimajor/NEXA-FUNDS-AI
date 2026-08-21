import numpy as np
import pandas as pd

from backend.engine.feature_engine import Feature


class OBV(Feature):

    name = "obv"

    def calculate(
        self,
        data: pd.DataFrame,
        parameters: dict | None = None,
    ) -> pd.Series:

        close = data["close"].astype(float)
        volume = data["volume"].astype(float)

        direction = np.sign(
            close.diff()
        ).fillna(0)

        return (
            direction * volume
        ).cumsum()


class VWAP(Feature):

    name = "vwap"

    def calculate(
        self,
        data: pd.DataFrame,
        parameters: dict | None = None,
    ) -> pd.Series:

        parameters = parameters or {}

        period = parameters.get("period")

        typical_price = (
            data["high"]
            + data["low"]
            + data["close"]
        ) / 3

        price_volume = (
            typical_price
            * data["volume"].astype(float)
        )

        if period is None:

            cumulative_volume = (
                data["volume"]
                .astype(float)
                .cumsum()
            )

            return (
                price_volume.cumsum()
                / cumulative_volume.replace(
                    0,
                    np.nan,
                )
            )

        period = int(period)

        return (
            price_volume.rolling(period).sum()
            /
            data["volume"]
            .astype(float)
            .rolling(period)
            .sum()
            .replace(0, np.nan)
        )


class MFI(Feature):

    name = "mfi"

    def calculate(
        self,
        data: pd.DataFrame,
        parameters: dict | None = None,
    ) -> pd.Series:

        parameters = parameters or {}

        period = int(
            parameters.get("period", 14)
        )

        typical_price = (
            data["high"]
            + data["low"]
            + data["close"]
        ) / 3

        money_flow = (
            typical_price
            * data["volume"].astype(float)
        )

        direction = typical_price.diff()

        positive_flow = money_flow.where(
            direction > 0,
            0.0,
        )

        negative_flow = money_flow.where(
            direction < 0,
            0.0,
        )

        positive_sum = (
            positive_flow
            .rolling(period)
            .sum()
        )

        negative_sum = (
            negative_flow
            .rolling(period)
            .sum()
        )

        money_ratio = (
            positive_sum
            / negative_sum.replace(
                0,
                np.nan,
            )
        )

        return (
            100
            - (
                100
                / (1 + money_ratio)
            )
        )


class CMF(Feature):

    name = "cmf"

    def calculate(
        self,
        data: pd.DataFrame,
        parameters: dict | None = None,
    ) -> pd.Series:

        parameters = parameters or {}

        period = int(
            parameters.get("period", 20)
        )

        denominator = (
            data["high"]
            - data["low"]
        ).replace(
            0,
            np.nan,
        )

        money_flow_multiplier = (
            (
                (
                    data["close"]
                    - data["low"]
                )
                - (
                    data["high"]
                    - data["close"]
                )
            )
            / denominator
        )

        money_flow_volume = (
            money_flow_multiplier
            * data["volume"].astype(float)
        )

        return (
            money_flow_volume
            .rolling(period)
            .sum()
            /
            data["volume"]
            .astype(float)
            .rolling(period)
            .sum()
            .replace(0, np.nan)
        )


class ADLine(Feature):

    name = "ad_line"

    def calculate(
        self,
        data: pd.DataFrame,
        parameters: dict | None = None,
    ) -> pd.Series:

        denominator = (
            data["high"]
            - data["low"]
        ).replace(
            0,
            np.nan,
        )

        multiplier = (
            (
                (
                    data["close"]
                    - data["low"]
                )
                - (
                    data["high"]
                    - data["close"]
                )
            )
            / denominator
        )

        money_flow_volume = (
            multiplier
            * data["volume"].astype(float)
        )

        return money_flow_volume.cumsum()


class ChaikinOscillator(Feature):

    name = "chaikin_oscillator"

    def calculate(
        self,
        data: pd.DataFrame,
        parameters: dict | None = None,
    ) -> pd.Series:

        parameters = parameters or {}

        fast_period = int(
            parameters.get("fast_period", 3)
        )

        slow_period = int(
            parameters.get("slow_period", 10)
        )

        denominator = (
            data["high"]
            - data["low"]
        ).replace(
            0,
            np.nan,
        )

        multiplier = (
            (
                (
                    data["close"]
                    - data["low"]
                )
                - (
                    data["high"]
                    - data["close"]
                )
            )
            / denominator
        )

        ad_volume = (
            multiplier
            * data["volume"].astype(float)
        ).cumsum()

        fast = ad_volume.ewm(
            span=fast_period,
            adjust=False,
        ).mean()

        slow = ad_volume.ewm(
            span=slow_period,
            adjust=False,
        ).mean()

        return fast - slow


class VolumeROC(Feature):

    name = "volume_roc"

    def calculate(
        self,
        data: pd.DataFrame,
        parameters: dict | None = None,
    ) -> pd.Series:

        parameters = parameters or {}

        period = int(
            parameters.get("period", 10)
        )

        volume = data["volume"].astype(float)

        previous = volume.shift(period)

        return (
            (
                volume - previous
            )
            / previous.replace(
                0,
                np.nan,
            )
        ) * 100


class RelativeVolume(Feature):

    name = "relative_volume"

    def calculate(
        self,
        data: pd.DataFrame,
        parameters: dict | None = None,
    ) -> pd.Series:

        parameters = parameters or {}

        period = int(
            parameters.get("period", 20)
        )

        volume = data["volume"].astype(float)

        average_volume = (
            volume
            .rolling(period)
            .mean()
        )

        return (
            volume
            / average_volume.replace(
                0,
                np.nan,
            )
        )


class VolumeSMA(Feature):

    name = "volume_sma"

    def calculate(
        self,
        data: pd.DataFrame,
        parameters: dict | None = None,
    ) -> pd.Series:

        parameters = parameters or {}

        period = int(
            parameters.get("period", 20)
        )

        return (
            data["volume"]
            .astype(float)
            .rolling(period)
            .mean()
        )


class VolumeZScore(Feature):

    name = "volume_zscore"

    def calculate(
        self,
        data: pd.DataFrame,
        parameters: dict | None = None,
    ) -> pd.Series:

        parameters = parameters or {}

        period = int(
            parameters.get("period", 20)
        )

        volume = data["volume"].astype(float)

        mean = (
            volume
            .rolling(period)
            .mean()
        )

        std = (
            volume
            .rolling(period)
            .std(ddof=0)
        )

        return (
            volume - mean
        ) / std.replace(
            0,
            np.nan,
        )


class ForceIndex(Feature):

    name = "force_index"

    def calculate(
        self,
        data: pd.DataFrame,
        parameters: dict | None = None,
    ) -> pd.Series:

        parameters = parameters or {}

        period = int(
            parameters.get("period", 13)
        )

        force = (
            data["close"].diff()
            * data["volume"].astype(float)
        )

        if period <= 1:
            return force

        return force.ewm(
            span=period,
            adjust=False,
        ).mean()


class EaseOfMovement(Feature):

    name = "ease_of_movement"

    def calculate(
        self,
        data: pd.DataFrame,
        parameters: dict | None = None,
    ) -> pd.Series:

        parameters = parameters or {}

        period = int(
            parameters.get("period", 14)
        )

        midpoint_move = (
            (
                data["high"]
                + data["low"]
            ) / 2
        ).diff()

        box_ratio = (
            data["volume"].astype(float)
            /
            (
                data["high"]
                - data["low"]
            ).replace(
                0,
                np.nan,
            )
        )

        raw_eom = (
            midpoint_move
            / box_ratio.replace(
                0,
                np.nan,
            )
        )

        return (
            raw_eom
            .rolling(period)
            .mean()
        )
