import numpy as np
import pandas as pd


class RegimeEngine:

    def __init__(
        self,
        ema_fast=20,
        ema_slow=50,
        adx_threshold=25,
        atr_percentile_low=20,
        atr_percentile_high=80
    ):

        self.ema_fast = ema_fast
        self.ema_slow = ema_slow

        self.adx_threshold = adx_threshold

        self.atr_percentile_low = (
            atr_percentile_low
        )

        self.atr_percentile_high = (
            atr_percentile_high
        )


    def calculate_ema(
        self,
        close,
        period
    ):

        return close.ewm(
            span=period,
            adjust=False
        ).mean()


    def calculate_atr(
        self,
        data,
        period=14
    ):

        high = data["high"]
        low = data["low"]
        close = data["close"]

        previous_close = close.shift(1)

        tr1 = high - low

        tr2 = (
            high - previous_close
        ).abs()

        tr3 = (
            low - previous_close
        ).abs()

        true_range = pd.concat(
            [tr1, tr2, tr3],
            axis=1
        ).max(axis=1)

        return true_range.rolling(
            period
        ).mean()


    def calculate_adx(
        self,
        data,
        period=14
    ):

        high = data["high"]
        low = data["low"]

        up_move = (
            high.diff()
        )

        down_move = (
            -low.diff()
        )


        plus_dm = pd.Series(
            np.where(
                (up_move > down_move)
                & (up_move > 0),
                up_move,
                0.0
            ),
            index=data.index
        )


        minus_dm = pd.Series(
            np.where(
                (down_move > up_move)
                & (down_move > 0),
                down_move,
                0.0
            ),
            index=data.index
        )


        atr = self.calculate_atr(
            data,
            period
        )


        plus_di = (
            100
            * plus_dm.rolling(period).mean()
            / atr
        )


        minus_di = (
            100
            * minus_dm.rolling(period).mean()
            / atr
        )


        denominator = (
            plus_di + minus_di
        ).replace(0, np.nan)


        dx = (
            (plus_di - minus_di).abs()
            /
            denominator
        ) * 100


        return dx.rolling(
            period
        ).mean()


    def classify_trend(
        self,
        close,
        ema_fast,
        ema_slow,
        adx
    ):

        if pd.isna(
            ema_fast
        ) or pd.isna(
            ema_slow
        ):

            return "NEUTRAL"


        if pd.isna(adx):

            return "NEUTRAL"


        bullish = (
            close > ema_fast
            and
            ema_fast > ema_slow
        )


        bearish = (
            close < ema_fast
            and
            ema_fast < ema_slow
        )


        if bullish:

            if adx >= 40:
                return "STRONG_BULL"

            if adx >= self.adx_threshold:
                return "BULL"


        if bearish:

            if adx >= 40:
                return "STRONG_BEAR"

            if adx >= self.adx_threshold:
                return "BEAR"


        return "NEUTRAL"


    def classify_market_condition(
        self,
        adx
    ):

        if pd.isna(adx):

            return "UNKNOWN"


        if adx >= self.adx_threshold:

            return "TRENDING"


        return "RANGING"


    def classify_volatility(
        self,
        atr_percentile
    ):

        if pd.isna(
            atr_percentile
        ):

            return "UNKNOWN"


        if atr_percentile <= 10:

            return "VERY_LOW"


        if atr_percentile <= (
            self.atr_percentile_low
        ):

            return "LOW"


        if atr_percentile >= 90:

            return "VERY_HIGH"


        if atr_percentile >= (
            self.atr_percentile_high
        ):

            return "HIGH"


        return "NORMAL"


    def calculate(
        self,
        data
    ):

        data = data.copy()

        close = pd.to_numeric(
            data["close"],
            errors="coerce"
        )


        ema_fast = self.calculate_ema(
            close,
            self.ema_fast
        )


        ema_slow = self.calculate_ema(
            close,
            self.ema_slow
        )


        atr = self.calculate_atr(
            data
        )


        adx = self.calculate_adx(
            data
        )


        atr_percentile = (
            atr.rolling(100).rank(
                pct=True
            ) * 100
        )


        data["REGIME_EMA_FAST"] = (
            ema_fast
        )

        data["REGIME_EMA_SLOW"] = (
            ema_slow
        )

        data["REGIME_ATR"] = (
            atr
        )

        data["REGIME_ADX"] = (
            adx
        )

        data["ATR_PERCENTILE"] = (
            atr_percentile
        )


        trends = []

        conditions = []

        volatilities = []

        combined = []


        for i in range(
            len(data)
        ):

            trend = self.classify_trend(

                close.iloc[i],

                ema_fast.iloc[i],

                ema_slow.iloc[i],

                adx.iloc[i]

            )


            condition = (
                self.classify_market_condition(
                    adx.iloc[i]
                )
            )


            volatility = (
                self.classify_volatility(
                    atr_percentile.iloc[i]
                )
            )


            if (
                condition == "UNKNOWN"
                or
                volatility == "UNKNOWN"
            ):

                combined_regime = "UNKNOWN"

            else:

                combined_regime = (
                    f"{trend}_"
                    f"{condition}_"
                    f"{volatility}"
                )


            trends.append(trend)

            conditions.append(
                condition
            )

            volatilities.append(
                volatility
            )

            combined.append(
                combined_regime
            )


        data["TREND_REGIME"] = trends

        data["MARKET_CONDITION"] = (
            conditions
        )

        data["VOLATILITY_REGIME"] = (
            volatilities
        )

        data["MARKET_REGIME"] = combined


        return data
