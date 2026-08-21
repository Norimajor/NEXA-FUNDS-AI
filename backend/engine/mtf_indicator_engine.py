from pathlib import Path

import numpy as np
import pandas as pd

from backend.engine.mtf_data_engine import MTFDataEngine


class MTFIndicatorEngine:
    """
    Calculates technical indicators independently on each
    available timeframe.

    Supported:
        H4
        H1
        M30
        M15
        M5
        M1

    Indicators:
        EMA
        SMA
        RSI
        ADX
        ATR
        CCI
        ROC
        MOMENTUM
        STOCHASTIC
        WILLIAMS_R
        TSI
        OBV
        MFI
        CMF
        VWAP
        RELATIVE_VOLUME
    """

    def __init__(
        self,
        data_directory="data",
    ):

        self.data_engine = MTFDataEngine(
            data_directory=data_directory
        )

    # ========================================================
    # PUBLIC
    # ========================================================

    def build(
        self,
        symbol,
        timeframes=None,
    ):

        datasets = self.data_engine.load_available(
            symbol
        )

        if not datasets:
            raise FileNotFoundError(
                f"No market data found for {symbol}"
            )

        if timeframes is None:
            timeframes = [
                "H4",
                "H1",
                "M30",
                "M15",
                "M5",
                "M1",
            ]

        result = {}

        for timeframe in timeframes:

            timeframe = timeframe.upper()

            if timeframe not in datasets:
                continue

            df = datasets[timeframe].copy()

            df = self.calculate_indicators(df)

            result[timeframe] = df

        return result

    # ========================================================
    # ALL INDICATORS
    # ========================================================

    def calculate_indicators(
        self,
        df,
    ):

        df = df.copy()

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # ====================================================
        # MOVING AVERAGES
        # ====================================================

        periods = [
            5,
            9,
            10,
            20,
            21,
            50,
            100,
            200,
        ]

        for period in periods:

            df[f"EMA_{period}"] = (
                close.ewm(
                    span=period,
                    adjust=False,
                ).mean()
            )

            df[f"SMA_{period}"] = (
                close.rolling(
                    period
                ).mean()
            )

        # ====================================================
        # RSI
        # ====================================================

        for period in [7, 14, 21]:

            delta = close.diff()

            gain = delta.clip(
                lower=0
            )

            loss = -delta.clip(
                upper=0
            )

            avg_gain = gain.ewm(
                alpha=1 / period,
                adjust=False,
                min_periods=period,
            ).mean()

            avg_loss = loss.ewm(
                alpha=1 / period,
                adjust=False,
                min_periods=period,
            ).mean()

            rs = (
                avg_gain /
                avg_loss.replace(0, np.nan)
            )

            df[f"RSI_{period}"] = (
                100 -
                (100 / (1 + rs))
            )

        # Default RSI alias
        df["RSI"] = df["RSI_14"]

        # ====================================================
        # ATR
        # ====================================================

        previous_close = close.shift(1)

        tr = pd.concat(
            [
                high - low,
                (high - previous_close).abs(),
                (low - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        for period in [14, 21]:

            df[f"ATR_{period}"] = (
                tr.rolling(period).mean()
            )

        df["ATR"] = df["ATR_14"]

        # ====================================================
        # ADX
        # ====================================================

        for period in [14, 20]:

            up_move = high.diff()

            down_move = -low.diff()

            plus_dm = np.where(
                (up_move > down_move)
                & (up_move > 0),
                up_move,
                0,
            )

            minus_dm = np.where(
                (down_move > up_move)
                & (down_move > 0),
                down_move,
                0,
            )

            plus_dm = pd.Series(
                plus_dm,
                index=df.index,
            )

            minus_dm = pd.Series(
                minus_dm,
                index=df.index,
            )

            atr = (
                tr.rolling(period).mean()
            )

            plus_di = (
                100 *
                plus_dm.rolling(period).mean()
                / atr
            )

            minus_di = (
                100 *
                minus_dm.rolling(period).mean()
                / atr
            )

            dx = (
                100 *
                (plus_di - minus_di).abs()
                /
                (plus_di + minus_di)
            )

            df[f"PLUS_DI_{period}"] = plus_di

            df[f"MINUS_DI_{period}"] = minus_di

            df[f"ADX_{period}"] = (
                dx.rolling(period).mean()
            )

        df["ADX"] = df["ADX_14"]

        # ====================================================
        # CCI
        # ====================================================

        typical_price = (
            high + low + close
        ) / 3

        for period in [14, 20]:

            mean = (
                typical_price
                .rolling(period)
                .mean()
            )

            mean_deviation = (
                typical_price
                .rolling(period)
                .apply(
                    lambda x: np.mean(
                        np.abs(
                            x - np.mean(x)
                        )
                    ),
                    raw=True,
                )
            )

            df[f"CCI_{period}"] = (
                (typical_price - mean)
                /
                (0.015 * mean_deviation)
            )

        df["CCI"] = df["CCI_14"]

        # ====================================================
        # ROC
        # ====================================================

        for period in [10, 14, 20]:

            df[f"ROC_{period}"] = (
                close.pct_change(period)
                * 100
            )

        df["ROC"] = df["ROC_14"]

        # ====================================================
        # MOMENTUM
        # ====================================================

        for period in [10, 14, 20]:

            df[f"MOMENTUM_{period}"] = (
                close.diff(period)
            )

        df["MOMENTUM"] = df["MOMENTUM_14"]

        # ====================================================
        # STOCHASTIC
        # ====================================================

        period = 14

        lowest_low = (
            low.rolling(period).min()
        )

        highest_high = (
            high.rolling(period).max()
        )

        denominator = (
            highest_high - lowest_low
        )

        df["STOCH_K"] = (
            100 *
            (close - lowest_low)
            /
            denominator.replace(0, np.nan)
        )

        df["STOCH_D"] = (
            df["STOCH_K"]
            .rolling(3)
            .mean()
        )

        # ====================================================
        # WILLIAMS %R
        # ====================================================

        df["WILLIAMS_R"] = (
            -100 *
            (
                highest_high - close
            )
            /
            denominator.replace(0, np.nan)
        )

        # ====================================================
        # TSI
        # ====================================================

        momentum = close.diff()

        abs_momentum = momentum.abs()

        double_smoothed = (
            momentum
            .ewm(
                span=25,
                adjust=False,
            )
            .mean()
            .ewm(
                span=13,
                adjust=False,
            )
            .mean()
        )

        double_smoothed_abs = (
            abs_momentum
            .ewm(
                span=25,
                adjust=False,
            )
            .mean()
            .ewm(
                span=13,
                adjust=False,
            )
            .mean()
        )

        df["TSI"] = (
            100 *
            double_smoothed /
            double_smoothed_abs.replace(
                0,
                np.nan,
            )
        )

        # ====================================================
        # OBV
        # ====================================================

        direction = np.sign(
            close.diff()
        ).fillna(0)

        df["OBV"] = (
            direction * volume
        ).cumsum()

        # ====================================================
        # VWAP
        # ====================================================

        cumulative_volume = (
            volume.cumsum()
        )

        cumulative_pv = (
            typical_price * volume
        ).cumsum()

        df["VWAP"] = (
            cumulative_pv /
            cumulative_volume.replace(
                0,
                np.nan,
            )
        )

        # ====================================================
        # RELATIVE VOLUME
        # ====================================================

        df["RELATIVE_VOLUME"] = (
            volume /
            volume.rolling(20).mean()
        )

        # ====================================================
        # MFI
        # ====================================================

        money_flow = (
            typical_price * volume
        )

        positive_flow = pd.Series(
            np.where(
                typical_price > typical_price.shift(1),
                money_flow,
                0,
            ),
            index=df.index,
        )

        negative_flow = pd.Series(
            np.where(
                typical_price < typical_price.shift(1),
                money_flow,
                0,
            ),
            index=df.index,
        )

        positive_sum = (
            positive_flow
            .rolling(14)
            .sum()
        )

        negative_sum = (
            negative_flow
            .rolling(14)
            .sum()
        )

        money_ratio = (
            positive_sum /
            negative_sum.replace(
                0,
                np.nan,
            )
        )

        df["MFI"] = (
            100 -
            (100 / (1 + money_ratio))
        )

        # ====================================================
        # CMF
        # ====================================================

        money_flow_multiplier = (
            (
                (close - low)
                -
                (high - close)
            )
            /
            (high - low).replace(
                0,
                np.nan,
            )
        )

        money_flow_volume = (
            money_flow_multiplier * volume
        )

        df["CMF"] = (
            money_flow_volume
            .rolling(20)
            .sum()
            /
            volume
            .rolling(20)
            .sum()
        )

        return df

    # ========================================================
    # SUMMARY
    # ========================================================

    def summary(
        self,
        symbol,
    ):

        data = self.build(symbol)

        print()
        print("=" * 60)
        print(
            f"MTF INDICATOR SUMMARY: {symbol}"
        )
        print("=" * 60)

        for timeframe, df in data.items():

            print(
                f"{timeframe:<5} "
                f"{len(df):>10,} rows  "
                f"{len(df.columns):>3} columns"
            )

            print(
                "      "
                f"EMA_20={df['EMA_20'].iloc[-1]:.5f}  "
                f"EMA_50={df['EMA_50'].iloc[-1]:.5f}  "
                f"RSI={df['RSI'].iloc[-1]:.2f}  "
                f"ADX={df['ADX'].iloc[-1]:.2f}"
            )