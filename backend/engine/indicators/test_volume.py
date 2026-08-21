import numpy as np
import pandas as pd

from backend.engine.feature_engine import FeatureEngine

from backend.engine.indicators.volume import (
    OBV,
    VWAP,
    MFI,
    CMF,
    ADLine,
    ChaikinOscillator,
    VolumeROC,
    RelativeVolume,
    VolumeSMA,
    VolumeZScore,
    ForceIndex,
    EaseOfMovement,
)


if __name__ == "__main__":

    np.random.seed(123)

    rows = 150

    returns = np.random.normal(
        0,
        0.001,
        rows,
    )

    close = (
        1.1000
        * np.exp(
            np.cumsum(returns)
        )
    )

    high = (
        close
        + np.random.uniform(
            0.0002,
            0.0010,
            rows,
        )
    )

    low = (
        close
        - np.random.uniform(
            0.0002,
            0.0010,
            rows,
        )
    )

    open_price = (
        close
        + np.random.normal(
            0,
            0.0003,
            rows,
        )
    )

    volume = np.random.randint(
        1000,
        10000,
        rows,
    )

    data = pd.DataFrame({

        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,

    })

    engine = FeatureEngine()

    engine.register(OBV())
    engine.register(VWAP())
    engine.register(MFI())
    engine.register(CMF())
    engine.register(ADLine())
    engine.register(ChaikinOscillator())
    engine.register(VolumeROC())
    engine.register(RelativeVolume())
    engine.register(VolumeSMA())
    engine.register(VolumeZScore())
    engine.register(ForceIndex())
    engine.register(EaseOfMovement())

    requests = [

        {
            "name": "obv",
            "output": "OBV",
        },

        {
            "name": "vwap",
            "output": "VWAP",
        },

        {
            "name": "mfi",
            "parameters": {
                "period": 14,
            },
            "output": "MFI_14",
        },

        {
            "name": "cmf",
            "parameters": {
                "period": 20,
            },
            "output": "CMF_20",
        },

        {
            "name": "ad_line",
            "output": "AD_LINE",
        },

        {
            "name": "chaikin_oscillator",
            "parameters": {
                "fast_period": 3,
                "slow_period": 10,
            },
            "output": "CHAIKIN",
        },

        {
            "name": "volume_roc",
            "parameters": {
                "period": 10,
            },
            "output": "VOLUME_ROC",
        },

        {
            "name": "relative_volume",
            "parameters": {
                "period": 20,
            },
            "output": "RELATIVE_VOLUME",
        },

        {
            "name": "volume_sma",
            "parameters": {
                "period": 20,
            },
            "output": "VOLUME_SMA",
        },

        {
            "name": "volume_zscore",
            "parameters": {
                "period": 20,
            },
            "output": "VOLUME_ZSCORE",
        },

        {
            "name": "force_index",
            "parameters": {
                "period": 13,
            },
            "output": "FORCE_INDEX",
        },

        {
            "name": "ease_of_movement",
            "parameters": {
                "period": 14,
            },
            "output": "EOM",
        },

    ]

    result = engine.calculate_all(
        data,
        requests,
    )

    print("========================================")
    print("       NEXA FUNDS AI")
    print("      VOLUME ENGINE TEST")
    print("========================================")
    print()

    print("Registered features:")
    print(engine.list_features())
    print()

    print("Generated columns:")

    print(
        [
            column
            for column in result.columns
            if column not in data.columns
        ]
    )

    print()

    print("Last 10 rows:")

    print(result.tail(10))

    print()

    print("NEXA FUNDS AI VOLUME ENGINE OK")
