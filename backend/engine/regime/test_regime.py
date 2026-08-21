import pandas as pd
import numpy as np

from backend.engine.regime.regime_engine import RegimeEngine


np.random.seed(42)


rows = 250


close = (
    1.1000
    +
    np.cumsum(
        np.random.normal(
            0,
            0.001,
            rows
        )
    )
)


data = pd.DataFrame({

    "timestamp": pd.date_range(
        "2026-01-01",
        periods=rows,
        freq="15min"
    ),

    "close": close,

})


data["open"] = (
    data["close"].shift(1)
    .fillna(
        data["close"]
    )
)


data["high"] = (
    data[
        ["open", "close"]
    ].max(axis=1)
    +
    0.0005
)


data["low"] = (
    data[
        ["open", "close"]
    ].min(axis=1)
    -
    0.0005
)


data["volume"] = np.random.randint(
    1000,
    10000,
    rows
)


engine = RegimeEngine()


result = engine.calculate(
    data
)


print("========================================")
print("       NEXA FUNDS AI")
print("       REGIME ENGINE TEST")
print("========================================")
print()


print("Generated columns:")

print([
    "REGIME_EMA_FAST",
    "REGIME_EMA_SLOW",
    "REGIME_ATR",
    "REGIME_ADX",
    "ATR_PERCENTILE",
    "TREND_REGIME",
    "MARKET_CONDITION",
    "VOLATILITY_REGIME",
    "MARKET_REGIME"
])


print()


print(
    result[
        [
            "close",
            "REGIME_ADX",
            "ATR_PERCENTILE",
            "TREND_REGIME",
            "MARKET_CONDITION",
            "VOLATILITY_REGIME",
            "MARKET_REGIME"
        ]
    ].tail(20).to_string()
)


print()


print("Trend distribution:")

print(
    result[
        "TREND_REGIME"
    ].value_counts()
)


print()


print("Market condition distribution:")

print(
    result[
        "MARKET_CONDITION"
    ].value_counts()
)


print()


print("Volatility distribution:")

print(
    result[
        "VOLATILITY_REGIME"
    ].value_counts()
)


print()


print("NEXA FUNDS AI REGIME ENGINE OK")
