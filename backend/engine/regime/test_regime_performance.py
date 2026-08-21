import pandas as pd

from backend.engine.regime.regime_performance import (
    RegimePerformanceEngine
)


trades = pd.DataFrame({

    "result": [
        "WIN",
        "LOSS",
        "WIN",
        "WIN",
        "LOSS",
        "LOSS",
        "WIN",
        "WIN",
        "LOSS",
        "WIN"
    ],

    "pnl": [
        200,
        -100,
        150,
        180,
        -100,
        -100,
        250,
        300,
        -100,
        220
    ],

    "TREND_REGIME": [
        "BULL",
        "BULL",
        "BULL",
        "BEAR",
        "BEAR",
        "BEAR",
        "STRONG_BULL",
        "STRONG_BULL",
        "NEUTRAL",
        "NEUTRAL"
    ],

    "MARKET_CONDITION": [
        "TRENDING",
        "TRENDING",
        "TRENDING",
        "TRENDING",
        "TRENDING",
        "RANGING",
        "TRENDING",
        "TRENDING",
        "RANGING",
        "RANGING"
    ],

    "VOLATILITY_REGIME": [
        "NORMAL",
        "NORMAL",
        "HIGH",
        "HIGH",
        "HIGH",
        "LOW",
        "NORMAL",
        "HIGH",
        "LOW",
        "NORMAL"
    ]

})


engine = RegimePerformanceEngine()


print("========================================")
print("       NEXA FUNDS AI")
print("  REGIME PERFORMANCE TEST")
print("========================================")
print()


trend = engine.analyze(
    trades,
    "TREND_REGIME"
)


print("TREND PERFORMANCE")
print(
    pd.DataFrame(trend).T.to_string()
)


print()


condition = engine.analyze(
    trades,
    "MARKET_CONDITION"
)


print("MARKET CONDITION PERFORMANCE")
print(
    pd.DataFrame(condition).T.to_string()
)


print()


volatility = engine.analyze(
    trades,
    "VOLATILITY_REGIME"
)


print("VOLATILITY PERFORMANCE")
print(
    pd.DataFrame(volatility).T.to_string()
)


print()


print("BEST TREND REGIMES")


print(
    engine.rank_regimes(
        trend
    ).to_string(
        index=False
    )
)


print()


print(
    "NEXA FUNDS AI REGIME PERFORMANCE ENGINE OK"
)
