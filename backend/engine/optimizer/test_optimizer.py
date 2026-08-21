import pandas as pd

from backend.engine.optimizer.strategy_optimizer import (
    StrategyOptimizer,
    Parameter
)


print("=" * 40)
print("       NEXA FUNDS AI")
print("   STRATEGY OPTIMIZER TEST")
print("=" * 40)


# --------------------------------------------------
# Test market data
# --------------------------------------------------

data = pd.DataFrame({
    "close": [
        1.1000, 1.1010, 1.1020, 1.1010,
        1.1030, 1.1050, 1.1040, 1.1060,
        1.1080, 1.1070, 1.1090, 1.1110,
        1.1100, 1.1120, 1.1140
    ]
})


# --------------------------------------------------
# Parameters NEXA FUNDS AI will optimize
# --------------------------------------------------

parameters = [

    Parameter(
        "ema_fast",
        [20, 50]
    ),

    Parameter(
        "ema_slow",
        [100, 200]
    ),

    Parameter(
        "risk_percent",
        [0.5, 1.0, 2.0]
    ),

    Parameter(
        "risk_reward",
        [1.5, 2.0, 3.0]
    )
]


# --------------------------------------------------
# Dummy strategy
# --------------------------------------------------

def strategy_function(data, params):

    # This is only a test.
    # Real strategy logic will be plugged in later.

    return data


# --------------------------------------------------
# Dummy backtest
# --------------------------------------------------

def backtest_function(data, params):

    rr = params["risk_reward"]

    if rr == 1.5:

        return [
            -100,
            150,
            -100,
            150,
            150,
            -100,
            150,
            -100
        ]

    if rr == 2.0:

        return [
            -100,
            200,
            -100,
            200,
            200,
            -100,
            -100,
            200
        ]

    return [
        -100,
        300,
        -100,
        -100,
        300,
        -100,
        300,
        -100
    ]


# --------------------------------------------------
# Run optimizer
# --------------------------------------------------

optimizer = StrategyOptimizer()

results = optimizer.optimize(
    data,
    parameters,
    strategy_function,
    backtest_function
)


print()
print("Parameter combinations tested:")
print(len(results))

print()
print("Top results:")
print(
    results[
        [
            "ema_fast",
            "ema_slow",
            "risk_percent",
            "risk_reward",
            "trades",
            "win_rate",
            "net_profit",
            "profit_factor",
            "expectancy",
            "max_losing_streak",
            "optimization_score"
        ]
    ].head(10).to_string(index=False)
)


print()
print("BEST MODEL")
print("-" * 40)

best = optimizer.best()

for key, value in best.items():
    print(f"{key}: {value}")


print()
print("NEXA FUNDS AI STRATEGY OPTIMIZER OK")
