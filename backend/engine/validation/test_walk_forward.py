import pandas as pd

from backend.engine.validation.walk_forward import (
    WalkForwardEngine
)


print("=" * 40)
print("       NEXA FUNDS AI")
print("   WALK-FORWARD VALIDATION TEST")
print("=" * 40)


# --------------------------------------------------
# Create synthetic market data
# --------------------------------------------------

data = pd.DataFrame({
    "close": [
        1.1000 + (i * 0.0001)
        for i in range(100)
    ]
})


# --------------------------------------------------
# Strategy
# --------------------------------------------------

def strategy_function(
    data,
    parameters
):

    return data


# --------------------------------------------------
# Backtest
# --------------------------------------------------

def backtest_function(
    data,
    parameters
):

    # Synthetic test trades.

    if len(data) < 10:
        return []

    return [
        100,
        -50,
        100,
        -50,
        100,
        -50,
        100,
        -50
    ]


# --------------------------------------------------
# Engine
# --------------------------------------------------

engine = WalkForwardEngine(
    train_size=0.60,
    test_size=0.20,
    step_size=0.20
)


windows = engine.generate_windows(
    len(data)
)


print()
print("Validation windows:")
print(len(windows))


for window in windows:

    print(
        window
    )


# --------------------------------------------------
# Run validation
# --------------------------------------------------

results = engine.run(
    data,
    strategy_function,
    backtest_function,
    {
        "ema_fast": 50,
        "ema_slow": 200
    }
)


print()
print("Walk-forward results:")
print(
    results.to_string(
        index=False
    )
)


# --------------------------------------------------
# Robustness
# --------------------------------------------------

score = engine.robustness_score(
    results
)


print()
print(
    "ROBUSTNESS SCORE:",
    score
)


print()
print(
    "NEXA FUNDS AI WALK-FORWARD ENGINE OK"
)
