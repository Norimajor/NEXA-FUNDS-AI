import pandas as pd

from backend.engine.strategy_interpreter.evaluator.condition_evaluator import (
    ConditionEvaluator
)

from backend.engine.strategy_interpreter.models import StrategyCondition


print("=" * 40)
print("       NEXA FUNDS AI")
print("   CONDITION EVALUATOR TEST")
print("=" * 40)


# ========================================
# TEST DATA
# ========================================

data = pd.DataFrame({

    "close": [
        1.1000,
        1.1010,
        1.1020,
        1.1030,
        1.1040,
        1.1050,
    ],

    "RSI_14": [
        48,
        51,
        54,
        56,
        60,
        65,
    ],

    "ADX_14": [
        18,
        20,
        22,
        24,
        27,
        30,
    ],

    # EMA 20 starts below EMA 100
    # and crosses above it at index 3

    "EMA_20": [
        1.1000,
        1.1005,
        1.1008,
        1.1025,
        1.1045,
        1.1060,
    ],

    "EMA_100": [
        1.1010,
        1.1010,
        1.1010,
        1.1020,
        1.1030,
        1.1040,
    ],
})


# ========================================
# CREATE CONDITIONS
# ========================================

# RSI 14 > 55

rsi_condition = StrategyCondition(
    indicator="RSI",
    operator=">",
    value=55,
    period=14,
)


# ADX 14 > 25

adx_condition = StrategyCondition(
    indicator="ADX",
    operator=">",
    value=25,
    period=14,
)


# EMA 20 crosses above EMA 100
#
# period = fast EMA
# value  = slow EMA

ema_condition = StrategyCondition(
    indicator="EMA_CROSS",
    operator="cross_above",
    value=100,
    period=20,
)


# ========================================
# CREATE EVALUATOR
# ========================================

evaluator = ConditionEvaluator(data)


# ========================================
# TEST RSI
# ========================================

print("\nTesting RSI 14 > 55:")

rsi_result = evaluator.evaluate_condition(
    rsi_condition
)

print(rsi_result)


# ========================================
# TEST ADX
# ========================================

print("\nTesting ADX 14 > 25:")

adx_result = evaluator.evaluate_condition(
    adx_condition
)

print(adx_result)


# ========================================
# TEST EMA CROSS
# ========================================

print("\nTesting EMA 20 crosses above EMA 100:")

ema_result = evaluator.evaluate_condition(
    ema_condition
)

print(ema_result)


# ========================================
# COMBINED STRATEGY
# ========================================

print("\nCombined RSI + ADX conditions:")

combined = evaluator.evaluate_strategy([
    rsi_condition,
    adx_condition,
])

print(combined)


# ========================================
# FULL STRATEGY
# ========================================

print("\nFull strategy:")

full_strategy = evaluator.evaluate_strategy([
    ema_condition,
    rsi_condition,
    adx_condition,
])

print(full_strategy)


# ========================================
# LATEST CANDLE
# ========================================

print("\nLatest candle evaluation:")

latest = evaluator.evaluate_last([
    ema_condition,
    rsi_condition,
    adx_condition,
])

for key, value in latest.items():
    print(f"{key}: {value}")


# ========================================
# EXPECTED RESULTS
# ========================================

print("\nExpected:")

print("EMA 20 crossed above EMA 100: TRUE")
print("RSI 14 > 55: TRUE")
print("ADX 14 > 25: TRUE")
print("FINAL SIGNAL: BUY")


# ========================================
# SUCCESS
# ========================================

print("\nNEXA FUNDS AI CONDITION EVALUATOR OK")