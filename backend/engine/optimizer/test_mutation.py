from backend.engine.optimizer.mutation_engine import (
    StrategyMutationEngine
)


print("=" * 40)
print("       NEXA FUNDS AI")
print("   STRATEGY MUTATION ENGINE TEST")
print("=" * 40)


strategy = {

    "name": "EMA 50/200 Strategy",

    "symbol": "EURUSD",

    "timeframe": "M15",

    "direction": "BOTH",

    "entry_conditions": [

        "EMA_50 > EMA_200 for BUY",

        "EMA_50 < EMA_200 for SELL"

    ],

    "risk_percent": 1.0,

    "risk_reward": 2.0

}


engine = StrategyMutationEngine()


# --------------------------------------------------
# Single mutations
# --------------------------------------------------

single = engine.mutate(strategy)


print()
print("Single mutations:")
print(len(single))


for candidate in single[:10]:

    print(
        candidate["mutations"]
    )


# --------------------------------------------------
# Combination mutations
# --------------------------------------------------

combined = engine.combine_mutations(
    strategy,
    max_mutations=2
)


print()
print("Two-component mutations:")
print(len(combined))


for candidate in combined[:10]:

    print(
        candidate["mutations"]
    )


# --------------------------------------------------
# Categories
# --------------------------------------------------

print()
print("Available mutation categories:")

for category in engine.mutation_library:

    count = len(
        engine.get_mutations_by_category(
            category
        )
    )

    print(
        f"{category}: {count}"
    )


print()
print(
    "Total mutation types:",
    len(engine.get_all_mutations())
)


print()
print(
    "NEXA FUNDS AI STRATEGY MUTATION ENGINE OK"
)
