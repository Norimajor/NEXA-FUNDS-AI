from backend.engine.strategies.strategy import (
    StrategyDefinition,
    StrategyEngine,
    Condition,
    RiskManagement,
    RiskMode,
)


if __name__ == "__main__":

    strategy = StrategyDefinition(

        name="EMA 50/200 Trend Strategy",

        description=(
            "Trend-following strategy using "
            "EMA 50 and EMA 200."
        ),

        symbol="EURUSD",

        timeframe="M15",

        direction="both",

        entry_conditions=[

            Condition(
                type="ema_cross",
                parameters={
                    "fast_period": 50,
                    "slow_period": 200,
                },
            )

        ],

        filters=[

            Condition(
                type="rsi",
                parameters={
                    "period": 14,
                    "operator": ">",
                    "value": 50,
                },
            )

        ],

        risk_management=RiskManagement(

            mode=RiskMode.PERCENT,

            risk_value=1.0,

            stop_loss={
                "type": "pips",
                "value": 30,
            },

            take_profit={
                "type": "risk_reward",
                "ratio": 2.0,
            },

        ),

    )

    validation = StrategyEngine.validate(
        strategy
    )

    summary = StrategyEngine.summarize(
        strategy
    )

    print("========================================")
    print("       NEXA FUNDS AI")
    print("     STRATEGY ENGINE TEST")
    print("========================================")
    print()

    print("Validation:")
    print(validation)
    print()

    print("Strategy summary:")

    for key, value in summary.items():
        print(f"{key}: {value}")

    print()

    if validation["valid"]:
        print("NEXA FUNDS AI STRATEGY ENGINE OK")
    else:
        print("STRATEGY VALIDATION FAILED")
