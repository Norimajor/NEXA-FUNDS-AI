import pandas as pd

from backend.engine.backtest_engine import BacktestEngine


if __name__ == "__main__":

    data = pd.DataFrame({

        "timestamp": pd.date_range(
            "2026-01-01",
            periods=20,
            freq="15min"
        ),

        "open": [
            1.1000, 1.1005, 1.1010, 1.1015, 1.1020,
            1.1025, 1.1030, 1.1035, 1.1040, 1.1045,
            1.1050, 1.1055, 1.1060, 1.1065, 1.1070,
            1.1075, 1.1080, 1.1085, 1.1090, 1.1095
        ],

        "high": [
            1.1010, 1.1015, 1.1020, 1.1025, 1.1030,
            1.1035, 1.1040, 1.1045, 1.1050, 1.1055,
            1.1060, 1.1065, 1.1070, 1.1075, 1.1080,
            1.1085, 1.1090, 1.1095, 1.1100, 1.1105
        ],

        "low": [
            1.0990, 1.0995, 1.1000, 1.1005, 1.1010,
            1.1015, 1.1020, 1.1025, 1.1030, 1.1035,
            1.1040, 1.1045, 1.1050, 1.1055, 1.1060,
            1.1065, 1.1070, 1.1075, 1.1080, 1.1085
        ],

        "close": [
            1.1005, 1.1010, 1.1015, 1.1020, 1.1025,
            1.1030, 1.1035, 1.1040, 1.1045, 1.1050,
            1.1055, 1.1060, 1.1065, 1.1070, 1.1075,
            1.1080, 1.1085, 1.1090, 1.1095, 1.1100
        ],

        "signal": [
            "NONE",
            "BUY",
            "NONE",
            "NONE",
            "NONE",
            "NONE",
            "SELL",
            "NONE",
            "NONE",
            "NONE",
            "BUY",
            "NONE",
            "NONE",
            "NONE",
            "SELL",
            "NONE",
            "NONE",
            "BUY",
            "NONE",
            "NONE"
        ]

    })


    engine = BacktestEngine(
        initial_balance=10000,
        risk_percent=1.0,
        reward_risk=2.0,
        commission=0.0
    )


    result = engine.run(
        data,
        stop_distance=0.0010
    )


    print("========================================")
    print("       NEXA FUNDS AI")
    print("       BACKTEST ENGINE TEST")
    print("========================================")
    print()

    print(
        "Initial balance:",
        result["initial_balance"]
    )

    print(
        "Final balance:",
        result["final_balance"]
    )

    print(
        "Net profit:",
        result["net_profit"]
    )

    print()

    print("Trade ledger:")

    if len(result["trades"]) > 0:

        print(
            result["trades"].to_string(
                index=False
            )
        )

    else:

        print("No trades generated.")


    print()

    print(
        "Total trades:",
        len(result["trades"])
    )

    print()

    print(
        "NEXA FUNDS AI BACKTEST ENGINE OK"
    )
