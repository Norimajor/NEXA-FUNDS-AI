from backend.engine.backtest_engine import BacktestEngine
from backend.engine.performance_engine import PerformanceEngine

import pandas as pd


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
            "NONE", "BUY", "NONE", "NONE", "NONE",
            "NONE", "SELL", "NONE", "NONE", "NONE",
            "BUY", "NONE", "NONE", "NONE", "SELL",
            "NONE", "NONE", "BUY", "NONE", "NONE"
        ]

    })


    backtester = BacktestEngine(

        initial_balance=10000,

        risk_percent=1.0,

        reward_risk=2.0,

        commission=0.0

    )


    backtest = backtester.run(

        data,

        stop_distance=0.0010

    )


    performance = PerformanceEngine(
        initial_balance=10000
    )


    metrics = performance.calculate(
        backtest["trades"]
    )


    print("========================================")
    print("       NEXA FUNDS AI")
    print("     PERFORMANCE ENGINE TEST")
    print("========================================")
    print()


    for key, value in metrics.items():

        if isinstance(value, float):

            print(
                f"{key}: {value:.4f}"
            )

        else:

            print(
                f"{key}: {value}"
            )


    print()

    print(
        "NEXA FUNDS AI PERFORMANCE ENGINE OK"
    )
