import pandas as pd

from backend.engine.feature_engine import FeatureEngine

from backend.engine.indicators.trend import (
    WMA,
    HMA,
    DEMA,
    TEMA,
    VWMA,
    ADX,
)


if __name__ == "__main__":

    data = pd.DataFrame({

        "open": [
            1.1000, 1.1008, 1.1017, 1.1022, 1.1028,
            1.1032, 1.1038, 1.1045, 1.1050, 1.1058,
            1.1062, 1.1068, 1.1075, 1.1080, 1.1087,
            1.1092, 1.1098, 1.1105, 1.1110, 1.1118,
        ],

        "high": [
            1.1010, 1.1020, 1.1025, 1.1030, 1.1035,
            1.1040, 1.1047, 1.1053, 1.1058, 1.1065,
            1.1070, 1.1075, 1.1082, 1.1088, 1.1094,
            1.1100, 1.1106, 1.1113, 1.1118, 1.1125,
        ],

        "low": [
            1.0995, 1.1005, 1.1010, 1.1015, 1.1020,
            1.1025, 1.1032, 1.1040, 1.1045, 1.1052,
            1.1057, 1.1062, 1.1069, 1.1074, 1.1081,
            1.1086, 1.1092, 1.1099, 1.1104, 1.1111,
        ],

        "close": [
            1.1008, 1.1017, 1.1022, 1.1028, 1.1032,
            1.1038, 1.1045, 1.1050, 1.1058, 1.1062,
            1.1068, 1.1075, 1.1080, 1.1087, 1.1092,
            1.1098, 1.1105, 1.1110, 1.1118, 1.1122,
        ],

        "volume": [
            1000, 1100, 1200, 1300, 1400,
            1500, 1600, 1700, 1800, 1900,
            2000, 2100, 2200, 2300, 2400,
            2500, 2600, 2700, 2800, 2900,
        ],

    })

    engine = FeatureEngine()

    engine.register(WMA())
    engine.register(HMA())
    engine.register(DEMA())
    engine.register(TEMA())
    engine.register(VWMA())
    engine.register(ADX())

    result = engine.calculate_all(

        data,

        [
            {
                "name": "wma",
                "parameters": {"period": 5},
                "output": "WMA_5",
            },
            {
                "name": "hma",
                "parameters": {"period": 5},
                "output": "HMA_5",
            },
            {
                "name": "dema",
                "parameters": {"period": 5},
                "output": "DEMA_5",
            },
            {
                "name": "tema",
                "parameters": {"period": 5},
                "output": "TEMA_5",
            },
            {
                "name": "vwma",
                "parameters": {"period": 5},
                "output": "VWMA_5",
            },
            {
                "name": "adx",
                "parameters": {"period": 5},
                "output": "ADX_5",
            },
        ],
    )

    print("========================================")
    print("       NEXA FUNDS AI")
    print("    TREND ENGINE TEST")
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

    print(result.tail(10))

    print()

    print("NEXA FUNDS AI TREND ENGINE OK")
