import pandas as pd

from backend.engine.feature_engine import FeatureEngine

from backend.engine.indicators.momentum import (
    Stochastic,
    CCI,
    WilliamsR,
    ROC,
    Momentum,
    TSI,
    AwesomeOscillator,
    UltimateOscillator,
    PPO,
)


if __name__ == "__main__":

    data = pd.DataFrame({

        "open": [
            1.1000, 1.1008, 1.1017, 1.1022, 1.1028,
            1.1032, 1.1038, 1.1045, 1.1050, 1.1058,
            1.1062, 1.1068, 1.1075, 1.1080, 1.1087,
            1.1092, 1.1098, 1.1105, 1.1110, 1.1118,
            1.1122, 1.1128, 1.1135, 1.1140, 1.1147,
            1.1152, 1.1158, 1.1165, 1.1170, 1.1178,
            1.1182, 1.1188, 1.1195, 1.1200, 1.1207,
            1.1212, 1.1218, 1.1225, 1.1230, 1.1238,
        ],

        "high": [
            1.1010, 1.1020, 1.1025, 1.1030, 1.1035,
            1.1040, 1.1047, 1.1053, 1.1058, 1.1065,
            1.1070, 1.1075, 1.1082, 1.1088, 1.1094,
            1.1100, 1.1106, 1.1113, 1.1118, 1.1125,
            1.1130, 1.1136, 1.1143, 1.1148, 1.1155,
            1.1160, 1.1166, 1.1173, 1.1178, 1.1185,
            1.1190, 1.1196, 1.1203, 1.1208, 1.1215,
            1.1220, 1.1226, 1.1233, 1.1238, 1.1245,
        ],

        "low": [
            1.0995, 1.1005, 1.1010, 1.1015, 1.1020,
            1.1025, 1.1032, 1.1040, 1.1045, 1.1052,
            1.1057, 1.1062, 1.1069, 1.1074, 1.1081,
            1.1086, 1.1092, 1.1099, 1.1104, 1.1111,
            1.1115, 1.1121, 1.1128, 1.1133, 1.1140,
            1.1145, 1.1151, 1.1158, 1.1163, 1.1170,
            1.1175, 1.1181, 1.1188, 1.1193, 1.1200,
            1.1205, 1.1211, 1.1218, 1.1223, 1.1230,
        ],

        "close": [
            1.1008, 1.1017, 1.1022, 1.1028, 1.1032,
            1.1038, 1.1045, 1.1050, 1.1058, 1.1062,
            1.1068, 1.1075, 1.1080, 1.1087, 1.1092,
            1.1098, 1.1105, 1.1110, 1.1118, 1.1122,
            1.1128, 1.1135, 1.1140, 1.1147, 1.1152,
            1.1158, 1.1165, 1.1170, 1.1178, 1.1182,
            1.1188, 1.1195, 1.1200, 1.1207, 1.1212,
            1.1218, 1.1225, 1.1230, 1.1238, 1.1242,
        ],

    })

    engine = FeatureEngine()

    engine.register(Stochastic())
    engine.register(CCI())
    engine.register(WilliamsR())
    engine.register(ROC())
    engine.register(Momentum())
    engine.register(TSI())
    engine.register(AwesomeOscillator())
    engine.register(UltimateOscillator())
    engine.register(PPO())

    requests = [

        {
            "name": "stochastic",
            "parameters": {
                "period": 5,
                "smooth_k": 3,
                "smooth_d": 3,
            },
        },

        {
            "name": "cci",
            "parameters": {
                "period": 5,
            },
            "output": "CCI_5",
        },

        {
            "name": "williams_r",
            "parameters": {
                "period": 5,
            },
            "output": "WILLIAMS_R_5",
        },

        {
            "name": "roc",
            "parameters": {
                "period": 5,
            },
            "output": "ROC_5",
        },

        {
            "name": "momentum",
            "parameters": {
                "period": 5,
            },
            "output": "MOMENTUM_5",
        },

        {
            "name": "tsi",
            "parameters": {
                "long_period": 10,
                "short_period": 5,
            },
            "output": "TSI",
        },

        {
            "name": "awesome_oscillator",
            "parameters": {
                "fast_period": 3,
                "slow_period": 10,
            },
            "output": "AO",
        },

        {
            "name": "ultimate_oscillator",
            "parameters": {
                "period_1": 5,
                "period_2": 10,
                "period_3": 20,
            },
            "output": "UO",
        },

        {
            "name": "ppo",
            "parameters": {
                "fast_period": 5,
                "slow_period": 10,
            },
            "output": "PPO",
        },

    ]

    result = engine.calculate_all(
        data,
        requests,
    )

    print("========================================")
    print("       NEXA FUNDS AI")
    print("     MOMENTUM ENGINE TEST")
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

    print("NEXA FUNDS AI MOMENTUM ENGINE OK")
