import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from ml.dataset_builder import DatasetBuilder
from ml.predictor import ModelPredictor
from ml.walk_forward import WalkForwardValidator
from core.data_processor import MarketDataProcessor
from core.trading_engine import TradingEngine


class DummyMulticlassModel:
    classes_ = np.array(["reversal", "continuation", "breakout", "failed_breakout", "no_edge"], dtype=object)

    def predict_proba(self, X):
        return np.array([[0.55, 0.20, 0.10, 0.10, 0.05]], dtype=float)


class DummyDirectionalModel:
    classes_ = np.array([
        "reversal_buy", "reversal_sell", "continuation_buy", "continuation_sell",
        "breakout_buy", "breakout_sell", "failed_breakout_buy", "failed_breakout_sell", "no_edge",
    ])

    def predict_proba(self, X):
        return np.array([[0.55, 0.10, 0.10, 0.05, 0.05, 0.03, 0.02, 0.03, 0.07]], dtype=float)


class ConstantIntervalModel:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.array([self.value])


class MultiClassModelPipelineTests(unittest.TestCase):
    def _make_candles(self, length=400):
        rows = []
        price = 100.0
        for i in range(length):
            open_ = price
            drift = 0.15 if i % 6 < 3 else -0.1
            close = open_ + drift
            high = max(open_, close) + 0.8
            low = min(open_, close) - 0.8
            rows.append({
                "timestamp": pd.Timestamp("2020-01-01") + pd.Timedelta(minutes=15 * i),
                "open": float(open_),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": 1200.0,
            })
            price = close
        return pd.DataFrame(rows)

    def test_dataset_builder_emits_market_structure_probabilities(self):
        from core.feature_engine import FEATURE_NAMES

        builder = DatasetBuilder(horizon=6, stop_atr=1.0, target_atr=1.5)
        feature_values = {name: 0.0 for name in FEATURE_NAMES}
        with (
            patch.object(builder.channels, 'detect', return_value=[]),
            patch.object(builder.zones, 'detect', return_value=[]),
            patch.object(builder.structure, 'analyze', return_value={'structure': 'NEUTRAL'}),
            patch.object(builder.features, 'build', return_value=feature_values),
        ):
            dataset = builder.build(self._make_candles(420))
        self.assertGreater(len(dataset), 0)
        for key in [
            "label",
            "label_name",
            "probability_reversal",
            "probability_continuation",
            "probability_breakout",
            "probability_failed_breakout",
            "probability_no_edge",
            "probability_long",
            "probability_short",
            "probability_flat",
            "grid_interval_buy_atr",
            "grid_interval_sell_atr",
            "swing_stop_buy_atr",
            "swing_target_buy_atr",
            "swing_stop_sell_atr",
            "swing_target_sell_atr",
            "pivot_high_distance_atr",
            "pivot_low_distance_atr",
            "double_top_similarity_atr",
            "double_bottom_similarity_atr",
            "sweep_high",
            "sweep_low",
        ]:
            self.assertIn(key, dataset.columns)

    def test_dataset_builder_samples_labels_at_requested_stride(self):
        from core.feature_engine import FEATURE_NAMES

        builder = DatasetBuilder(horizon=3)
        timestamps = pd.date_range('2024-01-01', periods=300, freq='15min')
        candles = pd.DataFrame({
            'timestamp': timestamps,
            'open': np.full(300, 100.0),
            'high': np.full(300, 101.0),
            'low': np.full(300, 99.0),
            'close': np.full(300, 100.0),
            'volume': np.full(300, 1200.0),
            'atr_14': np.ones(300),
            'swing_high': np.zeros(300, dtype=bool),
            'swing_low': np.zeros(300, dtype=bool),
        })
        feature_values = {name: 0.0 for name in FEATURE_NAMES}
        with (
            patch.object(builder.channels, 'detect', return_value=[]),
            patch.object(builder.zones, 'detect', return_value=[]),
            patch.object(builder.structure, 'analyze', return_value={'structure': 'NEUTRAL'}),
            patch.object(builder.features, 'build', return_value=feature_values),
        ):
            dataset = builder.build(candles, sample_stride=10)

        expected_timestamps = pd.to_datetime(timestamps[250:297:10], utc=True)
        self.assertEqual(dataset['timestamp'].tolist(), list(expected_timestamps))

    def test_grid_interval_labels_measure_adverse_move_until_target(self):
        builder = DatasetBuilder(horizon=2, target_atr=1.5, grid_target_distance=1.5)
        future = pd.DataFrame({
            "high": [101.0, 102.0],
            "low": [99.5, 98.0],
            "close": [100.5, 100.0],
        })

        buy_interval = builder._adverse_excursion_atr(future, 100.0, 1.0, "BUY")
        sell_interval = builder._adverse_excursion_atr(future, 100.0, 1.0, "SELL")

        self.assertEqual(buy_interval, 2.0)
        self.assertEqual(sell_interval, 2.0)

    def test_dataset_builder_bounded_history_preserves_sample_features(self):
        from core.feature_engine import FEATURE_NAMES

        builder = DatasetBuilder(horizon=6)
        candles = self._make_candles(257)
        prepared = builder._ensure_features(candles.copy())
        actual = builder.build(prepared)

        hist = prepared.iloc[:251].copy()
        channel = builder.channels.detect(hist)
        channel = channel[0] if channel else None
        zones = builder.zones.detect(hist)
        structure = builder.structure.analyze(hist)
        expected_features = builder.features.build(hist, channel, zones, structure)
        expected_context = builder._future_context(
            hist,
            prepared.iloc[251:257],
            float(hist.iloc[-1]['atr_14']),
            float(hist.iloc[-1].close),
        )
        expected_label = builder._label_from_context(expected_context)

        self.assertEqual(len(actual), 1)
        self.assertEqual(actual.iloc[0]['label'], expected_label[0])
        self.assertEqual(actual.iloc[0]['label_name'], expected_label[1])
        np.testing.assert_allclose(
            actual.iloc[0][FEATURE_NAMES].to_numpy(dtype=float),
            [expected_features.get(name, 0.0) for name in FEATURE_NAMES],
        )

    def test_fractal_pivot_is_available_only_after_confirmation(self):
        from core.data_processor import MarketDataProcessor
        from core.market_structure import MarketStructureAnalyzer

        candles = self._make_candles(300)
        candles.loc[:, "open"] = 10.0
        candles.loc[:, "close"] = 10.0
        candles.loc[:, "high"] = 11.0
        candles.loc[:, "low"] = 9.0
        candles.loc[28:29, "high"] = 12.0
        candles.loc[30, "high"] = 15.0
        candles.loc[31:32, "high"] = 12.0
        candles.loc[48:49, "low"] = 8.0
        candles.loc[50, "low"] = 5.0
        candles.loc[51:52, "low"] = 8.0

        prepared = MarketDataProcessor().prepare(candles)

        self.assertFalse(bool(prepared.loc[30, "swing_high"]))
        self.assertTrue(bool(prepared.loc[32, "swing_high"]))
        self.assertEqual(prepared.loc[32, "swing_high_price"], 15.0)
        self.assertFalse(bool(prepared.loc[50, "swing_low"]))
        self.assertTrue(bool(prepared.loc[52, "swing_low"]))
        self.assertEqual(prepared.loc[52, "swing_low_price"], 5.0)
        structure = MarketStructureAnalyzer(lookback=300).analyze(prepared)
        self.assertEqual(structure["last_swing_high"], 15.0)
        self.assertEqual(structure["last_swing_low"], 5.0)

    def test_reversal_barrier_labels_require_target_before_stop(self):
        builder = DatasetBuilder(horizon=3, stop_atr=1.0, target_atr=1.5)
        target_first = pd.DataFrame({"high": [101.0, 101.6], "low": [99.5, 99.2]})
        stop_first = pd.DataFrame({"high": [100.5, 100.7], "low": [98.9, 99.5]})
        ambiguous = pd.DataFrame({"high": [101.6], "low": [98.9]})

        self.assertEqual(builder._first_barrier(target_first, 100.0, 1.0, "BUY", 1.0, 1.5), "target")
        self.assertEqual(builder._first_barrier(stop_first, 100.0, 1.0, "BUY", 1.0, 1.5), "stop")
        self.assertEqual(builder._first_barrier(ambiguous, 100.0, 1.0, "BUY", 1.0, 1.5), "ambiguous")
        label = builder._label_from_context({
            "buy_barrier": "ambiguous",
            "sell_barrier": "target",
            "reversal_up": True,
            "reversal_down": False,
            "continuation_up": False,
            "continuation_down": False,
            "breakout_up": False,
            "breakout_down": False,
            "failed_up": False,
            "failed_down": False,
        })
        self.assertEqual(label, (8, "no_edge", "NONE"))

    def test_feature_engine_encodes_repeated_swing_similarity_and_rejection(self):
        from core.feature_engine import FEATURE_NAMES, FeatureEngine

        frame = pd.DataFrame({
            "open": [100.0, 100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 102.0, 102.0, 103.0],
            "low": [99.0, 98.0, 99.0, 98.0, 96.0],
            "close": [100.0, 100.0, 101.0, 100.0, 102.0],
            "ema_20": [100.0] * 5,
            "ema_50": [100.0] * 5,
            "ema_200": [100.0] * 5,
            "rsi_14": [50.0] * 5,
            "macd_hist": [0.0] * 5,
            "atr_14": [1.0] * 5,
            "volume_ratio": [1.0] * 5,
            "return_5": [0.0] * 5,
            "return_20": [0.0] * 5,
            "body_ratio": [0.5] * 5,
            "upper_wick": [0.0, 0.0, 0.0, 0.0, 1.0],
            "lower_wick": [0.0, 0.0, 0.0, 0.0, 4.0],
            "swing_high": [False, False, True, True, False],
            "swing_high_price": [np.nan, np.nan, 105.0, 104.9, np.nan],
            "swing_low": [False, True, False, False, True],
            "swing_low_price": [np.nan, 98.0, np.nan, np.nan, 98.2],
        })

        features = FeatureEngine().build(frame, None, [], {"structure": "NEUTRAL"})

        self.assertEqual(set(FEATURE_NAMES) - set(features), set())
        self.assertAlmostEqual(features["double_top_similarity_atr"], 0.1)
        self.assertGreater(features["equal_high_cluster"], 0.0)
        self.assertEqual(features["sweep_low"], 1.0)
        self.assertEqual(features["bullish_rejection"], 1.0)

    def test_training_feature_set_includes_pivot_patterns(self):
        from core.feature_engine import TRAINING_FEATURE_NAMES

        self.assertIn("double_top_similarity_atr", TRAINING_FEATURE_NAMES)
        self.assertIn("double_bottom_similarity_atr", TRAINING_FEATURE_NAMES)
        self.assertIn("sweep_high", TRAINING_FEATURE_NAMES)
        self.assertIn("sweep_low", TRAINING_FEATURE_NAMES)
        self.assertNotIn("channel_score", TRAINING_FEATURE_NAMES)

    def test_predictor_aggregates_multiclass_model_output(self):
        predictor = ModelPredictor.__new__(ModelPredictor)
        predictor.threshold = 0.5
        predictor.bundle = {
            "model": DummyMulticlassModel(),
            "feature_names": ["open", "close", "volume"],
            "class_names": ["reversal", "continuation", "breakout", "failed_breakout", "no_edge"],
            "version": "v_test",
        }
        result = predictor.predict(np.array([[100.0, 100.2, 1200.0]], dtype=float), ["open", "close", "volume"])
        self.assertIn("probability_reversal", result)
        self.assertIn("probability_continuation", result)
        self.assertIn("probability_breakout", result)
        self.assertIn("probability_failed_breakout", result)
        self.assertIn("probability_no_edge", result)
        self.assertIn("probability_long", result)
        self.assertIn("probability_short", result)
        self.assertIn("probability_flat", result)
        self.assertEqual(result["direction"], "NONE")
        self.assertEqual(result["signal"], "WAIT")

    def test_predictor_returns_setup_type_and_direction(self):
        class_names = list(DummyDirectionalModel.classes_)
        predictor = ModelPredictor.__new__(ModelPredictor)
        predictor.threshold = 0.5
        predictor.bundle = {
            "model": DummyDirectionalModel(),
            "class_names": class_names,
            "version": "directional_test",
        }

        result = predictor.predict(np.array([[1.0, 2.0]], dtype=float))

        self.assertEqual(result["setup_type"], "reversal")
        self.assertEqual(result["direction"], "BUY")
        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["setup_probability"], 0.55)
        self.assertAlmostEqual(result["probability_reversal"], 0.65)
        self.assertAlmostEqual(result["buy_probability"], 0.72)
        self.assertAlmostEqual(result["sell_probability"], 0.21)

    def test_market_trend_remains_detectable_when_setup_model_says_no_edge(self):
        candles = []
        for index in range(320):
            close = 200.0 - index * 0.25 + np.sin(index / 4) * 0.3
            open_ = close + 0.08
            candles.append({
                "timestamp": pd.Timestamp("2025-01-01") + pd.Timedelta(hours=index),
                "open": open_,
                "high": max(open_, close) + 0.35,
                "low": min(open_, close) - 0.35,
                "close": close,
                "volume": 1000.0,
            })

        class NoEdgePredictor:
            def predict(self, X, features):
                return {
                    "setup_type": "no_edge",
                    "direction": "NONE",
                    "signal": "WAIT",
                    "setup_probability": 0.3,
                    "probability_no_edge": 0.7,
                    "threshold": 0.62,
                }

        result = TradingEngine(NoEdgePredictor()).analyze(
            MarketDataProcessor().prepare(candles),
            use_structure_filters=False,
        )

        self.assertEqual(result["signal"], "WAIT")
        self.assertEqual(result["direction"], "NONE")
        self.assertEqual(result["trend_direction"], "SELL")
        self.assertGreaterEqual(result["trend_strength"], 0.4)

    def test_predictor_maps_expanded_features_to_legacy_model_schema(self):
        class ShapeCheckingModel:
            classes_ = np.array(["no_edge", "reversal_buy"])

            def predict_proba(self, X):
                if X.shape != (1, 2):
                    raise AssertionError(f"Unexpected model input shape: {X.shape}")
                np.testing.assert_allclose(X, [[2.0, 1.0]])
                return np.array([[0.2, 0.8]])

        predictor = ModelPredictor.__new__(ModelPredictor)
        predictor.threshold = 0.5
        predictor.bundle = {
            "model": ShapeCheckingModel(),
            "feature_names": ["legacy_second", "legacy_first"],
            "class_names": ["no_edge", "reversal_buy"],
            "version": "legacy_schema_test",
        }

        result = predictor.predict(
            np.array([[99.0, 98.0, 97.0]]),
            {"legacy_first": 1.0, "legacy_second": 2.0, "new_swing_feature": 3.0},
        )

        self.assertEqual(result["direction"], "BUY")

    def test_predictor_returns_direction_specific_interval_estimates(self):
        predictor = ModelPredictor.__new__(ModelPredictor)
        predictor.threshold = 0.5
        predictor.bundle = {
            "model": DummyDirectionalModel(),
            "class_names": list(DummyDirectionalModel.classes_),
            "version": "interval_test",
            "grid_interval_models": {
                "buy": ConstantIntervalModel(1.25),
                "sell": ConstantIntervalModel(8.0),
            },
            "swing_exit_models": {
                "buy_stop": ConstantIntervalModel(1.0),
                "buy_target": ConstantIntervalModel(2.0),
                "sell_stop": ConstantIntervalModel(1.5),
                "sell_target": ConstantIntervalModel(2.5),
            },
        }

        result = predictor.predict(np.array([[1.0, 2.0]], dtype=float))

        self.assertEqual(result["grid_interval_buy_atr"], 1.25)
        self.assertEqual(result["grid_interval_sell_atr"], 5.0)
        self.assertEqual(result["swing_stop_buy_atr"], 1.0)
        self.assertEqual(result["swing_target_buy_atr"], 2.0)
        self.assertEqual(result["swing_stop_sell_atr"], 1.5)
        self.assertEqual(result["swing_target_sell_atr"], 2.5)

    def test_predictor_loads_interval_heads_from_sidecar_without_replacing_classifier(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as model_dir:
            classifier_path = f"{model_dir}/classifier.joblib"
            interval_path = f"{model_dir}/intervals.joblib"
            joblib.dump({
                "model": DummyDirectionalModel(),
                "class_names": list(DummyDirectionalModel.classes_),
                "version": "classifier",
            }, classifier_path)
            joblib.dump({
                "grid_interval_models": {
                    "buy": ConstantIntervalModel(0.8),
                    "sell": ConstantIntervalModel(1.1),
                },
                "swing_exit_models": {
                    "buy_stop": ConstantIntervalModel(0.6),
                    "buy_target": ConstantIntervalModel(1.8),
                    "sell_stop": ConstantIntervalModel(0.7),
                    "sell_target": ConstantIntervalModel(1.9),
                },
            }, interval_path)

            predictor = ModelPredictor(classifier_path, 0.5, interval_path)
            result = predictor.predict(np.array([[1.0, 2.0]], dtype=float))

        self.assertEqual(predictor.bundle["version"], "classifier")
        self.assertEqual(result["grid_interval_buy_atr"], 0.8)
        self.assertEqual(result["grid_interval_sell_atr"], 1.1)
        self.assertEqual(result["swing_stop_buy_atr"], 0.6)
        self.assertEqual(result["swing_target_buy_atr"], 1.8)

    def test_model_trainer_fits_chronological_interval_regressors(self):
        from tempfile import TemporaryDirectory
        from ml.model_trainer import ModelTrainer

        rng = np.random.default_rng(42)
        X = rng.normal(size=(225, 3))
        y = np.tile(np.array(["buy", "sell", "flat"]), 75)
        targets = np.column_stack((np.linspace(0.2, 2.0, len(X)), np.linspace(0.3, 2.5, len(X))))
        with TemporaryDirectory() as model_dir:
            result = ModelTrainer(model_dir=model_dir).train(
                X, y, ["f1", "f2", "f3"], "interval_test", targets
            )
            bundle = joblib.load(result["path"])

        self.assertEqual(set(bundle["grid_interval_models"]), {"buy", "sell"})
        self.assertEqual(set(result["metrics"]["grid_interval_mae_atr"]), {"buy", "sell"})

    def test_interval_only_training_saves_sidecar_without_classifier(self):
        from tempfile import TemporaryDirectory
        from ml.model_trainer import ModelTrainer

        rng = np.random.default_rng(7)
        X = rng.normal(size=(225, 3))
        targets = np.column_stack((np.linspace(0.2, 1.5, len(X)), np.linspace(0.4, 2.0, len(X))))
        exit_targets = np.column_stack([np.linspace(0.2 + index * 0.1, 2.0 + index * 0.1, len(X)) for index in range(4)])
        with TemporaryDirectory() as model_dir:
            result = ModelTrainer(model_dir=model_dir).train_intervals(
                X, targets, ["f1", "f2", "f3"], "interval_sidecar_test", exit_targets
            )
            bundle = joblib.load(result["path"])

        self.assertNotIn("model", bundle)
        self.assertEqual(set(bundle["grid_interval_models"]), {"buy", "sell"})
        self.assertEqual(set(bundle["swing_exit_models"]), {"buy_stop", "buy_target", "sell_stop", "sell_target"})
        self.assertEqual(set(result["metrics"]["swing_exit_mae_atr"]), set(bundle["swing_exit_models"]))
        self.assertEqual(result["metrics"]["samples"], len(X))

    def test_model_trainer_exposes_multiclass_roc_auc(self):
        from ml.model_trainer import ModelTrainer

        rng = np.random.default_rng(42)
        X = np.column_stack([
            rng.normal(0, 1, 225),
            rng.normal(0, 1, 225),
            rng.normal(0, 1, 225),
        ])
        y = np.array([
            "reversal_buy", "reversal_sell", "continuation_buy", "continuation_sell",
            "breakout_buy", "breakout_sell", "failed_breakout_buy", "failed_breakout_sell", "no_edge",
        ] * 25)

        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as model_dir:
            trainer = ModelTrainer(model_dir=model_dir)
            result = trainer.train(X, y, ['f1', 'f2', 'f3'], version='roc_auc_regression_test')
            bundle = joblib.load(result['path'])

        self.assertIn('roc_auc', result['metrics'])
        self.assertGreaterEqual(float(result['metrics']['roc_auc']), 0.0)
        self.assertIn('reversal_buy', result['metrics']['classes'])
        self.assertEqual(bundle['model'].get_params()['class_weight'], 'balanced')
        self.assertEqual(
            set(result['metrics']['class_recall']),
            set(result['metrics']['classes']),
        )
        self.assertEqual(
            set(result['metrics']['class_precision']),
            set(result['metrics']['classes']),
        )

    def test_model_promotion_requires_reversal_class_quality(self):
        from ml.model_metrics import model_promotion_rejection_reasons

        walk_forward = {"mean_balanced_accuracy": 0.60}
        metrics = {
            "balanced_accuracy": 0.60,
            "roc_auc": 0.70,
            "class_support": {"reversal_buy": 30, "reversal_sell": 30},
            "class_recall": {"reversal_buy": 0.40, "reversal_sell": 0.10},
            "class_precision": {"reversal_buy": 0.50, "reversal_sell": 0.50},
        }

        self.assertEqual(
            model_promotion_rejection_reasons(walk_forward, metrics),
            ["reversal_sell recall below 0.20"],
        )

    def test_walk_forward_bounds_fits_with_expanding_chronological_windows(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(2000, 4))
        y = np.tile(np.array([
            'reversal_buy', 'reversal_sell', 'continuation_buy', 'continuation_sell',
            'breakout_buy', 'breakout_sell', 'failed_breakout_buy', 'failed_breakout_sell', 'no_edge',
        ]), 223)[:2000]

        result = WalkForwardValidator().validate(X, y)
        windows = result['windows']

        self.assertEqual(len(windows), 5)
        self.assertEqual(windows[0]['train_end'], 988)
        self.assertEqual(windows[0]['test_start'], 1000)
        self.assertEqual(windows[-1]['test_end'], len(X))
        for previous, current in zip(windows, windows[1:]):
            self.assertEqual(current['test_start'], previous['test_end'])
            self.assertLessEqual(current['train_end'], current['test_start'] - 12)
            self.assertGreater(current['test_end'], previous['test_end'])


if __name__ == "__main__":
    unittest.main()
