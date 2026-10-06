import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from ml.dataset_builder import DatasetBuilder
from ml.predictor import ModelPredictor
from ml.walk_forward import WalkForwardValidator


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
        dataset = DatasetBuilder(horizon=6, stop_atr=1.0, target_atr=1.5).build(self._make_candles(420))
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

        self.assertEqual(dataset['timestamp'].tolist(), list(timestamps[250:297:10]))

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

        trainer = ModelTrainer(model_dir='models')
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
        self.assertEqual(windows[0]['train_end'], 1000)
        self.assertEqual(windows[-1]['test_end'], len(X))
        for previous, current in zip(windows, windows[1:]):
            self.assertEqual(current['train_end'], previous['test_end'])
            self.assertGreater(current['test_end'], previous['test_end'])


if __name__ == "__main__":
    unittest.main()
