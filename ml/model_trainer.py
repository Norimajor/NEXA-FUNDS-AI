from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, precision_score, recall_score
from ml.model_metrics import weighted_ovr_roc_auc


class ModelTrainer:
    def __init__(self, model_dir='models'):
        self.dir = Path(model_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _fit_regression_heads(X, targets, names, split):
        models = {}
        errors = {}
        for index, name in enumerate(names):
            regressor = HistGradientBoostingRegressor(
                loss='absolute_error',
                learning_rate=.04,
                max_iter=250,
                max_leaf_nodes=15,
                min_samples_leaf=15,
                l2_regularization=1.0,
                random_state=42,
            )
            regressor.fit(X[:split], targets[:split, index])
            models[name] = regressor
            errors[name] = float(mean_absolute_error(
                targets[split:, index], regressor.predict(X[split:])
            ))
        return models, errors

    def train_intervals(self, X, targets, feature_names, version='interval_candidate', swing_targets=None, purge=12):
        X = np.asarray(X, dtype=float)
        targets = np.asarray(targets, dtype=float)
        if len(X) < 200 or targets.shape != (len(X), 2) or not np.isfinite(targets).all():
            raise ValueError('At least 200 rows and finite buy/sell interval targets are required.')
        split = int(len(X) * 0.8) - purge
        if split <= 0 or split >= len(X):
            raise ValueError('Insufficient history for a valid purged chronological split.')
        interval_models, interval_mae = self._fit_regression_heads(X, targets, ('buy', 'sell'), split)
        metrics = {
            'grid_interval_mae_atr': interval_mae,
            'samples': len(X),
            'purged_samples': purge,
            'chronological_test_samples': len(X) - split,
        }
        bundle = {
            'grid_interval_models': interval_models,
            'feature_names': list(feature_names),
            'version': version,
        }
        if swing_targets is not None:
            exit_targets = np.asarray(swing_targets, dtype=float)
            exit_names = ('buy_stop', 'buy_target', 'sell_stop', 'sell_target')
            if exit_targets.shape != (len(X), len(exit_names)) or not np.isfinite(exit_targets).all():
                raise ValueError('Swing exit targets must contain finite buy/sell SL/TP values for every sample.')
            exit_models, exit_mae = self._fit_regression_heads(X, exit_targets, exit_names, split)
            bundle['swing_exit_models'] = exit_models
            metrics['swing_exit_mae_atr'] = exit_mae
        path = self.dir / f'{version}.joblib'
        joblib.dump(bundle, path)
        (self.dir / f'{version}.json').write_text(json.dumps(metrics, indent=2))
        return {'path': str(path), 'version': version, 'metrics': metrics}

    def train(self, X, y, feature_names, version='candidate', interval_targets=None, purge=12):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if len(X) < 200:
            raise ValueError('At least 200 labelled historical examples are required.')
        split = int(len(X) * 0.8) - purge
        if split <= 0 or split >= len(X):
            raise ValueError('Insufficient history for a valid purged chronological split.')
        train_y = y[:split]
        test_y = y[split:]
        if len(np.unique(train_y)) < 2:
            raise ValueError('At least two classes must exist in the chronological training set.')

        model = HistGradientBoostingClassifier(
            learning_rate=.04,
            max_iter=350,
            max_leaf_nodes=15,
            min_samples_leaf=15,
            l2_regularization=1.0,
            class_weight='balanced',
            random_state=42,
        )
        model.fit(X[:split], train_y)
        pred = model.predict(X[split:])
        proba = model.predict_proba(X[split:])
        roc_auc = weighted_ovr_roc_auc(test_y, proba, model.classes_)
        recalls = recall_score(test_y, pred, labels=model.classes_, average=None, zero_division=0)
        precisions = precision_score(test_y, pred, labels=model.classes_, average=None, zero_division=0)
        supports = [(test_y == label).sum() for label in model.classes_]
        metrics = {
            'accuracy': float(accuracy_score(test_y, pred)),
            'balanced_accuracy': float(balanced_accuracy_score(test_y, pred)) if len(np.unique(test_y)) > 1 else 0.5,
            'roc_auc': roc_auc,
            'purged_samples': purge,
            'chronological_test_samples': len(X) - split,
            'classes': [str(c) for c in model.classes_],
            'class_recall': {str(label): float(value) for label, value in zip(model.classes_, recalls)},
            'class_precision': {str(label): float(value) for label, value in zip(model.classes_, precisions)},
            'class_support': {str(label): int(value) for label, value in zip(model.classes_, supports)},
        }

        bundle = {
            'model': model,
            'feature_names': list(feature_names),
            'class_names': [str(c) for c in model.classes_],
            'version': version,
        }
        if interval_targets is not None:
            targets = np.asarray(interval_targets, dtype=float)
            if targets.shape != (len(X), 2) or not np.isfinite(targets).all():
                raise ValueError('Interval targets must contain finite buy and sell values for every sample.')
            interval_models, interval_mae = self._fit_regression_heads(X, targets, ('buy', 'sell'), split)
            bundle['grid_interval_models'] = interval_models
            metrics['grid_interval_mae_atr'] = interval_mae
        path = self.dir / f'{version}.joblib'
        joblib.dump(bundle, path)
        (self.dir / f'{version}.json').write_text(json.dumps(metrics, indent=2))
        return {'path': str(path), 'version': version, 'metrics': metrics}
