from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, precision_score, recall_score
from core.trend_engine import MARKET_TREND_LOOKBACK
from ml.model_metrics import weighted_ovr_roc_auc


class ModelTrainer:
    def __init__(self, model_dir='models'):
        self.dir = Path(model_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _fit_regression_heads(X, targets, names, train_end, test_start):
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
            regressor.fit(X[:train_end], targets[:train_end, index])
            models[name] = regressor
            errors[name] = float(mean_absolute_error(
                targets[test_start:, index], regressor.predict(X[test_start:])
            ))
        return models, errors

    def train_trend(
        self,
        X,
        y,
        feature_names,
        version='trend_candidate',
        groups=None,
        class_weight='balanced',
        range_weight_scale=1.0,
    ):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if len(X) < 200 or y.shape != (len(X),) or not np.isfinite(X).all():
            raise ValueError('At least 200 finite feature rows and one market-trend label per row are required.')
        if groups is not None and len(groups) != len(X):
            raise ValueError('Trend training groups must contain one value per feature row.')
        if class_weight not in {'balanced', None}:
            raise ValueError("Trend class_weight must be 'balanced' or None.")
        if not np.isfinite(range_weight_scale) or not 0 < range_weight_scale <= 1:
            raise ValueError('Trend range_weight_scale must be greater than 0 and at most 1.')
        expected_classes = {'BUY', 'SELL', 'RANGE'}
        if set(y) != expected_classes:
            raise ValueError('Trend training labels must include BUY, SELL, and RANGE.')
        if groups is None:
            split = int(len(X) * 0.8)
            if split <= 0 or split >= len(X):
                raise ValueError('Insufficient history for a chronological trend-model holdout.')
            train_indices = np.arange(split)
            test_indices = np.arange(split, len(X))
            purged_samples = 0
        else:
            groups = np.asarray(groups)
            train_parts = []
            test_parts = []
            purged_samples = 0
            purge_per_group = MARKET_TREND_LOOKBACK // 100
            for group in np.unique(groups):
                indices = np.flatnonzero(groups == group)
                split = int(len(indices) * 0.8)
                purge = min(purge_per_group, max(0, split - 1))
                train_end = split - purge
                if train_end < 1 or split >= len(indices):
                    raise ValueError(
                        f'Insufficient grouped history for trend holdout: {group}.'
                    )
                train_parts.append(indices[:train_end])
                test_parts.append(indices[split:])
                purged_samples += purge
            train_indices = np.sort(np.concatenate(train_parts))
            test_indices = np.sort(np.concatenate(test_parts))
        train_y, test_y = y[train_indices], y[test_indices]
        if set(train_y) != expected_classes or set(test_y) != expected_classes:
            raise ValueError('The chronological trend-model train and test splits must each include BUY, SELL, and RANGE.')
        trend_weights = class_weight
        trend_sample_weights = None
        if class_weight == 'balanced' and range_weight_scale < 1:
            class_counts = {
                label: int(np.sum(train_y == label)) for label in expected_classes
            }
            trend_weights = {
                label: len(train_y) / (len(expected_classes) * class_counts[label])
                * (range_weight_scale if label == 'RANGE' else 1.0)
                for label in expected_classes
            }
            trend_sample_weights = np.asarray(
                [trend_weights[label] for label in train_y], dtype=float
            )
            trend_weights = None

        model = HistGradientBoostingClassifier(
            learning_rate=.04,
            max_iter=350,
            max_leaf_nodes=15,
            min_samples_leaf=15,
            l2_regularization=1.0,
            class_weight=trend_weights,
            random_state=42,
        )
        model.fit(
            X[train_indices], train_y, sample_weight=trend_sample_weights
        )
        test_prediction = model.predict(X[test_indices])
        test_probabilities = model.predict_proba(X[test_indices])
        classes = [str(label) for label in model.classes_]
        precision = precision_score(
            test_y, test_prediction, labels=model.classes_, average=None, zero_division=0
        )
        recall = recall_score(
            test_y, test_prediction, labels=model.classes_, average=None, zero_division=0
        )
        metrics = {
            'trend_lookback': MARKET_TREND_LOOKBACK,
            'trend_class_weight': (
                'balanced' if class_weight == 'balanced' else 'none'
            ),
            'trend_range_weight_scale': range_weight_scale,
            'purged_samples': purged_samples,
            'accuracy': float(accuracy_score(test_y, test_prediction)),
            'balanced_accuracy': float(balanced_accuracy_score(test_y, test_prediction)),
            'roc_auc': weighted_ovr_roc_auc(test_y, test_probabilities, model.classes_),
            'training_samples': int(len(train_indices)),
            'chronological_test_samples': int(len(test_indices)),
            'classes': classes,
            'class_precision': {
                label: float(value) for label, value in zip(classes, precision)
            },
            'class_recall': {
                label: float(value) for label, value in zip(classes, recall)
            },
            'test_class_support': {
                label: int(np.sum(test_y == label)) for label in classes
            },
            'test_confusion_matrix': [
                [
                    int(np.sum((test_y == actual) & (test_prediction == predicted)))
                    for predicted in model.classes_
                ]
                for actual in model.classes_
            ],
        }
        if groups is not None:
            group_values = groups[test_indices]
            metrics['holdout_by_group'] = {}
            for group in np.unique(group_values):
                group_mask = group_values == group
                group_recall = recall_score(
                    test_y[group_mask],
                    test_prediction[group_mask],
                    labels=sorted(expected_classes),
                    average=None,
                    zero_division=0,
                )
                metrics['holdout_by_group'][str(group)] = {
                    'samples': int(group_mask.sum()),
                    'accuracy': float(accuracy_score(
                        test_y[group_mask], test_prediction[group_mask]
                    )),
                    'balanced_accuracy': float(np.mean(group_recall)),
                }
        final_model = HistGradientBoostingClassifier(
            learning_rate=.04,
            max_iter=350,
            max_leaf_nodes=15,
            min_samples_leaf=15,
            l2_regularization=1.0,
            class_weight=trend_weights,
            random_state=42,
        )
        final_sample_weights = None
        final_class_weights = None
        if trend_weights is None and trend_sample_weights is not None:
            all_class_counts = {
                label: int(np.sum(y == label)) for label in expected_classes
            }
            final_class_weights = {
                label: len(y) / (len(expected_classes) * all_class_counts[label])
                * (range_weight_scale if label == 'RANGE' else 1.0)
                for label in expected_classes
            }
            final_sample_weights = np.asarray(
                [final_class_weights[label] for label in y], dtype=float
            )
        final_model.fit(X, y, sample_weight=final_sample_weights)
        bundle = {
            'trend_model': final_model,
            'trend_class_names': [str(label) for label in final_model.classes_],
            'feature_names': list(feature_names),
            'trend_lookback': MARKET_TREND_LOOKBACK,
            'trend_class_weight': (
                'balanced' if class_weight == 'balanced' else 'none'
            ),
            'trend_range_weight_scale': range_weight_scale,
            'trend_class_weights': final_class_weights,
            'training_groups': sorted(
                {str(group) for group in groups}
            ) if groups is not None else [],
            'version': version,
        }
        path = self.dir / f'{version}.joblib'
        joblib.dump(bundle, path)
        (self.dir / f'{version}.json').write_text(json.dumps(metrics, indent=2))
        return {'path': str(path), 'version': version, 'metrics': metrics}

    def train_intervals(self, X, targets, feature_names, version='interval_candidate', swing_targets=None, purge=12):
        X = np.asarray(X, dtype=float)
        targets = np.asarray(targets, dtype=float)
        if len(X) < 200 or targets.shape != (len(X), 2) or not np.isfinite(targets).all():
            raise ValueError('At least 200 rows and finite buy/sell interval targets are required.')
        test_start = int(len(X) * 0.8)
        train_end = test_start - purge
        if train_end <= 0 or test_start >= len(X):
            raise ValueError('Insufficient history for a valid purged chronological split.')
        interval_models, interval_mae = self._fit_regression_heads(
            X, targets, ('buy', 'sell'), train_end, test_start
        )
        metrics = {
            'grid_interval_mae_atr': interval_mae,
            'samples': len(X),
            'purged_samples': purge,
            'training_samples': train_end,
            'chronological_test_samples': len(X) - test_start,
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
            exit_models, exit_mae = self._fit_regression_heads(
                X, exit_targets, exit_names, train_end, test_start
            )
            bundle['swing_exit_models'] = exit_models
            metrics['swing_exit_mae_atr'] = exit_mae
        path = self.dir / f'{version}.joblib'
        joblib.dump(bundle, path)
        (self.dir / f'{version}.json').write_text(json.dumps(metrics, indent=2))
        return {'path': str(path), 'version': version, 'metrics': metrics}

    def train(
        self,
        X,
        y,
        feature_names,
        version='candidate',
        interval_targets=None,
        purge=12,
        swing_targets=None,
    ):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if len(X) < 200:
            raise ValueError('At least 200 labelled historical examples are required.')
        test_start = int(len(X) * 0.8)
        train_end = test_start - purge
        if train_end <= 0 or test_start >= len(X):
            raise ValueError('Insufficient history for a valid purged chronological split.')
        train_y = y[:train_end]
        test_y = y[test_start:]
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
        model.fit(X[:train_end], train_y)
        pred = model.predict(X[test_start:])
        proba = model.predict_proba(X[test_start:])
        roc_auc = weighted_ovr_roc_auc(test_y, proba, model.classes_)
        recalls = recall_score(test_y, pred, labels=model.classes_, average=None, zero_division=0)
        precisions = precision_score(test_y, pred, labels=model.classes_, average=None, zero_division=0)
        supports = [(test_y == label).sum() for label in model.classes_]
        metrics = {
            'accuracy': float(accuracy_score(test_y, pred)),
            'balanced_accuracy': float(balanced_accuracy_score(test_y, pred)) if len(np.unique(test_y)) > 1 else 0.5,
            'roc_auc': roc_auc,
            'purged_samples': purge,
            'training_samples': train_end,
            'chronological_test_samples': len(X) - test_start,
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
            interval_models, interval_mae = self._fit_regression_heads(
                X, targets, ('buy', 'sell'), train_end, test_start
            )
            bundle['grid_interval_models'] = interval_models
            metrics['grid_interval_mae_atr'] = interval_mae
        if swing_targets is not None:
            exit_targets = np.asarray(swing_targets, dtype=float)
            exit_names = ('buy_stop', 'buy_target', 'sell_stop', 'sell_target')
            if exit_targets.shape != (len(X), len(exit_names)) or not np.isfinite(exit_targets).all():
                raise ValueError('Swing exit targets must contain finite buy/sell SL/TP values for every sample.')
            exit_models, exit_mae = self._fit_regression_heads(
                X, exit_targets, exit_names, train_end, test_start
            )
            bundle['swing_exit_models'] = exit_models
            metrics['swing_exit_mae_atr'] = exit_mae
        path = self.dir / f'{version}.joblib'
        joblib.dump(bundle, path)
        (self.dir / f'{version}.json').write_text(json.dumps(metrics, indent=2))
        return {'path': str(path), 'version': version, 'metrics': metrics}
