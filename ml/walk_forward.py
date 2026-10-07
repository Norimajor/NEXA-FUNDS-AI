from __future__ import annotations
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from ml.model_metrics import weighted_ovr_roc_auc


class WalkForwardValidator:
    def validate(self, X, y, initial_train=250, test_size=50, step=50, max_windows=5, purge=12):
        X, y = np.asarray(X, float), np.asarray(y)
        out = []
        end = max(initial_train, len(X) // 2)
        remaining = len(X) - end
        if remaining <= 0:
            raise ValueError('Insufficient examples for walk-forward validation.')
        window_count = min(max_windows, max(1, 1 + max(0, remaining - test_size) // step))
        boundaries = np.linspace(end, len(X), window_count + 1, dtype=int)
        for index in range(window_count):
            test_start, test_end = boundaries[index:index + 2]
            train_end = test_start - purge
            if train_end <= 0:
                raise ValueError('Insufficient training history after applying the validation purge.')
            m = HistGradientBoostingClassifier(
                learning_rate=.04,
                max_iter=300,
                max_leaf_nodes=15,
                min_samples_leaf=15,
                l2_regularization=1,
                class_weight='balanced',
                random_state=42,
            )
            m.fit(X[:train_end], y[:train_end])
            prob = m.predict_proba(X[test_start:test_end])
            yt = y[test_start:test_end]
            pred = np.asarray(m.classes_)[prob.argmax(axis=1)]
            item = {
                'train_end': int(train_end),
                'test_start': int(test_start),
                'test_end': int(test_end),
                'balanced_accuracy': float(balanced_accuracy_score(yt, pred)) if len(np.unique(yt)) > 1 else 0.5,
            }
            item['roc_auc'] = weighted_ovr_roc_auc(yt, prob, m.classes_)
            out.append(item)
        if not out:
            raise ValueError('Insufficient examples for walk-forward validation.')
        return {
            'windows': out,
            'mean_balanced_accuracy': float(np.mean([x['balanced_accuracy'] for x in out])),
            'mean_roc_auc': float(np.mean([x['roc_auc'] for x in out])),
        }
