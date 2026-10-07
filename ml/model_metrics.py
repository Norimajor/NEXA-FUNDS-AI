from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def weighted_ovr_roc_auc(y_true, probabilities, classes) -> float:
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    classes = np.asarray(classes)
    labels, counts = np.unique(y_true, return_counts=True)
    if len(labels) < 2:
        return 0.5

    class_indices = {label: index for index, label in enumerate(classes)}
    aucs = []
    weights = []
    for label, count in zip(labels, counts):
        index = class_indices.get(label)
        scores = probabilities[:, index] if index is not None else np.zeros(len(y_true))
        aucs.append(float(roc_auc_score(y_true == label, scores)))
        weights.append(int(count))

    return float(np.average(aucs, weights=weights))


def model_promotion_rejection_reasons(walk_forward, metrics):
    reasons = []
    if walk_forward['mean_balanced_accuracy'] < 0.55:
        reasons.append('walk-forward balanced accuracy below 0.55')
    if metrics['balanced_accuracy'] < 0.55:
        reasons.append('chronological holdout balanced accuracy below 0.55')
    if metrics['roc_auc'] < 0.55:
        reasons.append('chronological holdout ROC AUC below 0.55')
    for class_name in ('reversal_buy', 'reversal_sell'):
        if metrics.get('class_support', {}).get(class_name, 0) < 20:
            reasons.append(f'{class_name} has fewer than 20 holdout examples')
        if metrics.get('class_recall', {}).get(class_name, 0.0) < 0.20:
            reasons.append(f'{class_name} recall below 0.20')
        if metrics.get('class_precision', {}).get(class_name, 0.0) < 0.40:
            reasons.append(f'{class_name} precision below 0.40')
    return reasons