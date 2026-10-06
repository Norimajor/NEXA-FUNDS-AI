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