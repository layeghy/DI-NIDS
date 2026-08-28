"""Unambiguous binary metrics for One-Class SVM output."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def to_one_class_labels(binary_labels: np.ndarray) -> np.ndarray:
    """Convert dataset labels (0 benign, 1 attack) to OSVM labels (+1, -1)."""

    labels = np.asarray(binary_labels)
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Binary labels must contain only 0 and 1")
    return np.where(labels == 0, 1, -1).astype(np.int8)


def one_class_metrics(
    expected: np.ndarray,
    predicted: np.ndarray,
    *,
    decision_scores: np.ndarray | None = None,
) -> dict[str, object]:
    """Return attack-positive and benign-positive scores separately."""

    expected = np.asarray(expected)
    predicted = np.asarray(predicted)
    if expected.shape != predicted.shape:
        raise ValueError("Expected and predicted labels must have the same shape")

    precision, recall, f1, support = precision_recall_fscore_support(
        expected,
        predicted,
        labels=[-1, 1],
        zero_division=0,
    )
    matrix = confusion_matrix(expected, predicted, labels=[-1, 1])
    result: dict[str, object] = {
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": float(np.mean(f1)),
        "attack": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0]),
        },
        "benign": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int(support[1]),
        },
        "confusion_matrix": {
            "attack_as_attack": int(matrix[0, 0]),
            "attack_as_benign": int(matrix[0, 1]),
            "benign_as_attack": int(matrix[1, 0]),
            "benign_as_benign": int(matrix[1, 1]),
        },
    }

    if decision_scores is not None and len(np.unique(expected)) == 2:
        anomaly_scores = -np.asarray(decision_scores)
        expected_attack = (expected == -1).astype(np.int8)
        result["attack_roc_auc"] = float(roc_auc_score(expected_attack, anomaly_scores))
    else:
        result["attack_roc_auc"] = None
    return result
