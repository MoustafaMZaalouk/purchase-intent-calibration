"""Metrics: PR-AUC, Brier score, calibration error and threshold choice.

The decision threshold is chosen on the VALIDATION split only and then
frozen before the test split is touched. That is the whole point of the
project: an honest held-out estimate of a decision rule, not a score
that was tuned on the data it reports.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)

_EPS = 1e-15


def expected_calibration_error(
    y_true: np.ndarray, scores: np.ndarray, n_bins: int = 10
) -> float:
    """Equal-width-bin expected calibration error (ECE).

    Bins span [0, 1] uniformly. For each non-empty bin, ECE accumulates
    (bin fraction) * |mean predicted score - empirical positive rate|.
    Empty bins contribute nothing. Returns a value in [0, 1]; lower is
    better calibrated.
    """
    y_true = np.asarray(y_true, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if y_true.shape != scores.shape:
        raise ValueError("y_true and scores must have the same shape")
    if y_true.size == 0:
        raise ValueError("cannot calibrate an empty array")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("scores must be probabilities in [0, 1]")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Rightmost bin closed on both sides so a score of exactly 1.0 lands
    # in a bin instead of being dropped.
    bin_index = np.clip(np.digitize(scores, edges[1:-1], right=True), 0, n_bins - 1)

    ece = 0.0
    n = scores.size
    for b in range(n_bins):
        mask = bin_index == b
        count = int(mask.sum())
        if count == 0:
            continue
        confidence = float(scores[mask].mean())
        accuracy = float(y_true[mask].mean())
        ece += (count / n) * abs(confidence - accuracy)
    return float(ece)


def select_threshold_f1(
    y_true: np.ndarray, scores: np.ndarray
) -> tuple[float, float]:
    """Pick the threshold that maximises F1 on the given (validation) data.

    A score is predicted positive when score >= threshold. Candidates are
    the thresholds from sklearn's precision_recall_curve, i.e. every
    distinct score value. Ties resolve to the lowest threshold (first in
    ascending order) for determinism. Returns (threshold, best_f1).
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if y_true.size == 0:
        raise ValueError("cannot choose a threshold on empty data")
    if y_true.sum() == 0:
        raise ValueError("cannot choose an F1 threshold with no positives")

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # precision[i], recall[i] belong to thresholds[i]; the curve also
    # carries one trailing (precision=1, recall=0) point with no
    # threshold, which is dropped by aligning on thresholds.
    denom = precision[:-1] + recall[:-1]
    f1 = np.zeros_like(denom)
    np.divide(2.0 * precision[:-1] * recall[:-1], denom, out=f1,
              where=denom > 0)
    best = int(np.argmax(f1))  # first max, thresholds ascending -> lowest
    return float(thresholds[best]), float(f1[best])


def safe_log_loss(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Log loss with probability clipping so a confident 0/1 cannot inf out."""
    clipped = np.clip(np.asarray(scores, dtype=float), _EPS, 1.0 - _EPS)
    return float(log_loss(y_true, clipped))


def threshold_metrics(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> dict:
    """Precision, recall, F1 and predicted-positive rate at a threshold."""
    y_true = np.asarray(y_true, dtype=int)
    predicted = (np.asarray(scores, dtype=float) >= threshold).astype(int)
    tp = int(((predicted == 1) & (y_true == 1)).sum())
    fp = int(((predicted == 1) & (y_true == 0)).sum())
    fn = int(((predicted == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "predicted_positive_rate": float(predicted.mean()),
    }


def evaluate_scores(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float | None = None,
    n_cal_bins: int = 10,
) -> dict:
    """Full metric bundle for one model on one split.

    threshold=None reports the ranking/probability metrics only; pass a
    validation-chosen threshold to add the decision-rule metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    result = {
        "n": int(y_true.size),
        "positive_rate": float(y_true.mean()),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "brier": float(brier_score_loss(y_true, scores)),
        "log_loss": safe_log_loss(y_true, scores),
        "ece": expected_calibration_error(y_true, scores, n_bins=n_cal_bins),
    }
    if threshold is not None:
        result["at_threshold"] = threshold_metrics(y_true, scores, threshold)
    return result
