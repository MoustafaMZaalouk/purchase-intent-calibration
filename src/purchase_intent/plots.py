"""Reproducible figures for the report and README.

Every figure is regenerated from a fresh experiment run by
scripts/make_plots.py - nothing in results/ is edited by hand.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless rendering, no display needed

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

MODEL_LABELS = {
    "logistic_regression": "Logistic regression (linear baseline)",
    "gradient_boosting": "Gradient boosting",
}

CALIBRATION_STYLES = (
    ("raw", "raw scores", "C0"),
    ("platt", "after Platt scaling", "C2"),
    ("isotonic", "after isotonic regression", "C3"),
)


def plot_pr_curves(scores: dict, out_path: str) -> None:
    """Precision-recall curves on the TEST split for both models.

    The marker on each curve is the operating point from the threshold
    chosen on validation, so the plot shows the actual decision rule,
    not a test-tuned one.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, split_scores in scores.items():
        y_test, s_test = split_scores["test"]
        precision, recall, _ = precision_recall_curve(y_test, s_test)
        ax.plot(recall, precision, lw=1.8, label=MODEL_LABELS.get(name, name))

        thr = split_scores["threshold"]
        predicted = s_test >= thr
        tp = int((predicted & (y_test == 1)).sum())
        fp = int((predicted & (y_test == 0)).sum())
        fn = int((~predicted & (y_test == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        ax.plot([r], [p], "o", ms=8,
                label=f"{MODEL_LABELS.get(name, name)} @ val-chosen t={thr:.3f}")

    base_rate = float(next(iter(scores.values()))["test"][0].mean())
    ax.axhline(base_rate, ls="--", c="grey", lw=1,
               label=f"prevalence baseline ({base_rate:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall on held-out test months (Oct-Dec)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _binned_calibration(y: np.ndarray, s: np.ndarray, n_bins: int):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(s, edges[1:-1], right=True), 0, n_bins - 1)
    conf, acc = [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() >= 5:  # near-empty bins add noise, not signal
            conf.append(float(s[mask].mean()))
            acc.append(float(y[mask].mean()))
    return conf, acc


def plot_reliability(scores: dict, out_path: str, n_bins: int = 10) -> None:
    """Reliability diagrams before/after calibration, one panel per model.

    Each panel shows the raw test scores, the Platt-scaled scores and
    the isotonic-calibrated scores (both calibrators fitted on
    validation only) against the perfect-calibration diagonal.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    for ax, (name, split_scores) in zip(axes, scores.items()):
        y_test, s_test = split_scores["test"]
        variants = {
            "raw": s_test,
            "platt": split_scores["test_platt"],
            "isotonic": split_scores["test_isotonic"],
        }
        for key, label, color in CALIBRATION_STYLES:
            conf, acc = _binned_calibration(y_test, variants[key], n_bins)
            ax.plot(conf, acc, "o-", lw=1.6, ms=4.5, color=color, label=label)
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        ax.set_title(MODEL_LABELS.get(name, name), fontsize=10)
        ax.set_xlabel("Mean predicted probability (bin)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Observed positive rate")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.suptitle("Calibration on held-out test months, before/after "
                 "validation-fitted recalibration", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
