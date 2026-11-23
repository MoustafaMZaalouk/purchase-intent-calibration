"""Post-hoc probability calibration: Platt scaling and isotonic regression.

Both calibrators are fitted on VALIDATION scores only and then applied
to frozen test scores, so the reported calibration improvement is a
held-out estimate, not an in-sample one.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """Logistic regression on the model's score (Platt, 1999)."""

    def __init__(self) -> None:
        self._lr = LogisticRegression()

    def fit(self, y_true: np.ndarray, scores: np.ndarray) -> PlattCalibrator:
        scores = np.asarray(scores, dtype=float).reshape(-1, 1)
        self._lr.fit(scores, np.asarray(y_true, dtype=int))
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float).reshape(-1, 1)
        return self._lr.predict_proba(scores)[:, 1]


class IsotonicCalibrator:
    """Isotonic (monotone step) regression from scores to frequencies."""

    def __init__(self) -> None:
        self._iso = IsotonicRegression(y_min=0.0, y_max=1.0,
                                       out_of_bounds="clip")

    def fit(self, y_true: np.ndarray, scores: np.ndarray) -> IsotonicCalibrator:
        self._iso.fit(np.asarray(scores, dtype=float),
                      np.asarray(y_true, dtype=float))
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return self._iso.predict(np.asarray(scores, dtype=float))


def fit_calibrators(y_val: np.ndarray, s_val: np.ndarray) -> dict:
    """Fit both calibrators on validation scores."""
    return {
        "platt": PlattCalibrator().fit(y_val, s_val),
        "isotonic": IsotonicCalibrator().fit(y_val, s_val),
    }
