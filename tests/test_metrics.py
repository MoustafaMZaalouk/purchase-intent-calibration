"""Metric unit tests with hand-computed expected values."""
from __future__ import annotations

import numpy as np
import pytest

from purchase_intent.metrics import (
    expected_calibration_error,
    safe_log_loss,
    select_threshold_f1,
    threshold_metrics,
)


class TestExpectedCalibrationError:
    def test_hand_computed_value(self):
        # One sample per bin of width 0.25:
        # |0.1-0| + |0.3-1| + |0.6-0| + |0.8-1|, each weighted 1/4.
        y = np.array([0, 1, 0, 1])
        s = np.array([0.1, 0.3, 0.6, 0.8])
        assert expected_calibration_error(y, s, n_bins=4) == pytest.approx(0.4)

    def test_perfect_calibration_is_zero(self):
        y = np.array([0, 0, 0, 1, 1, 1, 0, 1])
        s = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5])
        # bins: 0.0 -> rate 0, 0.5 -> rate 0.5, 1.0 -> rate 1
        assert expected_calibration_error(y, s, n_bins=10) == pytest.approx(0.0)

    def test_worst_calibration_is_one(self):
        y = np.array([0, 1])
        s = np.array([1.0, 0.0])  # confident and exactly wrong
        assert expected_calibration_error(y, s, n_bins=10) == pytest.approx(1.0)

    def test_bounds_on_random_data(self):
        rng = np.random.default_rng(1)
        s = rng.random(500)
        y = (rng.random(500) < s).astype(int)
        ece = expected_calibration_error(y, s, n_bins=10)
        assert 0.0 <= ece <= 1.0

    def test_score_exactly_one_lands_in_a_bin(self):
        y = np.array([1])
        s = np.array([1.0])
        assert expected_calibration_error(y, s, n_bins=10) == pytest.approx(0.0)

    def test_rejects_out_of_range_scores(self):
        with pytest.raises(ValueError, match="probabilities"):
            expected_calibration_error(np.array([0]), np.array([1.5]))

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="empty"):
            expected_calibration_error(np.array([]), np.array([]))


class TestSelectThresholdF1:
    def test_matches_brute_force(self):
        """The chosen threshold must achieve the brute-force best F1."""
        rng = np.random.default_rng(7)
        s = rng.random(400)
        y = (s + rng.normal(0, 0.3, 400) > 0.6).astype(int)

        threshold, f1 = select_threshold_f1(y, s)

        best_f1 = 0.0
        for candidate in np.unique(s):
            best_f1 = max(best_f1, threshold_metrics(y, s, candidate)["f1"])
        assert f1 == pytest.approx(best_f1)
        assert threshold_metrics(y, s, threshold)["f1"] == pytest.approx(f1)

    def test_perfect_scores_give_perfect_f1(self):
        y = np.array([0, 0, 1, 1])
        s = np.array([0.1, 0.2, 0.8, 0.9])
        threshold, f1 = select_threshold_f1(y, s)
        assert f1 == pytest.approx(1.0)
        assert 0.2 <= threshold <= 0.8

    def test_no_positives_rejected(self):
        with pytest.raises(ValueError, match="no positives"):
            select_threshold_f1(np.zeros(10, dtype=int), np.linspace(0, 1, 10))

    def test_deterministic(self):
        rng = np.random.default_rng(3)
        s = rng.random(200)
        y = (rng.random(200) < 0.3).astype(int)
        assert select_threshold_f1(y, s) == select_threshold_f1(y, s)


class TestThresholdMetrics:
    def test_hand_computed_counts(self):
        y = np.array([1, 1, 0, 0, 1])
        s = np.array([0.9, 0.4, 0.6, 0.1, 0.7])
        m = threshold_metrics(y, s, 0.5)
        # predicted positive: rows 0, 2, 4 -> tp=2, fp=1, fn=1
        assert m["precision"] == pytest.approx(2 / 3)
        assert m["recall"] == pytest.approx(2 / 3)
        assert m["f1"] == pytest.approx(2 / 3)
        assert m["predicted_positive_rate"] == pytest.approx(3 / 5)

    def test_no_predictions_is_zero_not_nan(self):
        y = np.array([1, 0])
        s = np.array([0.1, 0.2])
        m = threshold_metrics(y, s, 0.9)
        assert m["precision"] == 0.0
        assert m["f1"] == 0.0
        assert m["predicted_positive_rate"] == 0.0


class TestSafeLogLoss:
    def test_confident_wrong_prediction_is_finite(self):
        loss = safe_log_loss(np.array([0, 1]), np.array([1.0, 0.0]))
        assert np.isfinite(loss)
        assert loss > 10  # heavily penalised, but not inf
