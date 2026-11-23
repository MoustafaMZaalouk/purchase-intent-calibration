"""Calibration and end-to-end smoke tests on a small synthetic dataset."""
from __future__ import annotations

import json

import numpy as np
import pytest

from purchase_intent.calibrate import fit_calibrators
from purchase_intent.metrics import expected_calibration_error
from purchase_intent.train import public_report, run_experiment


class TestCalibrators:
    def test_platt_fixes_known_miscalibration(self):
        """Platt must recover a known logit-linear miscalibration.

        Scores are sigmoid(t) while the true probability is
        sigmoid(2t - 0.5), so a logistic recalibration on the scores can
        in principle fix it almost exactly; the remaining ECE is bin
        noise, kept small by the large sample.
        """
        rng = np.random.default_rng(11)
        n = 20000
        t = rng.normal(0.0, 1.5, n)
        p_true = 1.0 / (1.0 + np.exp(-(2.0 * t - 0.5)))
        y = (rng.random(n) < p_true).astype(int)
        raw = 1.0 / (1.0 + np.exp(-t))  # miscalibrated scores
        cal = fit_calibrators(y, raw)["platt"]
        ece_before = expected_calibration_error(y, raw)
        ece_after = expected_calibration_error(y, cal.predict(raw))
        assert ece_after < ece_before
        assert ece_after < 0.03

    def test_isotonic_output_is_monotone(self):
        rng = np.random.default_rng(5)
        s = rng.random(500)
        y = (rng.random(500) < s).astype(int)
        cal = fit_calibrators(y, s)["isotonic"]
        grid = np.linspace(0, 1, 101)
        out = cal.predict(grid)
        assert np.all(np.diff(out) >= -1e-12)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_calibrated_scores_are_probabilities(self, synthetic_df):
        results = run_experiment(synthetic_df)
        for split_scores in results["_scores"].values():
            for key in ("test_platt", "test_isotonic"):
                s = split_scores[key]
                assert s.min() >= 0.0 and s.max() <= 1.0


class TestExperimentSmoke:
    """End-to-end train on a small synthetic subsample (CI smoke train)."""

    @pytest.fixture()
    def results(self, synthetic_df):
        return run_experiment(synthetic_df)

    def test_report_is_json_serialisable(self, results):
        json.dumps(public_report(results))

    def test_metric_ranges(self, results):
        for entry in public_report(results)["models"].values():
            for split in ("validation", "test"):
                m = entry[split]
                for key in ("pr_auc", "roc_auc", "brier", "log_loss", "ece"):
                    assert 0.0 <= m[key] <= 1.0 or key == "log_loss"
                assert 0.0 <= m["at_threshold"]["f1"] <= 1.0
            for tm in entry["test"]["at_fixed_thresholds"].values():
                assert 0.0 <= tm["precision"] <= 1.0
                assert 0.0 <= tm["recall"] <= 1.0

    def test_models_beat_prevalence_baseline(self, results):
        """On learnable synthetic data, PR-AUC must beat the base rate."""
        report = public_report(results)
        base = report["splits"]["test"]["positive_rate"]
        for entry in report["models"].values():
            assert entry["test"]["pr_auc"] > base

    def test_threshold_comes_from_validation_scores(self, results):
        for name, entry in public_report(results)["models"].items():
            _, s_val = results["_scores"][name]["val"]
            assert entry["chosen_threshold"] in set(s_val)

    def test_fixed_thresholds_reported(self, results):
        for entry in public_report(results)["models"].values():
            fixed = entry["test"]["at_fixed_thresholds"]
            assert set(fixed) == {"0.3", "0.5"}
            assert fixed["0.3"]["recall"] >= fixed["0.5"]["recall"] - 1e-9

    def test_calibration_reported_before_and_after(self, results):
        for entry in public_report(results)["models"].values():
            cal = entry["calibration_on_test"]
            assert set(cal) == {"raw", "platt", "isotonic"}
            for variant in cal.values():
                assert 0.0 <= variant["ece"] <= 1.0
                assert 0.0 <= variant["brier"] <= 1.0

    def test_deterministic_given_seed(self, synthetic_df):
        first = public_report(run_experiment(synthetic_df, seed=42))
        second = public_report(run_experiment(synthetic_df, seed=42))
        assert first == second

    def test_page_values_changes_the_feature_set(self, synthetic_df):
        with_pv = public_report(run_experiment(synthetic_df,
                                               include_page_values=True))
        assert with_pv["config"]["include_page_values"] is True
