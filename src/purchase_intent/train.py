"""Model construction and the train/evaluate experiment flow."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .calibrate import fit_calibrators
from .data import DEFAULT_SEED, split_by_month
from .features import build_preprocessor, select_xy
from .metrics import evaluate_scores, select_threshold_f1, threshold_metrics

# Both models are trained FROM SCRATCH on this dataset. No pretrained
# weights, embeddings or frozen features are used anywhere.
MODEL_NAMES = ("logistic_regression", "gradient_boosting")

# Decision thresholds always reported on the test split, alongside the
# validation-F1 threshold: the symmetric default (0.5) and a lower bar
# (0.3) for the asymmetric-cost case where missing a likely purchaser is
# roughly twice as expensive as a false alarm.
COST_THRESHOLDS = (0.3, 0.5)


def build_models(include_page_values: bool = False,
                 seed: int = DEFAULT_SEED) -> dict:
    """The two compared estimators, each a full preprocessing pipeline.

    logistic_regression: the linear baseline. Standardised numerics,
    one-hot categoricals, L2 regularisation, class weights untouched.
    gradient_boosting: sklearn's histogram gradient boosting (a
    LightGBM-style tree ensemble) on the same encoded features.
    """
    pre = build_preprocessor(include_page_values)
    return {
        "logistic_regression": Pipeline([
            ("prep", clone(pre)),
            ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
        ]),
        "gradient_boosting": Pipeline([
            ("prep", clone(pre)),
            ("clf", HistGradientBoostingClassifier(random_state=seed)),
        ]),
    }


def predict_scores(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Probability of the positive (purchase) class."""
    return model.predict_proba(X)[:, 1]


def run_experiment(
    df: pd.DataFrame,
    include_page_values: bool = False,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Train both models, tune on validation, evaluate once on test.

    Temporal protocol: train on the early months, pick the F1 threshold
    and fit Platt/isotonic calibrators on the validation months, then
    score the frozen configuration on the held-out later months. Returns
    a dict with a JSON-serialisable "report" plus fitted models and
    per-split score arrays under private keys for the plotting script.
    """
    splits = split_by_month(df)
    models = build_models(include_page_values=include_page_values, seed=seed)

    fitted = {}
    for name, model in models.items():
        X_train, y_train = select_xy(splits["train"], include_page_values)
        fitted[name] = model.fit(X_train, y_train)

    results = {
        "config": {
            "seed": seed,
            "include_page_values": include_page_values,
            "split": "temporal by Month: train Feb-Jul, val Aug-Sep, "
                     "test Oct-Dec (chronological, no random rows)",
            "threshold_rule": "maximise F1 on validation, frozen before test; "
                              "test also reported at fixed 0.3 and 0.5",
            "calibration": "Platt and isotonic fitted on validation scores "
                           "only, applied to frozen test scores",
        },
        "splits": {
            name: {
                "n": len(frame),
                "positive_rate": float(frame["Revenue"].mean()),
                "months": sorted(frame["Month"].unique()),
            }
            for name, frame in splits.items()
        },
        "models": {},
    }

    split_scores_all = {}
    for name, model in fitted.items():
        split_scores = {}
        for split_name in ("train", "val", "test"):
            X, y = select_xy(splits[split_name], include_page_values)
            split_scores[split_name] = (y.to_numpy(), predict_scores(model, X))

        y_val, s_val = split_scores["val"]
        y_test, s_test = split_scores["test"]
        threshold, val_f1 = select_threshold_f1(y_val, s_val)
        calibrators = fit_calibrators(y_val, s_val)

        entry = {}
        entry["validation"] = evaluate_scores(y_val, s_val, threshold=threshold)
        entry["validation"]["at_threshold"]["selected_on"] = "validation"

        test = evaluate_scores(y_test, s_test, threshold=threshold)
        test["at_threshold"]["selected_on"] = "validation"
        test["at_fixed_thresholds"] = {
            str(t): threshold_metrics(y_test, s_test, t)
            for t in COST_THRESHOLDS
        }
        entry["test"] = test

        entry["calibration_on_test"] = {"raw": {
            "ece": test["ece"], "brier": test["brier"],
            "log_loss": test["log_loss"],
        }}
        calibrated_scores = {}
        for cal_name, cal in calibrators.items():
            s_cal = cal.predict(s_test)
            calibrated_scores[cal_name] = s_cal
            entry["calibration_on_test"][cal_name] = {
                "ece": float(evaluate_scores(y_test, s_cal)["ece"]),
                "brier": float(evaluate_scores(y_test, s_cal)["brier"]),
                "log_loss": float(evaluate_scores(y_test, s_cal)["log_loss"]),
            }

        entry["chosen_threshold"] = threshold
        entry["chosen_threshold_val_f1"] = val_f1
        results["models"][name] = entry

        split_scores["test_platt"] = calibrated_scores["platt"]
        split_scores["test_isotonic"] = calibrated_scores["isotonic"]
        split_scores["threshold"] = threshold
        split_scores_all[name] = split_scores

    results["_fitted_models"] = fitted
    results["_scores"] = split_scores_all
    return results


def public_report(results: dict) -> dict:
    """The JSON-serialisable part of run_experiment's return value."""
    return {k: v for k, v in results.items() if not k.startswith("_")}
