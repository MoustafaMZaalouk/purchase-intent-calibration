"""Data validation and temporal split tests.

These test behaviour, not shapes: a broken schema must raise, a valid
frame must pass, and the temporal split must be disjoint, complete and
chronological.
"""
from __future__ import annotations

import pytest

from purchase_intent.data import (
    DEFAULT_TEST_MONTHS,
    DEFAULT_VAL_MONTHS,
    MONTH_ORDER,
    split_by_month,
    validate_frame,
)


class TestValidateFrame:
    def test_synthetic_frame_passes(self, synthetic_df):
        assert validate_frame(synthetic_df) is synthetic_df

    def test_missing_column_rejected(self, synthetic_df):
        broken = synthetic_df.drop(columns=["ExitRates"])
        with pytest.raises(ValueError, match="ExitRates"):
            validate_frame(broken)

    def test_extra_column_rejected(self, synthetic_df):
        broken = synthetic_df.assign(Mystery=1)
        with pytest.raises(ValueError, match="unexpected columns"):
            validate_frame(broken)

    def test_non_bool_label_rejected(self, synthetic_df):
        broken = synthetic_df.copy()
        broken["Revenue"] = broken["Revenue"].astype(int)
        with pytest.raises(ValueError, match="Revenue must be bool"):
            validate_frame(broken)

    def test_negative_duration_rejected(self, synthetic_df):
        broken = synthetic_df.copy()
        broken.loc[broken.index[0], "Administrative_Duration"] = -1.0
        with pytest.raises(ValueError, match="negative"):
            validate_frame(broken)

    def test_bounce_rate_above_one_rejected(self, synthetic_df):
        broken = synthetic_df.copy()
        broken.loc[broken.index[0], "BounceRates"] = 1.5
        with pytest.raises(ValueError, match="BounceRates"):
            validate_frame(broken)

    def test_unknown_month_rejected(self, synthetic_df):
        broken = synthetic_df.copy()
        broken.loc[broken.index[0], "Month"] = "Smarch"
        with pytest.raises(ValueError, match="Month"):
            validate_frame(broken)

    def test_nan_numeric_rejected(self, synthetic_df):
        broken = synthetic_df.copy()
        broken.loc[broken.index[0], "SpecialDay"] = float("nan")
        with pytest.raises(ValueError, match="missing"):
            validate_frame(broken)


class TestSplitByMonth:
    def test_disjoint_session_ids(self, synthetic_df):
        """No session (original row index) appears in two splits."""
        splits = split_by_month(synthetic_df)
        train_idx = set(splits["train"].index)
        val_idx = set(splits["val"].index)
        test_idx = set(splits["test"].index)
        assert train_idx.isdisjoint(val_idx)
        assert train_idx.isdisjoint(test_idx)
        assert val_idx.isdisjoint(test_idx)
        # and together they cover every session exactly once
        assert train_idx | val_idx | test_idx == set(synthetic_df.index)

    def test_chronological_order(self, synthetic_df):
        """Every train month precedes every val month, which precedes test."""
        splits = split_by_month(synthetic_df)
        order = {m: i for i, m in enumerate(MONTH_ORDER)}
        train_max = max(order[m] for m in splits["train"]["Month"].unique())
        val_min = min(order[m] for m in splits["val"]["Month"].unique())
        val_max = max(order[m] for m in splits["val"]["Month"].unique())
        test_min = min(order[m] for m in splits["test"]["Month"].unique())
        assert train_max < val_min
        assert val_max < test_min
        assert set(splits["val"]["Month"].unique()) == set(DEFAULT_VAL_MONTHS)
        assert set(splits["test"]["Month"].unique()) == set(DEFAULT_TEST_MONTHS)

    def test_every_split_has_positives(self, synthetic_df):
        splits = split_by_month(synthetic_df)
        for frame in splits.values():
            assert frame["Revenue"].sum() > 0

    def test_overlapping_months_rejected(self, synthetic_df):
        with pytest.raises(ValueError, match="overlap"):
            split_by_month(synthetic_df, val_months=("Sep", "Oct"),
                           test_months=("Oct", "Nov"))

    def test_non_chronological_rejected(self, synthetic_df):
        with pytest.raises(ValueError, match="before"):
            split_by_month(synthetic_df, val_months=("Nov",),
                           test_months=("Mar", "May"))

    def test_unknown_month_rejected(self, synthetic_df):
        with pytest.raises(ValueError, match="MONTH_ORDER"):
            split_by_month(synthetic_df, val_months=("Smarch",),
                           test_months=("Oct",))


class TestRealDataset:
    """Integrity checks on the actual published CSV, when present."""

    @pytest.fixture()
    def real_df(self):
        from pathlib import Path

        from purchase_intent.data import load_dataset
        path = Path(__file__).resolve().parent.parent / "data" / (
            "online_shoppers_intention.csv"
        )
        if not path.exists():
            pytest.skip("dataset not downloaded; run scripts/download_data.py")
        return load_dataset(path), path

    def test_published_facts(self, real_df):
        from purchase_intent.data import (
            EXPECTED_POSITIVES,
            EXPECTED_ROWS,
            EXPECTED_SHA256,
            sha256_file,
        )
        df, path = real_df
        assert len(df) == EXPECTED_ROWS
        assert int(df["Revenue"].sum()) == EXPECTED_POSITIVES
        assert 0.15 < df["Revenue"].mean() < 0.16
        assert sha256_file(path) == EXPECTED_SHA256

    def test_duplicate_sessions_agree_on_label(self, real_df):
        # The published file contains 125 fully duplicated rows (known
        # quirk, documented in the README). The check that matters for
        # training: no two rows share all features but disagree on the
        # label, which would make the problem inconsistent.
        df, _ = real_df
        features = [c for c in df.columns if c != "Revenue"]
        labels_per_signature = df.groupby(features)["Revenue"].nunique()
        assert (labels_per_signature <= 1).all()

    def test_split_sizes_and_label_drift(self, real_df):
        df, _ = real_df
        splits = split_by_month(df)
        assert len(splits["train"]) > len(splits["val"])
        assert splits["train"]["Revenue"].mean() < splits["test"]["Revenue"].mean()
