"""Feature-policy tests: the PageValues default and encoding behaviour."""
from __future__ import annotations

import numpy as np

from purchase_intent.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from purchase_intent.features import (
    build_preprocessor,
    feature_columns,
    select_xy,
)


class TestPageValuesPolicy:
    def test_excluded_by_default(self):
        numeric, _ = feature_columns()
        assert "PageValues" not in numeric

    def test_included_on_request(self):
        numeric, _ = feature_columns(include_page_values=True)
        assert "PageValues" in numeric

    def test_select_xy_drops_page_values_by_default(self, synthetic_df):
        X, _ = select_xy(synthetic_df)
        assert "PageValues" not in X.columns

    def test_select_xy_keeps_page_values_when_asked(self, synthetic_df):
        X, _ = select_xy(synthetic_df, include_page_values=True)
        assert "PageValues" in X.columns

    def test_target_never_in_features(self, synthetic_df):
        for flag in (False, True):
            X, y = select_xy(synthetic_df, include_page_values=flag)
            assert "Revenue" not in X.columns
            assert set(y.unique()) <= {0, 1}


class TestPreprocessor:
    def test_feature_names_cover_all_inputs(self, synthetic_df):
        pre = build_preprocessor().fit(synthetic_df)
        names = list(pre.get_feature_names_out())
        for col in NUMERIC_FEATURES:
            assert f"num__{col}" in names
        # one-hot expands each categorical into one column per level
        cat_names = [n for n in names if n.startswith("cat__")]
        assert any("Month_May" in n for n in cat_names)
        assert any("VisitorType_New_Visitor" in n for n in cat_names)
        expected = len(NUMERIC_FEATURES) + len(cat_names)
        assert len(names) == expected

    def test_unknown_category_does_not_crash(self, synthetic_df):
        pre = build_preprocessor().fit(synthetic_df)
        row = synthetic_df.head(1).copy()
        row["VisitorType"] = "NeverSeenBefore"
        out = pre.transform(row)
        assert np.isfinite(out).all()

    def test_numeric_standardisation(self, synthetic_df):
        pre = build_preprocessor().fit(synthetic_df)
        out = pre.transform(synthetic_df)
        numeric_block = out[:, : len(NUMERIC_FEATURES)]
        assert np.abs(numeric_block.mean(axis=0)).max() < 1e-10
        assert np.abs(numeric_block.std(axis=0) - 1).max() < 1e-6

    def test_output_is_dense_for_boosting(self, synthetic_df):
        pre = build_preprocessor().fit(synthetic_df)
        out = pre.transform(synthetic_df.head(5))
        assert isinstance(out, np.ndarray)

    def test_categorical_list_unchanged_by_flag(self):
        _, cat_default = feature_columns()
        _, cat_pv = feature_columns(include_page_values=True)
        assert cat_default == cat_pv == CATEGORICAL_FEATURES
