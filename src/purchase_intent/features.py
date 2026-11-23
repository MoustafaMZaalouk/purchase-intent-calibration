"""Feature preprocessing shared by both models.

PageValues is excluded by default. It is computed from page views and
e-commerce transaction values and is effectively only known once a
session is over; the dataset authors note it is strongly associated
with the label. Keeping it out is the conservative default for a model
meant to score sessions early. Pass include_page_values=True to run the
comparison with it included.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PAGE_VALUES,
    TARGET,
)


def feature_columns(include_page_values: bool = False) -> tuple[list, list]:
    """Return (numeric, categorical) feature lists for the configuration."""
    numeric = list(NUMERIC_FEATURES)
    if include_page_values:
        numeric = numeric + [PAGE_VALUES]
    return numeric, list(CATEGORICAL_FEATURES)


def build_preprocessor(include_page_values: bool = False) -> ColumnTransformer:
    """Standard-scale numerics, one-hot encode categoricals.

    sparse_output=False because HistGradientBoostingClassifier does not
    accept sparse matrices; the one-hot output here is small.
    """
    numeric, categorical = feature_columns(include_page_values)
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ]
    )


def select_xy(df: pd.DataFrame, include_page_values: bool = False):
    """Split a validated frame into (X, y) honouring the PageValues policy."""
    numeric, categorical = feature_columns(include_page_values)
    X = df[numeric + categorical].copy()
    y = df[TARGET].astype(int)
    return X, y
