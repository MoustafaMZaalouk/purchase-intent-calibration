"""Shared fixtures: a small synthetic frame with the real schema."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from purchase_intent.data import (
    ALL_COLUMNS,
    CATEGORICAL_FEATURES,
    MONTH_ORDER,
    NUMERIC_FEATURES,
    PAGE_VALUES,
    TARGET,
)


def make_synthetic_frame(rows_per_month: int = 30, seed: int = 0) -> pd.DataFrame:
    """A schema-valid frame covering every month, with a learnable signal.

    ProductRelated_Duration is made positively associated with the label
    so model smoke tests have something to learn.
    """
    rng = np.random.default_rng(seed)
    blocks = []
    for month in MONTH_ORDER:
        n = rows_per_month
        score = rng.normal(0, 1, n) + rng.normal(0, 1, n)
        revenue = score > np.quantile(score, 0.8)  # ~20% positives
        block = pd.DataFrame({
            "Administrative": rng.integers(0, 5, n),
            "Administrative_Duration": rng.uniform(0, 100, n),
            "Informational": rng.integers(0, 3, n),
            "Informational_Duration": rng.uniform(0, 50, n),
            "ProductRelated": rng.integers(1, 20, n),
            "ProductRelated_Duration": np.where(
                revenue,
                rng.uniform(200, 800, n),
                rng.uniform(0, 300, n),
            ),
            "BounceRates": rng.uniform(0, 0.2, n),
            "ExitRates": rng.uniform(0, 0.2, n),
            PAGE_VALUES: np.where(revenue, rng.uniform(0, 20, n), 0.0),
            "SpecialDay": rng.uniform(0, 1, n).round(2),
            "Month": month,
            "OperatingSystems": rng.integers(1, 4, n),
            "Browser": rng.integers(1, 4, n),
            "Region": rng.integers(1, 4, n),
            "TrafficType": rng.integers(1, 4, n),
            "VisitorType": rng.choice(
                ["Returning_Visitor", "New_Visitor", "Other"], n
            ),
            "Weekend": rng.random(n) < 0.25,
            TARGET: revenue,
        })
        blocks.append(block)
    df = pd.concat(blocks, ignore_index=True)
    return df[ALL_COLUMNS]


@pytest.fixture()
def synthetic_df() -> pd.DataFrame:
    return make_synthetic_frame()


@pytest.fixture()
def numeric_and_categorical():
    return NUMERIC_FEATURES, CATEGORICAL_FEATURES
