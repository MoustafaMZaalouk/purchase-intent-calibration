"""Dataset access, schema validation and temporal splitting.

Dataset: Online Shoppers Purchasing Intention, UCI Machine Learning
Repository, dataset ID 468. Licensed CC BY 4.0 (verified on the dataset
page, https://archive.ics.uci.edu/dataset/468/). See README.md for the
full attribution and citation.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00468/online_shoppers_intention.csv"
)
DATASET_PAGE_URL = (
    "https://archive.ics.uci.edu/dataset/468/"
    "online+shoppers+purchasing+intention+dataset"
)

TARGET = "Revenue"

# Numeric behaviour features. PageValues is deliberately NOT in this
# list: it is excluded by default because it is borderline leakage for
# early-session scoring - it is an average over pages visited before a
# completed transaction, so its full value is only known once the
# session is over, and the dataset documentation flags it as strongly
# associated with the label. The comparison run with it included is
# produced by scripts/run_experiment.py --include-page-values.
NUMERIC_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "SpecialDay",
]

# Categorical features. OperatingSystems, Browser, Region and TrafficType
# are integer-coded categories in the CSV, not ordinal quantities, so
# they are treated as categorical and one-hot encoded.
CATEGORICAL_FEATURES = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

PAGE_VALUES = "PageValues"

ALL_COLUMNS = NUMERIC_FEATURES + [PAGE_VALUES] + CATEGORICAL_FEATURES + [TARGET]

# Chronological order of the months present in the dataset (Jan and Apr
# do not occur). The split must respect this order: predicting the
# future from the past is the honest version of this task.
MONTH_ORDER = ["Feb", "Mar", "May", "June", "Jul",
               "Aug", "Sep", "Oct", "Nov", "Dec"]

DEFAULT_VAL_MONTHS = ("Aug", "Sep")
DEFAULT_TEST_MONTHS = ("Oct", "Nov", "Dec")

MONTHS = set(MONTH_ORDER) | {"Jan", "Apr"}

VISITOR_TYPES = {"Returning_Visitor", "New_Visitor", "Other"}

# Facts about the published file, used by the integrity test.
EXPECTED_ROWS = 12330
EXPECTED_POSITIVES = 1908
EXPECTED_SHA256 = (
    "b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9"
)

DEFAULT_SEED = 42


def sha256_file(path: str | Path) -> str:
    """Return the hex SHA-256 of a file, reading it in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the raw dataset schema and value domains.

    Raises ValueError listing every problem found, so a broken file
    fails loudly instead of producing silent garbage metrics downstream.
    """
    problems = []

    missing = [c for c in ALL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Invalid dataset: missing columns: {missing}")

    extra = [c for c in df.columns if c not in ALL_COLUMNS]
    if extra:
        problems.append(f"unexpected columns: {extra}")

    if df[TARGET].dtype != bool:
        problems.append(f"{TARGET} must be bool, got {df[TARGET].dtype}")
    elif not set(df[TARGET].unique()) <= {True, False}:
        problems.append(f"{TARGET} must be True/False only")

    for col in NUMERIC_FEATURES + [PAGE_VALUES]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            problems.append(f"{col} must be numeric, got {df[col].dtype}")

    if df[NUMERIC_FEATURES + [PAGE_VALUES]].isna().any().any():
        problems.append("numeric features contain missing values")

    if (df[NUMERIC_FEATURES + [PAGE_VALUES]] < 0).any().any():
        problems.append("numeric features contain negative values")

    for col in ("BounceRates", "ExitRates"):
        if ((df[col] < 0) | (df[col] > 1)).any():
            problems.append(f"{col} must lie in [0, 1]")

    if not set(df["Month"].unique()) <= MONTHS:
        problems.append(f"unknown Month values: {sorted(set(df['Month']) - MONTHS)}")

    if not set(df["VisitorType"].unique()) <= VISITOR_TYPES:
        problems.append(
            "unknown VisitorType values: "
            f"{sorted(set(df['VisitorType']) - VISITOR_TYPES)}"
        )

    if df["Weekend"].dtype != bool:
        problems.append(f"Weekend must be bool, got {df['Weekend'].dtype}")

    if problems:
        raise ValueError("Invalid dataset: " + "; ".join(problems))

    return df


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate the CSV downloaded from the UCI repository."""
    df = pd.read_csv(path)
    return validate_frame(df)


def split_by_month(
    df: pd.DataFrame,
    val_months: tuple = DEFAULT_VAL_MONTHS,
    test_months: tuple = DEFAULT_TEST_MONTHS,
) -> dict[str, pd.DataFrame]:
    """Temporal train/validation/test split on the Month column.

    Every session in val_months is chronologically after every training
    session, and every test session after every validation session.
    Random row splits leak seasonal patterns (e.g. November holiday
    shopping) into training; this split does not.

    The returned frames keep their original CSV row index, which acts as
    the session ID: the three index sets are disjoint and together cover
    every row exactly once.
    """
    order = {m: i for i, m in enumerate(MONTH_ORDER)}
    unknown = (set(val_months) | set(test_months)) - set(order)
    if unknown:
        raise ValueError(f"months not present in MONTH_ORDER: {sorted(unknown)}")
    overlap = set(val_months) & set(test_months)
    if overlap:
        raise ValueError(f"val and test months overlap: {sorted(overlap)}")
    if not val_months or not test_months:
        raise ValueError("val_months and test_months must be non-empty")
    if max(order[m] for m in val_months) >= min(order[m] for m in test_months):
        raise ValueError("every val month must come before every test month")

    is_val = df["Month"].isin(val_months)
    is_test = df["Month"].isin(test_months)
    splits = {
        "train": df[~is_val & ~is_test],
        "val": df[is_val],
        "test": df[is_test],
    }
    for name, frame in splits.items():
        if frame.empty:
            raise ValueError(f"{name} split is empty")
        if frame[TARGET].sum() == 0:
            raise ValueError(f"{name} split has no positive sessions")
    return splits
