"""
Data loading, validation, splitting, and manifest generation.

Design decisions
----------------
- `duration` is excluded from all feature sets because it is only known
  after the call ends, which would cause target leakage in production.
- The final test set is never used for model selection or promotion — it is
  reserved for post-decision reporting.
- Stratified splits preserve the ~12% minority class ratio.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from modelguard.config import (
    CATEGORICAL_FEATURES,
    DATA_SEPARATOR,
    EXCLUDE_FEATURES,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    RANDOM_SEED,
    RAW_DATA_FILE,
    TARGET_COLUMN,
    REFERENCE_DATA_DIR,
    REFERENCE_DATA_FILE,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
    ARTIFACTS_DIR,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema specification
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

EXPECTED_DTYPES: dict[str, str] = {
    "age": "int64",
    "balance": "int64",
    "day": "int64",
    "campaign": "int64",
    "pdays": "int64",
    "previous": "int64",
}  # categorical and target remain as object/str

TARGET_CLASSES = {"no", "yes"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_raw_data(path: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """Load the raw bank-marketing CSV and return a DataFrame."""
    logger.info("Loading raw data from %s", path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Please copy bank-full.csv to data/raw/ and retry."
        )
    df = pd.read_csv(path, sep=DATA_SEPARATOR)
    logger.info("Loaded %d rows × %d columns", *df.shape)
    return df


def validate_data(df: pd.DataFrame) -> None:
    """
    Run schema and quality checks.  Raises ValueError on critical failures.

    Checks performed
    ----------------
    1. Required columns present.
    2. Target values encode both binary outcomes using no/yes and/or 0/1.
    3. No duplicated rows (warning only — bank data can legitimately repeat).
    4. Missing-value summary logged.
    5. Excluded features flagged.
    """
    # 1. Required columns
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # 2. Target classes
    actual_classes = set(df[TARGET_COLUMN].dropna().unique())
    target_map = {"yes": 1, "no": 0, 1: 1, 0: 0}
    normalized_classes = {
        target_map[value] for value in actual_classes if value in target_map
    }
    if len(normalized_classes) != 2 or any(
        value not in target_map for value in actual_classes
    ):
        raise ValueError(
            f"Unexpected target classes: {actual_classes}. "
            f"Expected both binary outcomes encoded as {TARGET_CLASSES} "
            "and/or {0, 1}."
        )

    # 3. Duplicates
    n_dup = df.duplicated().sum()
    if n_dup > 0:
        logger.warning("%d duplicate rows found (kept for now).", n_dup)

    # 4. Missing values
    null_counts = df[REQUIRED_COLUMNS].isnull().sum()
    if null_counts.any():
        logger.warning("Null value counts:\n%s", null_counts[null_counts > 0])
    else:
        logger.info("No null values detected.")

    # 5. Excluded features
    present_excluded = set(EXCLUDE_FEATURES) & set(df.columns)
    if present_excluded:
        logger.info(
            "Excluding feature(s) %s from modelling (post-call leakage).",
            present_excluded,
        )

    logger.info("Data validation passed.")


def compute_fingerprint(df: pd.DataFrame) -> str:
    """Return a SHA-256 fingerprint of the DataFrame content."""
    raw_bytes = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(raw_bytes).hexdigest()


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Drop excluded features and return (X, y).

    The target column is normalized to 1/0 from either yes/no or 1/0 input.
    """
    cols_to_drop = [TARGET_COLUMN] + [
        c for c in EXCLUDE_FEATURES if c in df.columns
    ]
    X = df.drop(columns=cols_to_drop)[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].map({"yes": 1, "no": 0, 1: 1, 0: 0}).astype("int64")
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_state: int = RANDOM_SEED,
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.Series, pd.Series, pd.Series,
]:
    """
    Stratified train / val / test split.

    The test set is kept isolated and used only for final champion-vs-challenger
    evaluation — never for model selection.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Ratios must sum to 1."
    )

    # First extract test set
    test_size = test_ratio
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Then split remainder into train and val
    val_fraction = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_fraction,
        stratify=y_temp,
        random_state=random_state,
    )

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(X_train), len(X_val), len(X_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_manifest(
    df: pd.DataFrame,
    splits: dict[str, int],
    output_path: Path | None = None,
) -> dict:
    """Save a JSON dataset manifest and return it."""
    output_path = output_path or (ARTIFACTS_DIR / "data_manifest.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source_file": str(RAW_DATA_FILE),
        "fingerprint": compute_fingerprint(df),
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "target_column": TARGET_COLUMN,
        "excluded_features": EXCLUDE_FEATURES,
        "feature_columns": FEATURE_COLUMNS,
        "target_distribution": df[TARGET_COLUMN].value_counts().to_dict(),
        "split_sizes": splits,
        "random_seed": RANDOM_SEED,
    }
    with open(output_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Data manifest saved to %s", output_path)
    return manifest


def save_reference_data(X_train: pd.DataFrame, y_train: pd.Series) -> Path:
    """
    Save the training set as reference data for drift monitoring.

    The reference dataset includes features + target so Evidently can compute
    both data drift and model performance reports.
    """
    REFERENCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ref = X_train.copy()
    ref[TARGET_COLUMN] = y_train.values
    ref.to_csv(REFERENCE_DATA_FILE, index=False)
    logger.info(
        "Reference data saved (%d rows) → %s", len(ref), REFERENCE_DATA_FILE
    )
    return REFERENCE_DATA_FILE


def validate_inference_schema(data: dict) -> Tuple[bool, str]:
    """
    Validate a single inference request dictionary.

    Returns (is_valid, error_message).
    """
    missing = [c for c in FEATURE_COLUMNS if c not in data]
    if missing:
        return False, f"Missing fields: {missing}"

    extra = [k for k in data if k not in FEATURE_COLUMNS]
    if extra:
        logger.debug("Extra fields in request (ignored): %s", extra)

    # Type coercion checks for numerical features
    for col in NUMERICAL_FEATURES:
        try:
            float(data[col])
        except (ValueError, TypeError):
            return False, f"Field '{col}' must be numeric, got: {data[col]!r}"

    return True, ""
