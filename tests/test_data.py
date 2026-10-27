"""Tests for data loading, validation, splitting, and manifest generation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modelguard.data import (
    compute_fingerprint,
    prepare_features,
    save_manifest,
    split_data,
    validate_data,
    validate_inference_schema,
)
from modelguard.config import FEATURE_COLUMNS, TARGET_COLUMN


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_df():
    """A minimal valid DataFrame mimicking bank-full.csv structure."""
    n = 100
    import numpy as np
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "age": rng.integers(18, 75, size=n),
        "job": rng.choice(["admin.", "blue-collar", "management", "retired"], size=n),
        "marital": rng.choice(["married", "single", "divorced"], size=n),
        "education": rng.choice(["primary", "secondary", "tertiary", "unknown"], size=n),
        "default": rng.choice(["yes", "no"], size=n),
        "balance": rng.integers(-500, 10000, size=n),
        "housing": rng.choice(["yes", "no"], size=n),
        "loan": rng.choice(["yes", "no"], size=n),
        "contact": rng.choice(["cellular", "telephone", "unknown"], size=n),
        "day": rng.integers(1, 31, size=n),
        "month": rng.choice(["jan", "feb", "mar", "may"], size=n),
        "duration": rng.integers(0, 600, size=n),  # should be excluded
        "campaign": rng.integers(1, 10, size=n),
        "pdays": rng.integers(-1, 400, size=n),
        "previous": rng.integers(0, 10, size=n),
        "poutcome": rng.choice(["unknown", "success", "failure", "other"], size=n),
        "y": rng.choice(["yes", "no"], p=[0.12, 0.88], size=n),
    })
    return df


# ---------------------------------------------------------------------------
# validate_data
# ---------------------------------------------------------------------------

class TestValidateData:
    def test_valid_df_passes(self, small_df):
        validate_data(small_df)  # should not raise

    def test_missing_column_raises(self, small_df):
        df = small_df.drop(columns=["age"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_data(df)

    def test_bad_target_class_raises(self, small_df):
        df = small_df.copy()
        df["y"] = df["y"].replace("no", "nope")
        with pytest.raises(ValueError, match="Unexpected target classes"):
            validate_data(df)

    def test_binary_integer_target_is_supported(self, small_df):
        df = small_df.copy()
        df["y"] = df["y"].map({"no": 0, "yes": 1})
        validate_data(df)
        _, y = prepare_features(df)
        assert set(y.unique()) == {0, 1}

    def test_mixed_binary_target_encodings_are_supported(self, small_df):
        df = small_df.copy()
        df.loc[df.index[::2], "y"] = df.loc[df.index[::2], "y"].map(
            {"no": 0, "yes": 1}
        )
        validate_data(df)
        _, y = prepare_features(df)
        assert set(y.unique()) == {0, 1}


# ---------------------------------------------------------------------------
# prepare_features
# ---------------------------------------------------------------------------

class TestPrepareFeatures:
    def test_duration_excluded(self, small_df):
        X, y = prepare_features(small_df)
        assert "duration" not in X.columns

    def test_target_not_in_features(self, small_df):
        X, y = prepare_features(small_df)
        assert TARGET_COLUMN not in X.columns

    def test_target_is_binary(self, small_df):
        X, y = prepare_features(small_df)
        assert set(y.unique()).issubset({0, 1})

    def test_feature_columns_match(self, small_df):
        X, y = prepare_features(small_df)
        assert list(X.columns) == FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# split_data
# ---------------------------------------------------------------------------

class TestSplitData:
    def test_sizes_sum_to_total(self, small_df):
        X, y = prepare_features(small_df)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        assert len(X_train) + len(X_val) + len(X_test) == len(X)

    def test_no_overlap(self, small_df):
        X, y = prepare_features(small_df)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        train_idx = set(X_train.index)
        val_idx = set(X_val.index)
        test_idx = set(X_test.index)
        assert train_idx.isdisjoint(val_idx)
        assert train_idx.isdisjoint(test_idx)
        assert val_idx.isdisjoint(test_idx)

    def test_reproducible(self, small_df):
        X, y = prepare_features(small_df)
        split1 = split_data(X, y)
        split2 = split_data(X, y)
        pd.testing.assert_frame_equal(split1[0], split2[0])


# ---------------------------------------------------------------------------
# compute_fingerprint
# ---------------------------------------------------------------------------

class TestFingerprint:
    def test_deterministic(self, small_df):
        assert compute_fingerprint(small_df) == compute_fingerprint(small_df)

    def test_changes_on_modification(self, small_df):
        fp1 = compute_fingerprint(small_df)
        df2 = small_df.copy()
        df2.iloc[0, 0] = 999
        fp2 = compute_fingerprint(df2)
        assert fp1 != fp2


# ---------------------------------------------------------------------------
# validate_inference_schema
# ---------------------------------------------------------------------------

class TestInferenceSchema:
    def _valid_record(self):
        return {
            "age": 35, "job": "technician", "marital": "single",
            "education": "secondary", "default": "no", "balance": 1200,
            "housing": "yes", "loan": "no", "contact": "cellular",
            "day": 15, "month": "may", "campaign": 1,
            "pdays": -1, "previous": 0, "poutcome": "unknown",
        }

    def test_valid_record_passes(self):
        is_valid, msg = validate_inference_schema(self._valid_record())
        assert is_valid, msg

    def test_missing_field_fails(self):
        record = self._valid_record()
        del record["age"]
        is_valid, msg = validate_inference_schema(record)
        assert not is_valid
        assert "age" in msg

    def test_non_numeric_numerical_field_fails(self):
        record = self._valid_record()
        record["age"] = "thirty-five"
        is_valid, msg = validate_inference_schema(record)
        assert not is_valid
        assert "age" in msg
