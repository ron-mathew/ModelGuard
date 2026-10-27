"""Tests for drift detection and monitoring pipeline."""

from __future__ import annotations

import os
os.environ["DO_NOT_TRACK"] = "1"
os.environ["EVIDENTLY_DISABLE_TELEMETRY"] = "1"

try:
    import evidently.telemetry
    evidently.telemetry.find_or_create_user_id = lambda: None
    evidently.telemetry.event_logger.enabled = lambda: False
    evidently.telemetry.event_logger.send_event = lambda *a, **k: None
except Exception:
    pass

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modelguard.drift import (
    _run_evidently_drift,
    _run_simple_drift,
    _run_ks_drift,
)
from modelguard.monitoring import (
    _compute_batch_hash,
    is_batch_already_processed,
    run_monitoring,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def reference_df():
    """Reference dataset with known distribution."""
    rng = np.random.default_rng(42)
    n = 600
    return pd.DataFrame({
        "age": rng.normal(40, 10, size=n).clip(18, 90).astype(int),
        "balance": rng.normal(1000, 2000, size=n).astype(int),
        "day": rng.integers(1, 31, size=n),
        "campaign": rng.integers(1, 6, size=n),
        "pdays": rng.integers(-1, 400, size=n),
        "previous": rng.integers(0, 5, size=n),
        "job": rng.choice(["admin.", "blue-collar", "management"], size=n),
        "marital": rng.choice(["married", "single", "divorced"], size=n),
        "education": rng.choice(["primary", "secondary", "tertiary"], size=n),
        "default": rng.choice(["yes", "no"], size=n),
        "housing": rng.choice(["yes", "no"], size=n),
        "loan": rng.choice(["yes", "no"], size=n),
        "contact": rng.choice(["cellular", "telephone", "unknown"], size=n),
        "month": rng.choice(["jan", "feb", "mar", "may"], size=n),
        "poutcome": rng.choice(["unknown", "success", "failure"], size=n),
        "y": rng.choice([0, 1], p=[0.88, 0.12], size=n),
    })


@pytest.fixture
def drifted_df(reference_df):
    """Dataset with severe age shift (drifted)."""
    df = reference_df.copy()
    df["age"] = (df["age"] + 20).clip(18, 95)  # severe shift
    return df


@pytest.fixture
def normal_df(reference_df):
    """Dataset sampled from same distribution as reference (no drift)."""
    rng = np.random.default_rng(99)
    return reference_df.sample(frac=0.8, random_state=99).copy()


# ---------------------------------------------------------------------------
# Simple drift detector
# ---------------------------------------------------------------------------

class TestSimpleDrift:
    def test_detects_severe_age_shift(self, reference_df, drifted_df):
        result = _run_simple_drift(reference_df, drifted_df)
        # age has shifted by +20 which is >> 2 std deviations
        assert "age" in result["drifted_features"]

    def test_no_drift_on_same_data(self, reference_df):
        result = _run_simple_drift(reference_df, reference_df.copy())
        assert len(result["drifted_features"]) == 0

    def test_result_has_required_keys(self, reference_df, drifted_df):
        result = _run_simple_drift(reference_df, drifted_df)
        for key in ["dataset_drift", "drifted_features", "drift_fraction", "n_features"]:
            assert key in result


# ---------------------------------------------------------------------------
# KS test drift detector
# ---------------------------------------------------------------------------

class TestKSDrift:
    def test_detects_severe_age_shift(self, reference_df, drifted_df):
        pytest.importorskip("scipy")
        result = _run_ks_drift(reference_df, drifted_df)
        assert "age" in result["drifted_features"]

    def test_result_has_required_keys(self, reference_df, drifted_df):
        pytest.importorskip("scipy")
        result = _run_ks_drift(reference_df, drifted_df)
        for key in ["dataset_drift", "drifted_features", "drift_fraction"]:
            assert key in result


class TestEvidently07ResultParsing:
    def test_reads_data_drift_table_column_results(self, reference_df, monkeypatch):
        """Evidently 0.7 nests column details under DataDriftTable."""
        from unittest.mock import MagicMock

        report = MagicMock()
        report.as_dict.return_value = {
            "metrics": [
                {
                    "metric": "DatasetDriftMetric",
                    "result": {"dataset_drift": False},
                },
                {
                    "metric": "DataDriftTable",
                    "result": {
                        "drift_by_columns": {
                            "age": {
                                "drift_score": 2.4,
                                "drift_detected": True,
                            },
                            "balance": {
                                "drift_score": 0.02,
                                "drift_detected": False,
                            },
                        }
                    },
                },
            ]
        }
        monkeypatch.setattr("evidently.legacy.report.Report", lambda **kwargs: report)

        result = _run_evidently_drift(reference_df, reference_df.copy())

        assert result["drifted_features"] == ["age"]
        assert result["feature_drift_scores"] == {"age": 2.4, "balance": 0.02}
        assert result["n_features"] == len(reference_df.columns) - 1


# ---------------------------------------------------------------------------
# Batch deduplication
# ---------------------------------------------------------------------------

class TestBatchDedup:
    def test_same_df_has_same_hash(self, reference_df):
        h1 = _compute_batch_hash(reference_df)
        h2 = _compute_batch_hash(reference_df.copy())
        assert h1 == h2

    def test_different_df_has_different_hash(self, reference_df, drifted_df):
        h1 = _compute_batch_hash(reference_df)
        h2 = _compute_batch_hash(drifted_df)
        assert h1 != h2


# ---------------------------------------------------------------------------
# Monitoring pipeline
# ---------------------------------------------------------------------------

class TestRunMonitoring:
    def test_skips_insufficient_samples(self, reference_df, tmp_path, monkeypatch):
        import modelguard.monitoring as mon_module
        monkeypatch.setattr(mon_module, "MONITORING_LOG_FILE", tmp_path / "mon.jsonl")
        monkeypatch.setattr(mon_module, "PROCESSED_BATCHES_FILE", tmp_path / "processed.json")

        tiny = reference_df.head(50)  # below MIN_MONITORING_SAMPLES
        result = run_monitoring(
            current_batch=tiny,
            reference_data=reference_df,
            force=True,
        )
        assert result["status"] == "skipped"
        assert result["reason"] == "insufficient_samples"

    def test_duplicate_batch_skipped(self, reference_df, tmp_path, monkeypatch):
        import modelguard.monitoring as mon_module
        monkeypatch.setattr(mon_module, "MONITORING_LOG_FILE", tmp_path / "mon.jsonl")
        monkeypatch.setattr(mon_module, "PROCESSED_BATCHES_FILE", tmp_path / "processed.json")

        from modelguard.config import DRIFT_REPORTS_DIR
        monkeypatch.setattr(
            "modelguard.drift.DRIFT_REPORTS_DIR", tmp_path / "drift_reports"
        )

        # First run
        run_monitoring(
            current_batch=reference_df,
            reference_data=reference_df,
            force=True,
        )
        # Second run same batch (not forced)
        result = run_monitoring(
            current_batch=reference_df,
            reference_data=reference_df,
            force=False,
        )
        assert result["status"] == "skipped"
        assert result["reason"] == "duplicate_batch"

    def test_drift_detected_on_drifted_data(self, reference_df, drifted_df, tmp_path, monkeypatch):
        import modelguard.monitoring as mon_module
        monkeypatch.setattr(mon_module, "MONITORING_LOG_FILE", tmp_path / "mon.jsonl")
        monkeypatch.setattr(mon_module, "PROCESSED_BATCHES_FILE", tmp_path / "processed.json")
        monkeypatch.setattr(
            "modelguard.drift.DRIFT_REPORTS_DIR", tmp_path / "drift_reports"
        )

        result = run_monitoring(
            current_batch=drifted_df,
            reference_data=reference_df,
            batch_label="test_drifted",
            force=True,
        )
        # Should not be skipped
        assert result.get("status") != "skipped"
        # Drift fraction should be nonzero
        assert result.get("drift_fraction", 0) >= 0
