"""Tests for the conditional retraining engine."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modelguard.evaluate import compare_models
from modelguard.monitoring import _compute_batch_hash


# ---------------------------------------------------------------------------
# Champion preservation
# ---------------------------------------------------------------------------

class TestChampionPreservation:
    """Verify that a failed challenger never replaces the champion."""

    def test_champion_not_overwritten_on_rejection(self):
        """compare_models returning False must not trigger promotion."""
        champion_metrics = {"roc_auc": 0.82, "f1_macro": 0.74, "pr_auc": 0.55}
        challenger_metrics = {"roc_auc": 0.70, "f1_macro": 0.68, "pr_auc": 0.45}

        accepted, reason = compare_models(
            champion_metrics=champion_metrics,
            challenger_metrics=challenger_metrics,
            improvement_threshold=0.0,
            min_roc_auc=0.65,
        )
        assert not accepted, "Weaker challenger must be rejected."
        # Simulate promotion logic: only promote when accepted is True
        new_champion_would_be_set = accepted  # retraining engine gates on this
        assert not new_champion_would_be_set, (
            "Champion would have been overwritten incorrectly."
        )

    def test_champion_preserved_below_min_threshold(self):
        """Challenger below min ROC-AUC is always rejected."""
        accept, reason = compare_models(
            champion_metrics={"roc_auc": 0.60},
            challenger_metrics={"roc_auc": 0.50},
            improvement_threshold=0.0,
            min_roc_auc=0.65,
        )
        assert not accept


# ---------------------------------------------------------------------------
# Challenger acceptance criteria
# ---------------------------------------------------------------------------

class TestAcceptanceCriteria:
    def test_challenger_accepted_with_clear_improvement(self):
        accept, _ = compare_models(
            champion_metrics={"roc_auc": 0.75},
            challenger_metrics={"roc_auc": 0.80},
        )
        assert accept

    def test_challenger_rejected_equal_score_with_improvement_threshold(self):
        accept, _ = compare_models(
            champion_metrics={"roc_auc": 0.75},
            challenger_metrics={"roc_auc": 0.75},
            improvement_threshold=0.01,
        )
        assert not accept

    def test_challenger_accepted_equal_score_without_improvement_threshold(self):
        accept, _ = compare_models(
            champion_metrics={"roc_auc": 0.75},
            challenger_metrics={"roc_auc": 0.75},
            improvement_threshold=0.0,
        )
        assert accept  # ties go to challenger (equal or better)


# ---------------------------------------------------------------------------
# Batch deduplication in retraining
# ---------------------------------------------------------------------------

class TestRetainBatchDedup:
    def test_same_batch_gives_same_hash(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "a": rng.integers(0, 100, size=50),
            "b": rng.normal(0, 1, size=50),
        })
        h1 = _compute_batch_hash(df)
        h2 = _compute_batch_hash(df.copy())
        assert h1 == h2

    def test_modified_batch_gives_different_hash(self):
        rng = np.random.default_rng(42)
        df1 = pd.DataFrame({"a": rng.integers(0, 100, size=50)})
        df2 = df1.copy()
        df2.iloc[0, 0] = 9999
        assert _compute_batch_hash(df1) != _compute_batch_hash(df2)


# ---------------------------------------------------------------------------
# Retraining engine integration (mocked MLflow / registry)
# ---------------------------------------------------------------------------

class TestRetrainingEngine:
    """
    These tests exercise the retrain.run_retraining flow with mocked
    MLflow calls to avoid needing a live tracking server.
    """

    @pytest.fixture
    def mock_env(self, tmp_path, monkeypatch):
        """Set up a minimal environment for retraining tests."""
        import modelguard.retrain as retrain_module
        import modelguard.monitoring as mon_module

        monkeypatch.setattr(retrain_module, "RETRAIN_LOG_FILE", tmp_path / "retrain.jsonl")
        monkeypatch.setattr(mon_module, "PROCESSED_BATCHES_FILE", tmp_path / "processed.json")

        # Patch registry functions
        monkeypatch.setattr("modelguard.retrain.reg.load_champion", MagicMock(
            return_value=_make_fitted_pipeline()
        ))
        monkeypatch.setattr("modelguard.retrain.reg.get_champion_version", MagicMock(return_value="1"))
        monkeypatch.setattr("modelguard.retrain.reg.register_model", MagicMock(return_value="2"))
        monkeypatch.setattr("modelguard.retrain.reg.set_alias", MagicMock())
        monkeypatch.setattr("modelguard.retrain.reg.set_model_tag", MagicMock())

        # Create fake test set
        test_dir = tmp_path / "test_set"
        test_dir.mkdir()
        rng = np.random.default_rng(42)
        n = 200
        X_test = pd.DataFrame({
            col: rng.integers(0, 50, size=n)
            for col in ["age", "balance", "day", "campaign", "pdays", "previous"]
        })
        for col in ["job", "marital", "education", "default", "housing",
                    "loan", "contact", "month", "poutcome"]:
            X_test[col] = "admin." if col == "job" else "no"
        y_test = pd.Series(rng.choice([0, 1], p=[0.88, 0.12], size=n))
        X_test.to_csv(test_dir / "X_test.csv", index=False)
        y_test.to_csv(test_dir / "y_test.csv", index=False)
        monkeypatch.setattr(retrain_module, "ARTIFACTS_DIR", tmp_path)

        validation_dir = tmp_path / "validation_set"
        validation_dir.mkdir()
        X_test.to_csv(validation_dir / "X_validation.csv", index=False)
        y_test.to_csv(validation_dir / "y_validation.csv", index=False)

        from modelguard.config import REFERENCE_DATA_FILE
        ref_dir = tmp_path / "reference"
        ref_dir.mkdir()
        ref_path = ref_dir / "reference.csv"
        ref_df = X_test.copy()
        ref_df["y"] = y_test.values
        ref_df.to_csv(ref_path, index=False)
        monkeypatch.setattr("modelguard.retrain.REFERENCE_DATA_FILE", ref_path)

        return tmp_path

    def test_retrain_error_when_no_champion(self, tmp_path, monkeypatch):
        import modelguard.retrain as retrain_module
        monkeypatch.setattr(retrain_module, "RETRAIN_LOG_FILE", tmp_path / "r.jsonl")
        monkeypatch.setattr(
            "modelguard.retrain.reg.load_champion",
            MagicMock(side_effect=RuntimeError("No champion"))
        )
        monkeypatch.setattr(
            "modelguard.retrain.reg.get_champion_version",
            MagicMock(return_value=None)
        )
        monkeypatch.setattr(
            "modelguard.retrain.REFERENCE_DATA_FILE",
            tmp_path / "no_ref.csv"  # doesn't exist → error
        )
        result = retrain_module.run_retraining(force=True)
        assert result.get("status") == "error"

    def test_unconfirmed_batch_without_drift_does_not_train(
        self, tmp_path, monkeypatch
    ):
        import modelguard.retrain as retrain_module

        monkeypatch.setattr(retrain_module, "RETRAIN_LOG_FILE", tmp_path / "r.jsonl")
        monkeypatch.setattr(
            retrain_module,
            "RETRAINED_BATCHES_FILE",
            tmp_path / "retrained.json",
        )
        monkeypatch.setattr(
            "modelguard.monitoring.run_monitoring",
            lambda **kwargs: {"should_retrain": False, "batch_hash": "batch-1"},
        )
        load_champion = MagicMock(side_effect=AssertionError("must not train"))
        monkeypatch.setattr("modelguard.retrain.reg.load_champion", load_champion)

        result = retrain_module.run_retraining(
            trigger_batch=pd.DataFrame({"age": [40]}),
            drift_already_confirmed=False,
        )

        assert result["status"] == "skipped"
        assert result["dataset_drift"] is False
        load_champion.assert_not_called()

    @pytest.mark.parametrize(
        ("challenger_auc", "expected_decision", "should_promote"),
        [(0.80, "promoted", True), (0.70, "rejected", False)],
    )
    def test_promotion_gate_is_deterministic_and_preserves_champion(
        self,
        mock_env,
        monkeypatch,
        challenger_auc,
        expected_decision,
        should_promote,
    ):
        import modelguard.retrain as retrain_module

        monkeypatch.setattr(
            retrain_module,
            "RETRAINED_BATCHES_FILE",
            mock_env / "retrained.json",
        )
        monkeypatch.setattr(
            retrain_module,
            "_train_challenger",
            lambda *args, **kwargs: (
                MagicMock(), {"roc_auc": challenger_auc}, "challenger-run"
            ),
        )

        def fake_evaluate(_pipeline, _X, _y, split_name):
            score = 0.75 if split_name == "champion_validation" else 0.75
            return {
                "roc_auc": score,
                "pr_auc": 0.4,
                "f1_macro": 0.6,
                "classification_report": "",
                "confusion_matrix": [],
                "n_samples": 200,
            }

        monkeypatch.setattr(retrain_module, "evaluate_pipeline", fake_evaluate)

        result = retrain_module.run_retraining(
            drift_already_confirmed=True,
            force=True,
        )

        assert result["decision"] == expected_decision
        alias_setter = retrain_module.reg.set_alias
        assert alias_setter.called is should_promote
        if not should_promote:
            assert result["champion_version"] == "1"


def _make_fitted_pipeline():
    """Return a pipeline mock that behaves like a fitted sklearn pipeline."""
    mock = MagicMock()
    rng = np.random.default_rng(0)
    n = 200

    def fake_proba(X):
        n_rows = len(X) if hasattr(X, "__len__") else 200
        p = rng.random(size=n_rows)
        return np.column_stack([1 - p, p])

    mock.predict_proba = fake_proba
    return mock
