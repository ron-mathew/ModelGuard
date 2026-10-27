"""Tests for preprocessing pipeline and training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modelguard.config import FEATURE_COLUMNS, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from modelguard.preprocessing import build_full_pipeline, build_preprocessor
from modelguard.evaluate import compare_models


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_dataset():
    """Tiny but valid dataset for pipeline tests."""
    n = 200
    rng = np.random.default_rng(42)
    X = pd.DataFrame({
        "age": rng.integers(18, 75, size=n),
        "job": rng.choice(["admin.", "blue-collar", "management", "retired"], size=n),
        "marital": rng.choice(["married", "single", "divorced"], size=n),
        "education": rng.choice(["primary", "secondary", "tertiary"], size=n),
        "default": rng.choice(["yes", "no"], size=n),
        "balance": rng.integers(-500, 10000, size=n),
        "housing": rng.choice(["yes", "no"], size=n),
        "loan": rng.choice(["yes", "no"], size=n),
        "contact": rng.choice(["cellular", "telephone", "unknown"], size=n),
        "day": rng.integers(1, 31, size=n),
        "month": rng.choice(["jan", "feb", "mar", "may"], size=n),
        "campaign": rng.integers(1, 10, size=n),
        "pdays": rng.integers(-1, 400, size=n),
        "previous": rng.integers(0, 10, size=n),
        "poutcome": rng.choice(["unknown", "success", "failure"], size=n),
    })
    y = pd.Series(rng.choice([0, 1], p=[0.88, 0.12], size=n))
    return X, y


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

class TestPreprocessor:
    def test_builds_without_error(self):
        pp = build_preprocessor()
        assert pp is not None

    def test_fit_transform_shape(self, tiny_dataset):
        X, y = tiny_dataset
        pp = build_preprocessor()
        X_transformed = pp.fit_transform(X)
        # Output should have more columns than input due to OHE
        assert X_transformed.shape[0] == len(X)
        assert X_transformed.shape[1] > len(NUMERICAL_FEATURES)

    def test_no_target_leakage(self, tiny_dataset):
        """Preprocessor fitted on train should work on val with different distribution."""
        X, y = tiny_dataset
        X_train, X_val = X.iloc[:150], X.iloc[150:]
        pp = build_preprocessor()
        pp.fit(X_train)
        # Should not raise for val set (handle_unknown='ignore')
        pp.transform(X_val)


# ---------------------------------------------------------------------------
# Full pipeline (preprocessor + classifier)
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_logistic_regression_pipeline_fits(self, tiny_dataset):
        from sklearn.linear_model import LogisticRegression
        from modelguard.config import LOGISTIC_REGRESSION_PARAMS
        X, y = tiny_dataset
        pipeline = build_full_pipeline(LogisticRegression(**LOGISTIC_REGRESSION_PARAMS))
        pipeline.fit(X, y)
        preds = pipeline.predict(X)
        assert len(preds) == len(y)

    def test_random_forest_pipeline_fits(self, tiny_dataset):
        from sklearn.ensemble import RandomForestClassifier
        from modelguard.config import RANDOM_FOREST_PARAMS
        X, y = tiny_dataset
        params = {k: v for k, v in RANDOM_FOREST_PARAMS.items() if k != "n_jobs"}
        params["n_estimators"] = 10  # keep fast for tests
        pipeline = build_full_pipeline(RandomForestClassifier(**params))
        pipeline.fit(X, y)
        probas = pipeline.predict_proba(X)
        assert probas.shape == (len(y), 2)

    def test_predict_proba_sums_to_one(self, tiny_dataset):
        from sklearn.linear_model import LogisticRegression
        from modelguard.config import LOGISTIC_REGRESSION_PARAMS
        X, y = tiny_dataset
        pipeline = build_full_pipeline(LogisticRegression(**LOGISTIC_REGRESSION_PARAMS))
        pipeline.fit(X, y)
        probas = pipeline.predict_proba(X)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)

    def test_unseen_categories_handled(self, tiny_dataset):
        """Pipeline trained without 'admin.' job should handle it at inference."""
        from sklearn.linear_model import LogisticRegression
        from modelguard.config import LOGISTIC_REGRESSION_PARAMS
        X, y = tiny_dataset
        X_train = X[X["job"] != "admin."].copy()
        y_train = y[X_train.index]
        pipeline = build_full_pipeline(LogisticRegression(**LOGISTIC_REGRESSION_PARAMS))
        pipeline.fit(X_train, y_train)
        X_test = X[X["job"] == "admin."].head(5).copy()
        # Should not raise
        probas = pipeline.predict_proba(X_test)
        assert probas.shape[1] == 2


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

class TestModelComparison:
    def _metrics(self, roc_auc: float) -> dict:
        return {"roc_auc": roc_auc, "f1_macro": 0.7, "pr_auc": 0.5}

    def test_challenger_accepted_when_better(self):
        accept, reason = compare_models(
            champion_metrics=self._metrics(0.75),
            challenger_metrics=self._metrics(0.78),
            improvement_threshold=0.0,
            min_roc_auc=0.65,
        )
        assert accept is True
        assert "accepted" in reason.lower()

    def test_challenger_rejected_when_worse(self):
        accept, reason = compare_models(
            champion_metrics=self._metrics(0.80),
            challenger_metrics=self._metrics(0.75),
            improvement_threshold=0.0,
            min_roc_auc=0.65,
        )
        assert accept is False

    def test_challenger_rejected_below_min_roc_auc(self):
        accept, reason = compare_models(
            champion_metrics=self._metrics(0.70),
            challenger_metrics=self._metrics(0.60),
            improvement_threshold=0.0,
            min_roc_auc=0.65,
        )
        assert accept is False
        assert "minimum" in reason.lower()

    def test_challenger_rejected_when_not_enough_improvement(self):
        accept, reason = compare_models(
            champion_metrics=self._metrics(0.75),
            challenger_metrics=self._metrics(0.76),
            improvement_threshold=0.02,
            min_roc_auc=0.65,
        )
        assert accept is False

    def test_champion_preserved_when_challenger_rejected(self):
        """Rejection must not modify the champion — tested via return value."""
        accept, _ = compare_models(
            champion_metrics=self._metrics(0.80),
            challenger_metrics=self._metrics(0.70),
        )
        assert accept is False
