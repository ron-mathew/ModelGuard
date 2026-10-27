"""
ModelGuard — Centralized configuration.

All tunable parameters, paths, and constants live here.
Do not hardcode these values anywhere else in the codebase.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]  # ModelGuard/
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
REFERENCE_DATA_DIR = DATA_DIR / "reference"
SIMULATED_DATA_DIR = DATA_DIR / "simulated"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
REPORTS_DIR = ROOT_DIR / "reports"
DRIFT_REPORTS_DIR = REPORTS_DIR / "drift"
MONITORING_REPORTS_DIR = REPORTS_DIR / "monitoring"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
RAW_DATA_FILE = RAW_DATA_DIR / "bank-full.csv"
DATA_SEPARATOR = ";"
TARGET_COLUMN = "y"

# Feature to EXCLUDE (call duration — known only post-call, causes leakage).
# Reference: Moro et al. (2014) — "A Data-Driven Approach to Predict the
# Success of Bank Telemarketing".
EXCLUDE_FEATURES = ["duration"]

CATEGORICAL_FEATURES = [
    "job", "marital", "education", "default",
    "housing", "loan", "contact", "month", "poutcome",
]
NUMERICAL_FEATURES = [
    "age", "balance", "day", "campaign", "pdays", "previous",
]

# All features used for modelling (after excluding EXCLUDE_FEATURES)
FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# ---------------------------------------------------------------------------
# Data splits
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
NUMERICAL_IMPUTE_STRATEGY = "median"
CATEGORICAL_IMPUTE_STRATEGY = "most_frequent"
HANDLE_UNKNOWN_CATEGORIES = "ignore"  # OHE unknown categories at inference

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
LOGISTIC_REGRESSION_PARAMS: dict = {
    "C": 1.0,
    "max_iter": 1000,
    "solver": "lbfgs",
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
}

RANDOM_FOREST_PARAMS: dict = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_leaf": 4,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": 1,
}

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{ROOT_DIR / 'mlruns' / 'mlflow.db'}",
)
MLFLOW_ARTIFACT_ROOT = os.environ.get(
    "MLFLOW_ARTIFACT_ROOT",
    str(ROOT_DIR / "mlruns" / "artifacts"),
)
MLFLOW_EXPERIMENT_NAME = "modelguard-bank-marketing"
MLFLOW_MODEL_NAME = "bank-marketing-classifier"
# Explicit serving dependencies avoid MLflow's expensive environment scan and
# keep the logged model environment aligned with requirements.txt.
MLFLOW_MODEL_PIP_REQUIREMENTS = [
    "mlflow==3.16.0",
    "numpy==2.2.6",
    "pandas==2.3.3",
    "scikit-learn==1.7.2",
    "skops==0.14.0",
]

# Alias used to identify the active production model.
CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", 5000))
MODEL_REFRESH_INTERVAL_SECONDS = float(
    os.environ.get("MODEL_REFRESH_INTERVAL_SECONDS", 5.0)
)
PREDICTION_LOG_FILE = ARTIFACTS_DIR / "predictions.jsonl"

# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------
REFERENCE_DATA_FILE = REFERENCE_DATA_DIR / "reference.csv"
DRIFT_THRESHOLD = float(os.environ.get("DRIFT_THRESHOLD", 0.5))
# Fraction of features that must drift to flag overall drift
MIN_MONITORING_SAMPLES = 500  # minimum rows for a valid monitoring batch

PROCESSED_BATCHES_FILE = ARTIFACTS_DIR / "processed_batches.json"

# ---------------------------------------------------------------------------
# Retraining
# ---------------------------------------------------------------------------
# Challenger must beat champion by this margin on validation ROC-AUC
CHALLENGER_IMPROVEMENT_THRESHOLD = float(
    os.environ.get("CHALLENGER_IMPROVEMENT_THRESHOLD", 0.0)
)
# Minimum acceptable challenger ROC-AUC regardless of champion score
CHALLENGER_MIN_ROC_AUC = float(os.environ.get("CHALLENGER_MIN_ROC_AUC", 0.65))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
