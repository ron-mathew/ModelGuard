"""
Data drift detection using Evidently (or a fallback statistical approach).

Design decisions
----------------
- Evidently is used for feature-level drift detection when available.
- If Evidently is not installed, a fallback using scipy KS-test is used.
- Reference data = training set distribution (saved during data preparation).
- Current data = incoming production batch (simulated or real logs).
- Drift threshold = fraction of features that must drift to flag overall drift.
- Results are saved as JSON and (if Evidently) as HTML reports.

Limitations
-----------
- Drift detection on input features does NOT prove that model accuracy dropped.
  Ground-truth labels are required to measure actual performance degradation.
- Evaluating on a fixed historical test set does not reflect true production
  performance under distributional shift.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

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

from modelguard.config import (
    DRIFT_THRESHOLD,
    DRIFT_REPORTS_DIR,
    FEATURE_COLUMNS,
    MIN_MONITORING_SAMPLES,
    NUMERICAL_FEATURES,
    REFERENCE_DATA_FILE,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Evidently detection (primary)
# ---------------------------------------------------------------------------

def _run_evidently_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    report_path: Optional[Path] = None,
) -> dict:
    """Run Evidently DataDriftPreset and return structured results."""
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
    except (ImportError, ModuleNotFoundError):
        from evidently.legacy.report import Report
        from evidently.legacy.metric_preset import DataDriftPreset

    feature_cols = [c for c in FEATURE_COLUMNS if c in reference.columns]
    ref_feat = reference[feature_cols]
    cur_feat = current[feature_cols]

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_feat, current_data=cur_feat)

    result_dict = report.as_dict()

    # Save HTML report
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(report_path))
        logger.info("Drift HTML report saved to %s", report_path)

    # Parse results
    metrics_list = result_dict.get("metrics", [])
    dataset_drift_metric = next(
        (m for m in metrics_list if m.get("metric") == "DatasetDriftMetric"),
        None,
    )

    drifted_features: list[str] = []
    feature_drift: dict[str, float] = {}

    # Evidently 0.7 groups per-column results under DataDriftTable, while
    # earlier versions emitted individual ColumnDriftMetric entries.
    drift_table = next(
        (m.get("result", {}) for m in metrics_list
         if m.get("metric") == "DataDriftTable"),
        {},
    )
    columns = drift_table.get("drift_by_columns", {})
    if columns:
        for col, details in columns.items():
            score = details.get("drift_score")
            feature_drift[col] = float(score) if score is not None else 0.0
            if details.get("drift_detected", False):
                drifted_features.append(col)
    else:
        for m in metrics_list:
            if m.get("metric") == "ColumnDriftMetric":
                details = m.get("result", {})
                col = details.get("column_name", "")
                score = details.get("drift_score")
                feature_drift[col] = float(score) if score is not None else 0.0
                if details.get("drift_detected", False):
                    drifted_features.append(col)

    dataset_drift = False
    if dataset_drift_metric:
        dataset_drift = dataset_drift_metric.get("result", {}).get(
            "dataset_drift", False
        )

    return {
        "engine": "evidently",
        "dataset_drift": dataset_drift,
        "drifted_features": drifted_features,
        "feature_drift_scores": feature_drift,
        "n_drifted": len(drifted_features),
        "n_features": len(feature_cols),
        "drift_fraction": len(drifted_features) / max(len(feature_cols), 1),
    }


# ---------------------------------------------------------------------------
# Statistical fallback (KS test)
# ---------------------------------------------------------------------------

def _run_ks_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    p_threshold: float = 0.05,
) -> dict:
    """Kolmogorov-Smirnov drift detection on numerical features."""
    from scipy import stats

    drifted: list[str] = []
    scores: dict[str, float] = {}

    numerical_cols = [c for c in NUMERICAL_FEATURES if c in reference.columns and c in current.columns]
    for col in numerical_cols:
        stat, pval = stats.ks_2samp(
            reference[col].dropna().values,
            current[col].dropna().values,
        )
        scores[col] = float(pval)
        if pval < p_threshold:
            drifted.append(col)

    n_features = len(numerical_cols)
    drift_fraction = len(drifted) / max(n_features, 1)
    dataset_drift = drift_fraction >= DRIFT_THRESHOLD

    return {
        "engine": "ks_test",
        "dataset_drift": dataset_drift,
        "drifted_features": drifted,
        "feature_drift_scores": scores,
        "n_drifted": len(drifted),
        "n_features": n_features,
        "drift_fraction": drift_fraction,
    }


# ---------------------------------------------------------------------------
# Simple pure-python fallback (no scipy)
# ---------------------------------------------------------------------------

def _run_simple_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> dict:
    """Very simple drift detection using mean shift for numerical features."""
    drifted: list[str] = []
    scores: dict[str, float] = {}

    numerical_cols = [c for c in NUMERICAL_FEATURES if c in reference.columns and c in current.columns]
    for col in numerical_cols:
        ref_mean = reference[col].mean()
        cur_mean = current[col].mean()
        ref_std = reference[col].std()
        if ref_std > 0:
            shift = abs(cur_mean - ref_mean) / ref_std
        else:
            shift = abs(cur_mean - ref_mean)
        scores[col] = float(shift)
        if shift > 2.0:  # more than 2 standard deviations shift
            drifted.append(col)

    n_features = len(numerical_cols)
    drift_fraction = len(drifted) / max(n_features, 1)
    dataset_drift = drift_fraction >= DRIFT_THRESHOLD

    return {
        "engine": "simple_mean_shift",
        "dataset_drift": dataset_drift,
        "drifted_features": drifted,
        "feature_drift_scores": scores,
        "n_drifted": len(drifted),
        "n_features": n_features,
        "drift_fraction": drift_fraction,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_drift(
    current_data: pd.DataFrame,
    reference_data: Optional[pd.DataFrame] = None,
    report_name: str = "drift_report",
) -> dict:
    """
    Compare current_data against reference_data for feature drift.

    Parameters
    ----------
    current_data    : incoming production batch (feature columns only or with target)
    reference_data  : reference dataset; loads from REFERENCE_DATA_FILE if None
    report_name     : base name for saved report files

    Returns
    -------
    dict with keys: engine, dataset_drift, drifted_features,
                    feature_drift_scores, n_drifted, n_features,
                    drift_fraction, timestamp, n_current_samples
    """
    # Load reference if not provided
    if reference_data is None:
        if not REFERENCE_DATA_FILE.exists():
            raise FileNotFoundError(
                f"Reference data not found at {REFERENCE_DATA_FILE}. "
                "Run training first to generate reference data."
            )
        reference_data = pd.read_csv(REFERENCE_DATA_FILE)
        logger.info("Reference data loaded: %d rows.", len(reference_data))

    # Validate minimum samples
    if len(current_data) < MIN_MONITORING_SAMPLES:
        logger.warning(
            "Current batch has %d samples < minimum %d. "
            "Drift results may be unreliable.",
            len(current_data), MIN_MONITORING_SAMPLES,
        )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_html = DRIFT_REPORTS_DIR / f"{report_name}_{ts}.html"
    DRIFT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Try detection engines in priority order
    drift_result: dict = {}
    try:
        import evidently  # noqa: F401
        drift_result = _run_evidently_drift(
            reference_data, current_data, report_path=report_html
        )
    except ImportError:
        logger.warning(
            "evidently not installed; trying scipy KS test fallback."
        )
        try:
            from scipy import stats  # noqa: F401
            drift_result = _run_ks_drift(reference_data, current_data)
        except ImportError:
            logger.warning(
                "scipy not installed; using simple mean-shift fallback."
            )
            drift_result = _run_simple_drift(reference_data, current_data)

    # Augment result
    drift_result["timestamp"] = datetime.now(timezone.utc).isoformat()
    drift_result["n_current_samples"] = int(len(current_data))
    drift_result["n_reference_samples"] = int(len(reference_data))
    drift_result["threshold_used"] = DRIFT_THRESHOLD

    # Save JSON result
    json_path = DRIFT_REPORTS_DIR / f"{report_name}_{ts}.json"
    with open(json_path, "w") as fh:
        json.dump(drift_result, fh, indent=2, default=str)
    logger.info("Drift JSON result saved to %s", json_path)
    drift_result["report_json"] = str(json_path)
    if report_html.exists():
        drift_result["report_html"] = str(report_html)

    logger.info(
        "Drift detection complete. dataset_drift=%s  drifted=%d/%d  fraction=%.2f",
        drift_result["dataset_drift"],
        drift_result["n_drifted"],
        drift_result["n_features"],
        drift_result["drift_fraction"],
    )
    return drift_result
