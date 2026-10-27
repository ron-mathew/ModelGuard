"""
Monitoring pipeline: batch-level drift analysis and result aggregation.

Responsibilities
----------------
- Load or receive a monitoring batch.
- Check for duplicate batch processing.
- Run drift detection.
- Save structured monitoring events.
- Determine whether retraining should be triggered.

This module is intentionally kept separate from the retraining engine so
each concern can be tested and replaced independently.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from modelguard.config import (
    DRIFT_THRESHOLD,
    FEATURE_COLUMNS,
    MIN_MONITORING_SAMPLES,
    MONITORING_REPORTS_DIR,
    PROCESSED_BATCHES_FILE,
    TARGET_COLUMN,
)
from modelguard.drift import detect_drift

logger = logging.getLogger(__name__)

MONITORING_LOG_FILE = MONITORING_REPORTS_DIR / "monitoring_log.jsonl"


# ---------------------------------------------------------------------------
# Batch deduplication
# ---------------------------------------------------------------------------

def _compute_batch_hash(df: pd.DataFrame) -> str:
    raw_bytes = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(raw_bytes).hexdigest()[:16]


def _load_processed_batches() -> set[str]:
    PROCESSED_BATCHES_FILE.parent.mkdir(parents=True, exist_ok=True)
    if PROCESSED_BATCHES_FILE.exists():
        with open(PROCESSED_BATCHES_FILE) as fh:
            return set(json.load(fh))
    return set()


def _record_processed_batch(batch_or_hash: str | pd.DataFrame) -> None:
    if isinstance(batch_or_hash, pd.DataFrame):
        batch_hash = _compute_batch_hash(batch_or_hash)
    else:
        batch_hash = str(batch_or_hash)
    processed = _load_processed_batches()
    processed.add(batch_hash)
    with open(PROCESSED_BATCHES_FILE, "w") as fh:
        json.dump(list(processed), fh, indent=2)


def is_batch_already_processed(df: pd.DataFrame) -> bool:
    batch_hash = _compute_batch_hash(df)
    return batch_hash in _load_processed_batches()


# ---------------------------------------------------------------------------
# Monitoring event persistence
# ---------------------------------------------------------------------------

def _append_monitoring_event(event: dict) -> None:
    MONITORING_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MONITORING_LOG_FILE, "a") as fh:
        fh.write(json.dumps(event, default=str) + "\n")


def load_monitoring_history() -> list[dict]:
    """Return all recorded monitoring events."""
    if not MONITORING_LOG_FILE.exists():
        return []
    events = []
    with open(MONITORING_LOG_FILE) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return events


# ---------------------------------------------------------------------------
# Main monitoring function
# ---------------------------------------------------------------------------

def run_monitoring(
    current_batch: pd.DataFrame,
    reference_data: Optional[pd.DataFrame] = None,
    batch_label: str = "batch",
    force: bool = False,
) -> dict:
    """
    Run drift monitoring on a batch of production-like data.

    Parameters
    ----------
    current_batch : DataFrame with feature columns (target optional)
    reference_data: reference dataset; loaded from disk if None
    batch_label   : human-readable label for logging
    force         : process the batch even if already recorded

    Returns
    -------
    dict with drift results and a 'should_retrain' flag
    """
    batch_hash = _compute_batch_hash(current_batch)
    logger.info(
        "Starting monitoring run. batch=%s  hash=%s  rows=%d",
        batch_label, batch_hash, len(current_batch),
    )

    # Deduplication check
    if not force and is_batch_already_processed(current_batch):
        msg = f"Batch '{batch_label}' (hash={batch_hash}) already processed. Skipping."
        logger.info(msg)
        return {
            "status": "skipped",
            "reason": "duplicate_batch",
            "batch_hash": batch_hash,
        }

    # Minimum samples check
    if len(current_batch) < MIN_MONITORING_SAMPLES:
        msg = (
            f"Batch has only {len(current_batch)} rows "
            f"(minimum: {MIN_MONITORING_SAMPLES}). Monitoring skipped."
        )
        logger.warning(msg)
        return {
            "status": "skipped",
            "reason": "insufficient_samples",
            "batch_hash": batch_hash,
            "n_rows": len(current_batch),
        }

    # Run drift detection
    drift_result = detect_drift(
        current_data=current_batch,
        reference_data=reference_data,
        report_name=f"monitoring_{batch_label}",
    )

    # Determine whether to trigger retraining
    should_retrain = bool(drift_result.get("dataset_drift", False))
    trigger_reason = (
        f"Dataset drift detected — "
        f"{drift_result['n_drifted']}/{drift_result['n_features']} features drifted "
        f"({drift_result['drift_fraction']:.1%} ≥ threshold {DRIFT_THRESHOLD:.1%})."
        if should_retrain else
        f"No dataset drift — "
        f"{drift_result['n_drifted']}/{drift_result['n_features']} features drifted "
        f"({drift_result['drift_fraction']:.1%} < threshold {DRIFT_THRESHOLD:.1%})."
    )

    # Build monitoring event
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "batch_label": batch_label,
        "batch_hash": batch_hash,
        "n_rows": len(current_batch),
        "drift_engine": drift_result.get("engine"),
        "dataset_drift": drift_result.get("dataset_drift"),
        "drifted_features": drift_result.get("drifted_features", []),
        "drift_fraction": drift_result.get("drift_fraction"),
        "should_retrain": should_retrain,
        "trigger_reason": trigger_reason,
        "report_json": drift_result.get("report_json"),
        "report_html": drift_result.get("report_html"),
    }

    _append_monitoring_event(event)
    _record_processed_batch(current_batch)

    logger.info(
        "Monitoring done. dataset_drift=%s  should_retrain=%s",
        drift_result.get("dataset_drift"), should_retrain,
    )
    return {**drift_result, "should_retrain": should_retrain, "batch_hash": batch_hash}
