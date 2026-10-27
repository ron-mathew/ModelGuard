"""
Conditional retraining engine.

Workflow
--------
1. Validate incoming batch and check for duplicates.
2. Confirm drift threshold has been met.
3. Train a challenger model using the approved pipeline.
4. Compare challenger and champion on the persisted validation set.
5. Promote challenger only if ALL acceptance criteria pass.
6. Retain champion if challenger fails or any error occurs.
7. Record full audit trail.

Safeguards
----------
- Champion is NEVER overwritten before challenger evaluation completes.
- Duplicate batch prevention via batch hashing.
- Challenger is only promoted if it meets BOTH the minimum ROC-AUC and the
  improvement-over-champion threshold.
- All decisions are recorded with reasons in the retraining log.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from modelguard.config import (
    ARTIFACTS_DIR,
    CHALLENGER_ALIAS,
    CHALLENGER_IMPROVEMENT_THRESHOLD,
    CHALLENGER_MIN_ROC_AUC,
    CHAMPION_ALIAS,
    LOGISTIC_REGRESSION_PARAMS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_NAME,
    MLFLOW_MODEL_PIP_REQUIREMENTS,
    MLFLOW_TRACKING_URI,
    MONITORING_REPORTS_DIR,
    RANDOM_FOREST_PARAMS,
    RANDOM_SEED,
    REFERENCE_DATA_FILE,
    TARGET_COLUMN,
)
from modelguard.data import (
    compute_fingerprint,
    prepare_features,
    validate_data,
)
from modelguard.evaluate import compare_models, evaluate_pipeline
from modelguard.monitoring import (
    _compute_batch_hash,
)
from modelguard.preprocessing import build_full_pipeline
from modelguard import registry as reg

logger = logging.getLogger(__name__)

RETRAIN_LOG_FILE = MONITORING_REPORTS_DIR / "retraining_log.jsonl"
RETRAINED_BATCHES_FILE = ARTIFACTS_DIR / "retrained_batches.json"


def _load_retrained_batches() -> set[str]:
    if not RETRAINED_BATCHES_FILE.exists():
        return set()
    with RETRAINED_BATCHES_FILE.open(encoding="utf-8") as fh:
        return set(json.load(fh))


def _record_retrained_batch(batch: pd.DataFrame) -> None:
    hashes = _load_retrained_batches()
    hashes.add(_compute_batch_hash(batch))
    RETRAINED_BATCHES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RETRAINED_BATCHES_FILE.open("w", encoding="utf-8") as fh:
        json.dump(sorted(hashes), fh, indent=2)


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------

def _append_retrain_event(event: dict) -> None:
    RETRAIN_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RETRAIN_LOG_FILE, "a") as fh:
        fh.write(json.dumps(event, default=str) + "\n")


def load_retrain_history() -> list[dict]:
    if not RETRAIN_LOG_FILE.exists():
        return []
    events = []
    with open(RETRAIN_LOG_FILE) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return events


# ---------------------------------------------------------------------------
# Test set loading
# ---------------------------------------------------------------------------

def _load_test_set():
    test_dir = ARTIFACTS_DIR / "test_set"
    X_test = pd.read_csv(test_dir / "X_test.csv")
    y_test = pd.read_csv(test_dir / "y_test.csv").squeeze()
    return X_test, y_test


def _load_validation_set():
    validation_dir = ARTIFACTS_DIR / "validation_set"
    X_validation = pd.read_csv(validation_dir / "X_validation.csv")
    y_validation = pd.read_csv(validation_dir / "y_validation.csv").squeeze()
    return X_validation, y_validation


# ---------------------------------------------------------------------------
# Challenger training
# ---------------------------------------------------------------------------

def _train_challenger(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    data_fingerprint: str,
    model_type: str = "random_forest",
) -> tuple[object, dict, str]:
    """
    Train a challenger pipeline and log it to MLflow.

    Returns (fitted_pipeline, val_metrics, run_id).
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    if model_type == "random_forest":
        estimator = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
        params = {k: v for k, v in RANDOM_FOREST_PARAMS.items() if k != "n_jobs"}
    else:
        estimator = LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)
        params = LOGISTIC_REGRESSION_PARAMS

    pipeline = build_full_pipeline(estimator)

    with mlflow.start_run(run_name=f"challenger_{model_type}") as run:
        run_id = run.info.run_id
        params["model_type"] = model_type
        params["role"] = "challenger"
        params["data_fingerprint"] = data_fingerprint
        mlflow.log_params(params)
        mlflow.set_tags({
            "validation_protocol": "baseline_validation_v1",
            "validation_split": "baseline_validation",
        })

        t0 = time.perf_counter()
        pipeline.fit(X_train, y_train)
        duration = time.perf_counter() - t0

        val_metrics = evaluate_pipeline(pipeline, X_val, y_val, "challenger_validation")
        mlflow.log_metric("val_roc_auc", val_metrics["roc_auc"])
        mlflow.log_metric("val_pr_auc", val_metrics["pr_auc"])
        mlflow.log_metric("val_f1_macro", val_metrics["f1_macro"])
        mlflow.log_metric("train_duration_s", duration)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            input_example=X_val.head(1),
            serialization_format="skops",
            # These specific types are required for our in-process fitted
            # preprocessing and RandomForest objects. Never trust external files.
            skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
            pip_requirements=MLFLOW_MODEL_PIP_REQUIREMENTS,
        )
        logger.info(
            "Challenger trained: run_id=%s  val_roc_auc=%.4f",
            run_id, val_metrics["roc_auc"],
        )

    return pipeline, val_metrics, run_id


# ---------------------------------------------------------------------------
# Main retraining workflow
# ---------------------------------------------------------------------------

def run_retraining(
    trigger_batch: Optional[pd.DataFrame] = None,
    drift_already_confirmed: bool = False,
    model_type: str = "random_forest",
    force: bool = False,
) -> dict:
    """
    Execute the full conditional retraining workflow.

    Parameters
    ----------
    trigger_batch           : the batch that triggered the drift alert
    drift_already_confirmed : skip drift check (use when called after monitoring)
    model_type              : 'random_forest' or 'logistic_regression'
    force                   : bypass duplicate-batch guard

    Returns
    -------
    dict with keys: status, decision, reason, champion_version,
                    challenger_version, champion_metrics, challenger_metrics
    """
    start_ts = datetime.now(timezone.utc).isoformat()
    audit: dict = {
        "timestamp": start_ts,
        "trigger_batch_rows": len(trigger_batch) if trigger_batch is not None else None,
        "model_type": model_type,
        "drift_already_confirmed": drift_already_confirmed,
        "comparison_protocol": "baseline_validation_v1",
    }

    # -----------------------------------------------------------------------
    # 1. Batch deduplication guard
    # -----------------------------------------------------------------------
    if trigger_batch is not None and not force:
        batch_hash = _compute_batch_hash(trigger_batch)
        audit["batch_hash"] = batch_hash
        if batch_hash in _load_retrained_batches():
            reason = (
                f"Batch (hash={batch_hash}) already triggered a retraining run. "
                "Skipping to avoid redundant training."
            )
            logger.info(reason)
            audit.update({"status": "skipped", "reason": reason})
            _append_retrain_event(audit)
            return audit

    # Monitoring and retraining maintain separate batch ledgers. Re-evaluate
    # an unconfirmed batch even if monitoring already recorded it.
    if trigger_batch is not None and not drift_already_confirmed:
        from modelguard.monitoring import run_monitoring

        drift_result = run_monitoring(
            current_batch=trigger_batch,
            batch_label="retraining_trigger",
            force=True,
        )
        audit["monitoring_status"] = drift_result.get("status", "completed")
        audit["monitoring_batch_hash"] = drift_result.get("batch_hash")
        audit["dataset_drift"] = drift_result.get("dataset_drift", False)
        if not drift_result.get("should_retrain", False):
            reason = (
                "Retraining skipped because the trigger batch did not pass "
                "the drift check."
            )
            audit.update({"status": "skipped", "reason": reason})
            _append_retrain_event(audit)
            return audit

    # -----------------------------------------------------------------------
    # 2. Load reference data for retraining
    # -----------------------------------------------------------------------
    logger.info("Loading reference data for retraining.")
    if not REFERENCE_DATA_FILE.exists():
        reason = "Reference data not found. Cannot retrain."
        logger.error(reason)
        audit.update({"status": "error", "reason": reason})
        _append_retrain_event(audit)
        return audit

    ref_df = pd.read_csv(REFERENCE_DATA_FILE)
    # Use reference data as the retraining corpus (with proper re-split)
    # In production you would merge reference + new labeled data.

    # If trigger_batch has labels, merge it to augment training data
    if trigger_batch is not None and TARGET_COLUMN in trigger_batch.columns:
        logger.info(
            "Augmenting reference data with labeled batch (%d rows).",
            len(trigger_batch),
        )
        ref_df = pd.concat([ref_df, trigger_batch], ignore_index=True)

    validate_data(ref_df)
    X_train, y_train = prepare_features(ref_df)
    data_fingerprint = compute_fingerprint(ref_df)
    audit["data_fingerprint"] = data_fingerprint
    audit["n_train"] = len(X_train)

    # -----------------------------------------------------------------------
    # 3. Compare models on the persisted baseline validation set. The final
    # test split is kept out of the promotion decision.
    # -----------------------------------------------------------------------
    logger.info("Loading champion model for comparison.")
    try:
        champion_pipeline, champion_version = reg.load_champion_versioned(
            model_name=MLFLOW_MODEL_NAME
        )
    except RuntimeError as exc:
        reason = f"Cannot load champion model: {exc}"
        logger.error(reason)
        audit.update({"status": "error", "reason": reason})
        _append_retrain_event(audit)
        return audit

    try:
        # Load the final holdout for reporting only; its metrics do not enter
        # the acceptance decision below.
        X_test, y_test = _load_test_set()
    except FileNotFoundError:
        reason = "Held-out test set not found. Run initial training first."
        logger.error(reason)
        audit.update({"status": "error", "reason": reason})
        _append_retrain_event(audit)
        return audit

    try:
        X_validation, y_validation = _load_validation_set()
    except FileNotFoundError:
        reason = (
            "Held-out validation set not found. Rerun baseline training to "
            "create artifacts/validation_set before retraining."
        )
        logger.error(reason)
        audit.update({"status": "error", "reason": reason})
        _append_retrain_event(audit)
        return audit

    champion_val_metrics = evaluate_pipeline(
        champion_pipeline,
        X_validation,
        y_validation,
        "champion_validation",
    )
    audit["champion_version"] = champion_version
    audit["champion_validation_roc_auc"] = champion_val_metrics["roc_auc"]
    logger.info(
        "Champion v%s validation ROC-AUC=%.4f",
        champion_version, champion_val_metrics["roc_auc"],
    )

    # -----------------------------------------------------------------------
    # 4. Train challenger
    # -----------------------------------------------------------------------
    logger.info("Training challenger (%s).", model_type)
    try:
        challenger_pipeline, challenger_val_metrics, challenger_run_id = (
            _train_challenger(
                X_train,
                y_train,
                X_validation,
                y_validation,
                data_fingerprint,
                model_type,
            )
        )
    except Exception as exc:
        reason = f"Challenger training failed: {exc}"
        logger.exception(reason)
        audit.update({"status": "error", "reason": reason})
        _append_retrain_event(audit)
        return audit

    audit["challenger_run_id"] = challenger_run_id
    audit["challenger_validation_roc_auc"] = challenger_val_metrics["roc_auc"]

    # -----------------------------------------------------------------------
    # 5. Champion / challenger comparison
    # -----------------------------------------------------------------------
    accept, comparison_reason = compare_models(
        champion_metrics=champion_val_metrics,
        challenger_metrics=challenger_val_metrics,
        improvement_threshold=CHALLENGER_IMPROVEMENT_THRESHOLD,
        min_roc_auc=CHALLENGER_MIN_ROC_AUC,
    )
    audit["comparison_reason"] = comparison_reason
    logger.info("Model comparison: %s", comparison_reason)

    # -----------------------------------------------------------------------
    # 6. Promotion or rejection
    # -----------------------------------------------------------------------
    if accept:
        # Register and promote challenger
        challenger_version = reg.register_model(
            run_id=challenger_run_id,
            model_name=MLFLOW_MODEL_NAME,
            artifact_path="model",
        )
        reg.set_alias(
            model_name=MLFLOW_MODEL_NAME,
            version=challenger_version,
            alias=CHAMPION_ALIAS,
        )
        reg.set_model_tag(
            model_name=MLFLOW_MODEL_NAME,
            version=challenger_version,
            key="model_type",
            value=model_type,
        )
        reg.set_model_tag(
            model_name=MLFLOW_MODEL_NAME,
            version=challenger_version,
            key="val_roc_auc",
            value=str(round(challenger_val_metrics["roc_auc"], 4)),
        )
        # Clean up old champion alias (it's now on new version)
        # Note: set_alias automatically moves it — no need to delete.

        decision = "promoted"
        logger.info(
            "CHAMPION UPDATED: version %s → %s",
            champion_version, challenger_version,
        )
        audit.update({
            "status": "success",
            "decision": "promoted",
            "new_champion_version": challenger_version,
        })

        # Mark batch as processed
        if trigger_batch is not None:
            _record_retrained_batch(trigger_batch)

    else:
        decision = "rejected"
        logger.info("Challenger rejected. Champion v%s retained.", champion_version)
        audit.update({
            "status": "success",
            "decision": "rejected",
            "retained_champion_version": champion_version,
        })

        # Still mark batch as processed so we don't retry indefinitely
        if trigger_batch is not None:
            _record_retrained_batch(trigger_batch)

    # Evaluate the held-out test partition only after the promotion decision,
    # for reporting. It never influences acceptance or rejection.
    champion_test_metrics = evaluate_pipeline(
        champion_pipeline, X_test, y_test, "champion_test_report"
    )
    challenger_test_metrics = evaluate_pipeline(
        challenger_pipeline, X_test, y_test, "challenger_test_report"
    )
    # Attach post-decision test results to the precise challenger run so the
    # dashboard never reuses a free-floating metrics file for another version.
    try:
        with mlflow.start_run(run_id=challenger_run_id):
            mlflow.log_metrics({
                "test_roc_auc": challenger_test_metrics["roc_auc"],
                "test_pr_auc": challenger_test_metrics["pr_auc"],
                "test_f1_macro": challenger_test_metrics["f1_macro"],
                "test_accuracy": challenger_test_metrics["accuracy"],
                "test_precision_macro": challenger_test_metrics["precision_macro"],
                "test_recall_macro": challenger_test_metrics["recall_macro"],
            })
            mlflow.set_tag("test_metric_split", "held_out_test")
    except Exception:
        # The audit record below remains the source of truth if a tracking
        # backend is unavailable after evaluation or a test double has no run.
        logger.warning("Could not attach test metrics to challenger run %s.", challenger_run_id, exc_info=True)
    audit["champion_test_roc_auc"] = champion_test_metrics["roc_auc"]
    audit["challenger_test_roc_auc"] = challenger_test_metrics["roc_auc"]
    audit["champion_metrics"] = {k: v for k, v in champion_test_metrics.items()
                                 if k not in ("classification_report", "confusion_matrix")}
    audit["challenger_metrics"] = {k: v for k, v in challenger_test_metrics.items()
                                   if k not in ("classification_report", "confusion_matrix")}
    _append_retrain_event(audit)

    return {
        "status": "success",
        "decision": decision,
        "reason": comparison_reason,
        "champion_version": champion_version,
        "challenger_version": (
            audit.get("new_champion_version") or challenger_run_id
        ),
        "champion_metrics": champion_test_metrics,
        "challenger_metrics": challenger_test_metrics,
    }
