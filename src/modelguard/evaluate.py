"""
Evaluation utilities — metrics, confusion matrix, classification report.

Design decisions
----------------
- Primary metric for model selection: ROC-AUC (insensitive to threshold).
- Secondary metrics reported: F1 (macro), precision, recall, PR-AUC.
- Accuracy is included for reference but NOT used for model selection.
- Predictions are at the default 0.5 threshold for classification metrics;
  ROC-AUC and PR-AUC are threshold-agnostic.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def evaluate_pipeline(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    split_name: str = "validation",
) -> dict[str, Any]:
    """
    Evaluate a fitted pipeline on a dataset split and return a metrics dict.

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline
    X        : feature DataFrame
    y        : true labels (0/1)
    split_name : label for logging (e.g. 'validation', 'test')

    Returns
    -------
    dict with keys: roc_auc, pr_auc, f1_macro, precision_macro,
                    recall_macro, accuracy, confusion_matrix, report,
                    n_samples, inference_latency_ms
    """
    logger.info("Evaluating on %s split (%d samples).", split_name, len(y))

    t0 = time.perf_counter()
    y_proba = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    roc_auc = float(roc_auc_score(y, y_proba))
    pr_auc = float(average_precision_score(y, y_proba))
    f1 = float(f1_score(y, y_pred, average="macro", zero_division=0))
    precision = float(precision_score(y, y_pred, average="macro", zero_division=0))
    recall = float(recall_score(y, y_pred, average="macro", zero_division=0))
    accuracy = float((y_pred == y.values).mean())
    cm = confusion_matrix(y, y_pred).tolist()
    report = classification_report(y, y_pred, zero_division=0)

    metrics: dict[str, Any] = {
        "split": split_name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1_macro": f1,
        "precision_macro": precision,
        "recall_macro": recall,
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "classification_report": report,
        "n_samples": int(len(y)),
        "inference_latency_ms": round(elapsed_ms, 3),
    }

    logger.info(
        "[%s] ROC-AUC=%.4f  PR-AUC=%.4f  F1=%.4f  Accuracy=%.4f",
        split_name, roc_auc, pr_auc, f1, accuracy,
    )
    return metrics


def compare_models(
    champion_metrics: dict,
    challenger_metrics: dict,
    improvement_threshold: float = 0.0,
    min_roc_auc: float = 0.65,
) -> tuple[bool, str]:
    """
    Decide whether the challenger should replace the champion.

    Acceptance criteria (ALL must pass)
    ------------------------------------
    1. Challenger ROC-AUC ≥ min_roc_auc.
    2. Challenger ROC-AUC ≥ champion ROC-AUC + improvement_threshold.

    Returns
    -------
    (accept: bool, reason: str)
    """
    challenger_roc = challenger_metrics["roc_auc"]
    champion_roc = champion_metrics["roc_auc"]

    if challenger_roc < min_roc_auc:
        reason = (
            f"Challenger ROC-AUC ({challenger_roc:.4f}) < "
            f"minimum threshold ({min_roc_auc:.4f}). Rejected."
        )
        return False, reason

    if challenger_roc < champion_roc + improvement_threshold:
        reason = (
            f"Challenger ROC-AUC ({challenger_roc:.4f}) does not exceed "
            f"champion ROC-AUC ({champion_roc:.4f}) + "
            f"improvement threshold ({improvement_threshold:.4f}). Rejected."
        )
        return False, reason

    reason = (
        f"Challenger accepted: ROC-AUC {challenger_roc:.4f} vs "
        f"champion {champion_roc:.4f} "
        f"(Δ = {challenger_roc - champion_roc:+.4f})."
    )
    return True, reason
