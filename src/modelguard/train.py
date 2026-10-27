"""
Training entry point.

Usage
-----
    python -m modelguard.train [--model {lr,rf,both}] [--data-path PATH]

Design decisions
----------------
- Two classifiers: Logistic Regression (lr) and Random Forest (rf).
- Class imbalance handled via class_weight='balanced' in both estimators.
- A separate MLflow run is created for each model.
- The pipeline (preprocessor + estimator) is logged as an MLflow model.
- After both runs, the better model (by validation ROC-AUC) is registered
  in the MLflow Model Registry and tagged with the 'champion' alias.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from modelguard.config import (
    ARTIFACTS_DIR,
    CHALLENGER_MIN_ROC_AUC,
    CHAMPION_ALIAS,
    LOGISTIC_REGRESSION_PARAMS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MLFLOW_ARTIFACT_ROOT,
    MLFLOW_MODEL_NAME,
    MLFLOW_MODEL_PIP_REQUIREMENTS,
    RANDOM_FOREST_PARAMS,
    RANDOM_SEED,
)
from modelguard.data import (
    load_raw_data,
    prepare_features,
    save_manifest,
    save_reference_data,
    split_data,
    validate_data,
)
from modelguard.evaluate import evaluate_pipeline
from modelguard.preprocessing import build_full_pipeline
from modelguard import registry as reg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _configure_mlflow() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # ensure artifact store root is set if using local filesystem
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger.info("MLflow tracking URI: %s", MLFLOW_TRACKING_URI)


def _train_one(
    model_name: str,
    estimator,
    X_train,
    y_train,
    X_val,
    y_val,
    data_fingerprint: str,
    extra_params: dict | None = None,
) -> tuple[dict, str]:
    """
    Train a single pipeline, log to MLflow, and return (metrics, run_id).
    """
    pipeline = build_full_pipeline(estimator)

    with mlflow.start_run(run_name=model_name) as run:
        run_id = run.info.run_id

        # Log hyper-parameters
        params = extra_params or {}
        params["model_type"] = model_name
        params["random_seed"] = RANDOM_SEED
        params["data_fingerprint"] = data_fingerprint
        mlflow.log_params(params)
        mlflow.set_tags({
            "validation_protocol": "baseline_validation_v1",
            "validation_split": "baseline_validation",
        })

        # Train
        t0 = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_duration_s = time.perf_counter() - t0
        logger.info(
            "[%s] Training finished in %.2fs.", model_name, train_duration_s
        )

        # Evaluate
        val_metrics = evaluate_pipeline(pipeline, X_val, y_val, "validation")

        # Log scalar metrics
        mlflow.log_metric("val_roc_auc", val_metrics["roc_auc"])
        mlflow.log_metric("val_pr_auc", val_metrics["pr_auc"])
        mlflow.log_metric("val_f1_macro", val_metrics["f1_macro"])
        mlflow.log_metric("val_precision_macro", val_metrics["precision_macro"])
        mlflow.log_metric("val_recall_macro", val_metrics["recall_macro"])
        mlflow.log_metric("val_accuracy", val_metrics["accuracy"])
        mlflow.log_metric("train_duration_s", train_duration_s)
        mlflow.log_metric(
            "val_inference_latency_ms", val_metrics["inference_latency_ms"]
        )

        # Log classification report as text artifact
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = ARTIFACTS_DIR / f"{model_name}_classification_report.txt"
        report_path.write_text(val_metrics["classification_report"])
        mlflow.log_artifact(str(report_path))

        # Log confusion matrix as JSON artifact
        cm_path = ARTIFACTS_DIR / f"{model_name}_confusion_matrix.json"
        cm_path.write_text(json.dumps(val_metrics["confusion_matrix"], indent=2))
        mlflow.log_artifact(str(cm_path))

        # Log with skops, MLflow's safe scikit-learn serialization format.
        # The fitted preprocessing state uses numpy.dtype and RandomForest
        # stores its trusted training tree as sklearn.tree._tree.Tree. The
        # specific types were reviewed for models created in this process;
        # never apply this allowlist to an externally supplied model file.
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=None,  # register manually after comparison
            input_example=X_val.head(1),
            serialization_format="skops",
            skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
            pip_requirements=MLFLOW_MODEL_PIP_REQUIREMENTS,
        )

        logger.info(
            "[%s] run_id=%s  val_roc_auc=%.4f",
            model_name, run_id, val_metrics["roc_auc"],
        )

    return val_metrics, run_id


# ---------------------------------------------------------------------------
# Main training workflow
# ---------------------------------------------------------------------------

def run_training(models: str = "both", data_path: Path | None = None) -> dict:
    """
    Execute the full training workflow.

    Parameters
    ----------
    models : 'lr' | 'rf' | 'both'
    data_path : override for the raw data file path

    Returns
    -------
    dict with keys 'best_model', 'best_run_id', 'val_metrics'
    """
    _configure_mlflow()

    # 1. Load and validate data
    df = load_raw_data(data_path) if data_path else load_raw_data()
    validate_data(df)

    # 2. Prepare features and split
    X, y = prepare_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # 3. Save manifest and reference data
    from modelguard.data import compute_fingerprint
    fingerprint = compute_fingerprint(df)
    save_manifest(
        df,
        splits={
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test),
        },
    )
    save_reference_data(X_train, y_train)

    # Preserve the same untouched validation rows for future champion /
    # challenger comparisons. Retraining must not carve a validation split
    # from the champion's original training reference data.
    validation_dir = ARTIFACTS_DIR / "validation_set"
    validation_dir.mkdir(parents=True, exist_ok=True)
    X_val.to_csv(validation_dir / "X_validation.csv", index=False)
    y_val.to_csv(validation_dir / "y_validation.csv", index=False)

    # 4. Persist test set for future champion/challenger evaluation
    test_dir = ARTIFACTS_DIR / "test_set"
    test_dir.mkdir(parents=True, exist_ok=True)
    X_test.to_csv(test_dir / "X_test.csv", index=False)
    y_test.to_csv(test_dir / "y_test.csv", index=False)
    logger.info("Test set saved to %s", test_dir)

    # 5. Train models
    results: dict[str, tuple[dict, str]] = {}

    if models in ("lr", "both"):
        lr = LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)
        val_metrics, run_id = _train_one(
            "logistic_regression", lr,
            X_train, y_train, X_val, y_val,
            fingerprint,
            extra_params=LOGISTIC_REGRESSION_PARAMS,
        )
        results["logistic_regression"] = (val_metrics, run_id)

    if models in ("rf", "both"):
        rf = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
        val_metrics, run_id = _train_one(
            "random_forest", rf,
            X_train, y_train, X_val, y_val,
            fingerprint,
            extra_params={
                k: v for k, v in RANDOM_FOREST_PARAMS.items()
                if k != "n_jobs"
            },
        )
        results["random_forest"] = (val_metrics, run_id)

    # 6. Select best model by validation ROC-AUC
    best_name = max(results, key=lambda k: results[k][0]["roc_auc"])
    best_metrics, best_run_id = results[best_name]
    logger.info(
        "Best model: %s (val_roc_auc=%.4f, run_id=%s)",
        best_name, best_metrics["roc_auc"], best_run_id,
    )

    # 7. Register the best model and promote to champion
    model_version = reg.register_model(
        run_id=best_run_id,
        model_name=MLFLOW_MODEL_NAME,
        artifact_path="model",
    )
    reg.set_alias(
        model_name=MLFLOW_MODEL_NAME,
        version=model_version,
        alias=CHAMPION_ALIAS,
    )
    reg.set_model_tag(
        model_name=MLFLOW_MODEL_NAME,
        version=model_version,
        key="model_type",
        value=best_name,
    )
    reg.set_model_tag(
        model_name=MLFLOW_MODEL_NAME,
        version=model_version,
        key="val_roc_auc",
        value=str(round(best_metrics["roc_auc"], 4)),
    )

    # Also evaluate best model on test set and log
    champion_pipeline = reg.load_champion()
    test_metrics = evaluate_pipeline(champion_pipeline, X_test, y_test, "test")
    # Test metrics are logged only after model selection, keeping the held-out
    # split out of the promotion decision while tying them to this exact run.
    with mlflow.start_run(run_id=best_run_id):
        mlflow.log_metrics({
            "test_roc_auc": test_metrics["roc_auc"],
            "test_pr_auc": test_metrics["pr_auc"],
            "test_f1_macro": test_metrics["f1_macro"],
            "test_accuracy": test_metrics["accuracy"],
            "test_precision_macro": test_metrics["precision_macro"],
            "test_recall_macro": test_metrics["recall_macro"],
        })
        mlflow.set_tag("test_metric_split", "held_out_test")
        mlflow.set_tag("test_dataset_fingerprint", fingerprint)
    logger.info(
        "Champion test metrics — ROC-AUC=%.4f  F1=%.4f",
        test_metrics["roc_auc"], test_metrics["f1_macro"],
    )

    # Save test metrics
    test_metrics_path = ARTIFACTS_DIR / "champion_test_metrics.json"
    import json
    test_metrics_out = {k: v for k, v in test_metrics.items()
                        if k != "classification_report"}
    test_metrics_out["classification_report"] = test_metrics["classification_report"]
    test_metrics_out.update({
        "model_name": MLFLOW_MODEL_NAME,
        "model_version": str(model_version),
        "run_id": best_run_id,
        "data_fingerprint": fingerprint,
    })
    with open(test_metrics_path, "w") as fh:
        json.dump(test_metrics_out, fh, indent=2, default=str)
    logger.info("Champion test metrics saved to %s", test_metrics_path)

    return {
        "best_model": best_name,
        "best_run_id": best_run_id,
        "model_version": model_version,
        "val_metrics": best_metrics,
        "test_metrics": test_metrics,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train ModelGuard classifiers.")
    parser.add_argument(
        "--model",
        choices=["lr", "rf", "both"],
        default="both",
        help="Which classifier(s) to train.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Override path to raw CSV dataset.",
    )
    args = parser.parse_args()
    result = run_training(models=args.model, data_path=args.data_path)
    logger.info("Training complete. Result: %s", result)


if __name__ == "__main__":
    main()
