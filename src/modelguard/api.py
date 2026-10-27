"""
Flask Prediction API.

Endpoints
---------
GET  /health        — liveness check
GET  /model-info    — current champion version and metadata
POST /predict       — run inference on a single record
GET  /metrics       — operational metrics (prediction counts, error rate)

Design decisions
----------------
- The champion model is cached in memory and refreshed when registry aliases
  change, without interrupting inference.
- Invalid requests return 400 with a JSON error body.
- Internal errors return 500 with a JSON error body.
- Predictions are logged to a JSONL file for downstream monitoring.
- The API does NOT silently fall back to any model if the champion is missing.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request

from modelguard.config import (
    API_HOST,
    API_PORT,
    ARTIFACTS_DIR,
    CHAMPION_ALIAS,
    FEATURE_COLUMNS,
    LOG_FORMAT,
    LOG_LEVEL,
    MLFLOW_MODEL_NAME,
    MODEL_REFRESH_INTERVAL_SECONDS,
    PREDICTION_LOG_FILE,
)
from modelguard.data import validate_inference_schema
from modelguard import registry as reg

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
app = Flask(__name__)
_model_cache: dict = {
    "pipeline": None,
    "version": None,
    "loaded_at": None,
    "last_checked_monotonic": 0.0,
    "lock": threading.Lock(),
}
_operational_stats: dict = {
    "total_predictions": 0,
    "total_errors": 0,
    "startup_time": datetime.now(timezone.utc).isoformat(),
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_into_cache() -> None:
    """Load or reload the exact champion version. Thread-safe."""
    with _model_cache["lock"]:
        logger.info("Loading champion model into cache …")
        pipeline, version = reg.load_champion_versioned(
            model_name=MLFLOW_MODEL_NAME
        )
        _model_cache["pipeline"] = pipeline
        _model_cache["version"] = version
        _model_cache["loaded_at"] = datetime.now(timezone.utc).isoformat()
        _model_cache["last_checked_monotonic"] = time.monotonic()
        logger.info("Champion model v%s loaded successfully.", version)


def _refresh_model_if_due(force: bool = False) -> None:
    """Refresh the cached champion periodically without interrupting requests.

    A newly loaded model replaces the cache only after the exact registry
    version has loaded successfully. On load failure, the last good model
    remains available and the next check retries after the configured interval.
    """
    now = time.monotonic()
    if (
        not force
        and now - _model_cache.get("last_checked_monotonic", 0.0)
        < MODEL_REFRESH_INTERVAL_SECONDS
    ):
        return

    with _model_cache["lock"]:
        now = time.monotonic()
        if (
            not force
            and now - _model_cache.get("last_checked_monotonic", 0.0)
            < MODEL_REFRESH_INTERVAL_SECONDS
        ):
            return
        _model_cache["last_checked_monotonic"] = now
        try:
            champion_version = reg.get_champion_version()
            if champion_version is None or str(champion_version) == str(
                _model_cache.get("version")
            ):
                return
            pipeline, loaded_version = reg.load_champion_versioned(
                model_name=MLFLOW_MODEL_NAME,
                version=champion_version,
            )
        except Exception:
            logger.exception("Champion refresh failed; retaining last loaded model.")
            return

        _model_cache["pipeline"] = pipeline
        _model_cache["version"] = loaded_version
        _model_cache["loaded_at"] = datetime.now(timezone.utc).isoformat()
        logger.info("Refreshed active model to champion v%s.", loaded_version)


def _get_model_snapshot():
    _refresh_model_if_due()
    with _model_cache["lock"]:
        pipeline = _model_cache["pipeline"]
        version = _model_cache["version"]
    if pipeline is None:
        raise RuntimeError(
            "No champion model loaded. "
            "Ensure a champion is registered in MLflow before serving requests."
        )
    return pipeline, version


# ---------------------------------------------------------------------------
# Prediction logging
# ---------------------------------------------------------------------------

def _log_prediction(record: dict) -> None:
    PREDICTION_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PREDICTION_LOG_FILE, "a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """Liveness probe."""
    _refresh_model_if_due()
    with _model_cache["lock"]:
        model_loaded = _model_cache["pipeline"] is not None
        champion_version = _model_cache.get("version")
    return jsonify({
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "champion_version": champion_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }), 200 if model_loaded else 503


@app.route("/model-info", methods=["GET"])
def model_info():
    """Return metadata about the currently loaded model."""
    _refresh_model_if_due()
    with _model_cache["lock"]:
        version = _model_cache.get("version")
        loaded_at = _model_cache.get("loaded_at")
    if version is None:
        return jsonify({"error": "No model loaded."}), 503

    versions = reg.get_all_versions(model_name=MLFLOW_MODEL_NAME)
    current = next((v for v in versions if v["version"] == version), {})
    return jsonify({
        "model_name": MLFLOW_MODEL_NAME,
        "champion_version": version,
        "loaded_at": loaded_at,
        "tags": current.get("tags", {}),
        "aliases": current.get("aliases", []),
        "run_id": current.get("run_id"),
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Run inference on a single record.

    Request body (JSON)
    -------------------
    {
        "age": 35,
        "job": "technician",
        "marital": "single",
        "education": "secondary",
        "default": "no",
        "balance": 1200,
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "day": 15,
        "month": "may",
        "campaign": 1,
        "pdays": -1,
        "previous": 0,
        "poutcome": "unknown"
    }

    Response
    --------
    {
        "prediction": 0,
        "label": "no",
        "probability_yes": 0.12,
        "model_version": "1",
        "request_id": "...",
        "timestamp": "..."
    }
    """
    _operational_stats["total_predictions"] += 1
    ts = datetime.now(timezone.utc).isoformat()

    # Parse JSON
    data = request.get_json(silent=True)
    if data is None:
        _operational_stats["total_errors"] += 1
        return jsonify({"error": "Request body must be valid JSON."}), 400

    # Schema validation
    is_valid, err_msg = validate_inference_schema(data)
    if not is_valid:
        _operational_stats["total_errors"] += 1
        return jsonify({"error": err_msg}), 400

    # Run inference
    try:
        pipeline, served_version = _get_model_snapshot()
        input_df = pd.DataFrame([{col: data.get(col) for col in FEATURE_COLUMNS}])
        proba = pipeline.predict_proba(input_df)[0]
        prob_yes = float(proba[1])
        prediction = int(prob_yes >= 0.5)
        label = "yes" if prediction == 1 else "no"
    except RuntimeError as exc:
        _operational_stats["total_errors"] += 1
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        logger.exception("Prediction failed.")
        _operational_stats["total_errors"] += 1
        return jsonify({"error": "Internal server error during prediction."}), 500

    # Log prediction
    log_entry = {
        "timestamp": ts,
        "model_version": served_version,
        "input": {col: data.get(col) for col in FEATURE_COLUMNS},
        "prediction": prediction,
        "label": label,
        "probability_yes": round(prob_yes, 4),
    }
    _log_prediction(log_entry)

    return jsonify({
        "prediction": prediction,
        "label": label,
        "probability_yes": round(prob_yes, 4),
        "model_version": served_version,
        "timestamp": ts,
    }), 200


@app.route("/metrics", methods=["GET"])
def metrics():
    """Return operational metrics."""
    total = _operational_stats["total_predictions"]
    errors = _operational_stats["total_errors"]
    error_rate = (errors / total) if total > 0 else 0.0

    # Count predictions logged to file
    logged = 0
    if PREDICTION_LOG_FILE.exists():
        with open(PREDICTION_LOG_FILE) as fh:
            logged = sum(1 for line in fh if line.strip())

    return jsonify({
        "total_predictions_session": total,
        "total_errors_session": errors,
        "error_rate_session": round(error_rate, 4),
        "predictions_logged_file": logged,
        "startup_time": _operational_stats["startup_time"],
        "champion_version": _model_cache.get("version"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }), 200


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app(preload_model: bool = True) -> Flask:
    """Create and configure the Flask application."""
    if preload_model:
        try:
            load_model_into_cache()
        except Exception as exc:
            logger.error("Could not pre-load champion model: %s", exc)
            # API starts in degraded mode — /health will return 503
    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    application = create_app(preload_model=True)
    application.run(host=API_HOST, port=API_PORT, debug=False)
