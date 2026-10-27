"""
MLflow Model Registry helpers.

Wraps MLflow client calls so the rest of the codebase stays decoupled
from the SDK version-specific API.

MLflow 3.x uses model aliases instead of stage transitions.
The champion is identified by the alias defined in config.CHAMPION_ALIAS.
"""

from __future__ import annotations

import logging
from typing import Optional

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from modelguard.config import (
    CHAMPION_ALIAS,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
)

logger = logging.getLogger(__name__)


def _client() -> MlflowClient:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


def register_model(
    run_id: str,
    model_name: str = MLFLOW_MODEL_NAME,
    artifact_path: str = "model",
) -> str:
    """
    Register a model artifact from a completed run.

    Returns the registered model version string.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = _client()
    model_uri = f"runs:/{run_id}/{artifact_path}"
    result = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )
    logger.info(
        "Registered model '%s' version %s from run %s.",
        model_name, result.version, run_id,
    )
    return result.version


def set_alias(
    model_name: str,
    version: str,
    alias: str,
) -> None:
    """Assign an alias to a model version."""
    client = _client()
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=version,
    )
    logger.info(
        "Alias '%s' → %s version %s.", alias, model_name, version
    )


def delete_alias(model_name: str, alias: str) -> None:
    """Remove an alias from a model (no-op if not set)."""
    client = _client()
    try:
        client.delete_registered_model_alias(name=model_name, alias=alias)
        logger.info("Alias '%s' removed from '%s'.", alias, model_name)
    except Exception as exc:
        logger.debug("Could not delete alias '%s': %s", alias, exc)


def set_model_tag(
    model_name: str,
    version: str,
    key: str,
    value: str,
) -> None:
    """Set a metadata tag on a model version."""
    client = _client()
    client.set_model_version_tag(
        name=model_name, version=version, key=key, value=value
    )


def get_champion_version() -> Optional[str]:
    """
    Return the version number of the current champion, or None if unset.
    """
    client = _client()
    try:
        mv = client.get_model_version_by_alias(
            name=MLFLOW_MODEL_NAME, alias=CHAMPION_ALIAS
        )
        return str(mv.version)
    except Exception:
        return None


def load_champion(model_name: str = MLFLOW_MODEL_NAME):
    """
    Load and return the champion model pipeline.

    Raises
    ------
    RuntimeError if no champion is registered.
    """
    pipeline, _version = load_champion_versioned(model_name=model_name)
    return pipeline


def load_champion_versioned(
    model_name: str = MLFLOW_MODEL_NAME,
    version: Optional[str] = None,
):
    """Load one immutable registry version and return ``(model, version)``.

    Resolving the alias once and loading its explicit version prevents the
    loaded pipeline from being mislabeled if promotion happens mid-load.
    """
    version = version or get_champion_version()
    if version is None:
        raise RuntimeError(
            f"No champion model found for '{model_name}'. "
            "Run training first to register a champion."
        )
    uri = f"models:/{model_name}/{version}"
    logger.info("Loading champion model: %s", uri)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    pipeline = mlflow.sklearn.load_model(uri)
    return pipeline, str(version)


def get_all_versions(model_name: str = MLFLOW_MODEL_NAME) -> list[dict]:
    """Return summary dicts for all registered versions of a model."""
    client = _client()
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        # Some MLflow 3.x backend responses omit aliases from version search
        # results. The registered-model alias map is authoritative.
        alias_map = client.get_registered_model(model_name).aliases or {}
        aliases_by_version: dict[str, list[str]] = {}
        for alias, version in alias_map.items():
            aliases_by_version.setdefault(str(version), []).append(alias)
        return [
            {
                "version": str(v.version),
                "run_id": v.run_id,
                "tags": dict(v.tags),
                "aliases": sorted(
                    set(v.aliases or []) | set(aliases_by_version.get(str(v.version), []))
                ),
                "creation_timestamp": v.creation_timestamp,
            }
            for v in versions
        ]
    except Exception as exc:
        logger.warning("Could not fetch model versions: %s", exc)
        return []


def get_run_metrics(run_id: str) -> dict:
    """Return metrics dict for a specific MLflow run."""
    client = _client()
    run = client.get_run(run_id)
    return dict(run.data.metrics)
