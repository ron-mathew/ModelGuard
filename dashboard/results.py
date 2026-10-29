"""Read-only access to the small, provenance-bearing offline results bundle."""

from __future__ import annotations

import json
from pathlib import Path


RESULTS_BUNDLE_PATH = Path(__file__).resolve().parent / "results" / "results_bundle.json"
EMPTY_STATE_MESSAGES = {
    "evaluation": "No verified offline evaluation bundle is available in this deployment.",
    "drift": "No verified offline monitoring bundle is available in this deployment.",
    "retraining": "No verified offline retraining bundle is available in this deployment.",
}


def load_results_bundle(path: Path | str = RESULTS_BUNDLE_PATH) -> dict | None:
    """Load a recognized historical bundle; missing or malformed data stays empty."""
    try:
        with Path(path).open(encoding="utf-8") as handle:
            bundle = json.load(handle)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(bundle, dict) or bundle.get("schema_version") != 1:
        return None
    if bundle.get("status") != "historical_offline":
        return None
    evaluation = bundle.get("evaluation")
    if not isinstance(evaluation, dict) or not isinstance(evaluation.get("models"), list):
        return None
    if not evaluation["models"]:
        return None
    return bundle


def model_provenance_label(model: dict) -> str:
    """Build a visible label from the run identity and recorded split."""
    version = model.get("version")
    version_label = f"v{version} · " if version is not None else ""
    return (
        f"{model.get('model_type', 'Model')} {version_label}"
        f"{str(model.get('split', 'unknown')).title()} split · "
        f"MLflow run {model.get('run_id', 'unavailable')}"
    )


def empty_state_message(section: str) -> str:
    """Return an explicit, truthful message for a missing results section."""
    return EMPTY_STATE_MESSAGES.get(
        section, "No verified historical results are available for this section."
    )
