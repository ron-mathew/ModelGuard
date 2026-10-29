"""Small service-policy helpers shared by the dashboard and its tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping


DASHBOARD_ONLY_MESSAGE = (
    "Dashboard-only deployment: live API predictions and live MLflow registry "
    "access are disabled. The dashboard only shows data available in local files."
)


def dashboard_only_enabled(env: Mapping[str, str]) -> bool:
    """Return whether the dashboard should avoid API and MLflow services."""
    return env.get("MODEL_GUARD_DASHBOARD_ONLY", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def api_url(env: Mapping[str, str], dashboard_only: bool) -> str | None:
    """Resolve the configured API URL without inventing one in offline mode."""
    if dashboard_only:
        return None
    value = env.get("MODEL_GUARD_API_URL")
    if value:
        return value.rstrip("/")
    return "http://localhost:5000"


def disabled_service_result(service: str) -> dict:
    return {
        "ok": False,
        "status": None,
        "data": {"error": f"{service} is disabled in dashboard-only mode."},
    }


def service_snapshot(
    dashboard_only: bool,
    api_base_url: str | None,
    request_json: Callable[[str], dict],
) -> dict:
    """Collect service status, or return explicit disabled states offline."""
    if dashboard_only:
        disabled = disabled_service_result("Prediction API")
        return {"health": disabled, "model": disabled, "metrics": disabled}
    if not api_base_url:
        unavailable = {
            "ok": False, "status": None,
            "data": {"error": "MODEL_GUARD_API_URL is not configured."},
        }
        return {"health": unavailable, "model": unavailable, "metrics": unavailable}
    return {
        "health": request_json(f"{api_base_url}/health"),
        "model": request_json(f"{api_base_url}/model-info"),
        "metrics": request_json(f"{api_base_url}/metrics"),
    }


def registry_snapshot(dashboard_only: bool, load_registry: Callable[[], dict]) -> dict:
    """Skip all registry access when the dashboard is explicitly offline."""
    if dashboard_only:
        return {
            "ok": False, "versions": [], "champion": None,
            "error": "Live MLflow registry access is disabled in dashboard-only mode.",
            "disabled": True,
        }
    return load_registry()
