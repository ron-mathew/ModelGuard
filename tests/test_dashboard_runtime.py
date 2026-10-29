"""Service selection behavior for full-stack and dashboard-only deployments."""

from dashboard.runtime import (
    DASHBOARD_ONLY_MESSAGE,
    api_url,
    dashboard_only_enabled,
    registry_snapshot,
    service_snapshot,
)


def test_dashboard_only_flag_is_explicit_and_case_insensitive():
    assert dashboard_only_enabled({"MODEL_GUARD_DASHBOARD_ONLY": "true"})
    assert dashboard_only_enabled({"MODEL_GUARD_DASHBOARD_ONLY": " YES "})
    assert not dashboard_only_enabled({})
    assert not dashboard_only_enabled({"MODEL_GUARD_DASHBOARD_ONLY": "false"})


def test_dashboard_only_has_no_localhost_fallback_and_clear_status():
    assert api_url({}, dashboard_only=True) is None
    assert api_url({"MODEL_GUARD_API_URL": "http://localhost:5000"}, dashboard_only=True) is None
    assert "live API predictions" in DASHBOARD_ONLY_MESSAGE
    assert "live MLflow registry access" in DASHBOARD_ONLY_MESSAGE


def test_dashboard_only_skips_api_and_registry_calls():
    calls = []

    def request(url):
        calls.append(url)
        return {"ok": True}

    services = service_snapshot(True, None, request)
    registry = registry_snapshot(True, lambda: calls.append("registry"))

    assert calls == []
    assert set(services) == {"health", "model", "metrics"}
    assert all(not result["ok"] for result in services.values())
    assert all("disabled" in result["data"]["error"].lower() for result in services.values())
    assert registry["disabled"] is True
    assert registry["versions"] == []
    assert registry["champion"] is None
    assert "disabled" in registry["error"].lower()


def test_full_stack_uses_configured_api_and_registry():
    calls = []

    def request(url):
        calls.append(url)
        return {"ok": True, "url": url}

    configured_url = api_url(
        {"MODEL_GUARD_API_URL": "http://api:5000/"}, dashboard_only=False
    )
    services = service_snapshot(False, configured_url, request)
    registry = registry_snapshot(False, lambda: {"ok": True, "versions": ["1"]})

    assert calls == [
        "http://api:5000/health",
        "http://api:5000/model-info",
        "http://api:5000/metrics",
    ]
    assert all(result["ok"] for result in services.values())
    assert registry == {"ok": True, "versions": ["1"]}


def test_missing_full_stack_api_url_does_not_call_localhost():
    calls = []
    services = service_snapshot(False, None, lambda url: calls.append(url))
    assert calls == []
    assert all("not configured" in result["data"]["error"] for result in services.values())
