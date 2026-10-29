"""Verified offline results bundle and dashboard-only rendering tests."""

from pathlib import Path

from dashboard.results import (
    empty_state_message,
    load_results_bundle,
    model_provenance_label,
)


def test_bundle_contains_verified_metrics_and_provenance():
    bundle = load_results_bundle()
    assert bundle is not None
    assert bundle["status"] == "historical_offline"
    assert len(bundle["dataset"]["fingerprint"]) == 64

    models = bundle["evaluation"]["models"]
    assert {model["model_type"] for model in models} == {"Logistic Regression", "Random Forest"}
    assert all(model["split"] == "validation" for model in models)
    assert all(model["sample_count"] == 6782 for model in models)
    assert "not asserted as a run-level" in bundle["dataset"]["run_linkage_note"]
    for model in models:
        assert model["run_id"]
        assert model["provenance"]["run_id"] == model["run_id"]
        assert model["provenance"]["dataset_fingerprint"] == bundle["dataset"]["fingerprint"]
        assert set(model["metrics"]) == {"roc_auc", "pr_auc", "macro_f1", "accuracy"}
    champion = next(model for model in models if model["model_type"] == "Random Forest")
    assert champion["version"] == 4
    assert champion["provenance"]["alias"] == "champion"
    label = model_provenance_label(champion)
    assert "v4" in label and "Validation split" in label and champion["run_id"] in label


def test_bundle_contains_historical_drift_and_retraining_records():
    bundle = load_results_bundle()
    assert bundle is not None
    scenarios = bundle["drift"]["scenarios"]
    normal = next(item for item in scenarios if not item["dataset_drift"])
    shifted = next(item for item in scenarios if item["dataset_drift"])
    assert normal["drift_threshold"] == shifted["drift_threshold"] == 0.5
    assert normal["drifted_features"] == []
    assert len(shifted["drifted_features"]) == 8
    assert len(shifted["feature_drift_scores"]) == 15
    assert "not live" in bundle["drift"]["notice"].lower()

    events = bundle["retraining"]["events"]
    assert [event["decision"] for event in events] == ["rejected", "skipped", "rejected"]
    assert not any(event["decision"] == "promoted" for event in events)
    assert "not recorded" in bundle["retraining"]["comparison_metric_label"].lower()


def test_missing_or_unrecognized_bundle_returns_empty_state(tmp_path):
    missing = tmp_path / "missing.json"
    assert load_results_bundle(missing) is None
    malformed = tmp_path / "malformed.json"
    malformed.write_text('{"schema_version": 99}', encoding="utf-8")
    assert load_results_bundle(malformed) is None
    assert "No verified" in empty_state_message("evaluation")
    assert "No verified" in empty_state_message("drift")
    assert "No verified" in empty_state_message("retraining")


def test_dashboard_only_all_pages_render_historical_bundle(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("MODEL_GUARD_DASHBOARD_ONLY", "true")
    app_path = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=120).run(timeout=120)
    assert not app.exception
    assert any("historical results" in item.value.lower() for item in app.caption)
    assert any("Historical/offline validation results" in item.value for item in app.markdown)
    assert len(app.dataframe) >= 2

    for page in ("Model Details", "Monitoring & Drift", "Retraining History", "Predictions"):
        app.radio[0].set_value(page).run(timeout=60)
        assert not app.exception, f"{page}: {[str(item) for item in app.exception]}"
        assert any("Dashboard-only deployment" in item.value for item in app.info)
        if page == "Model Details":
            assert any("historical/offline evaluation snapshot" in item.value.lower() for item in app.info)
        elif page == "Monitoring & Drift":
            assert any("historical checks are offline" in item.value.lower() for item in app.info)
        elif page == "Retraining History":
            assert any("historical audit records" in item.value.lower() for item in app.info)
        elif page == "Predictions":
            assert any("live prediction api is disabled" in item.value.lower() for item in app.info)
    predict = [button for button in app.button if button.label == "Predict"]
    assert predict and predict[0].disabled
