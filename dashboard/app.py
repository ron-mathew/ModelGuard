"""ModelGuard's Streamlit operations console."""

from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from modelguard.config import (  # noqa: E402
    ARTIFACTS_DIR,
    CATEGORICAL_FEATURES,
    DATA_SEPARATOR,
    DRIFT_REPORTS_DIR,
    DRIFT_THRESHOLD,
    FEATURE_COLUMNS,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    MONITORING_REPORTS_DIR,
    NUMERICAL_FEATURES,
    PREDICTION_LOG_FILE,
    RAW_DATA_FILE,
)
from modelguard import registry as reg  # noqa: E402
from modelguard.monitoring import load_monitoring_history  # noqa: E402
from modelguard.retrain import load_retrain_history  # noqa: E402
from runtime import (  # noqa: E402
    DASHBOARD_ONLY_MESSAGE,
    api_url as resolve_api_url,
    dashboard_only_enabled,
    registry_snapshot as apply_registry_mode,
    service_snapshot as collect_service_snapshot,
)


DASHBOARD_ONLY = dashboard_only_enabled(os.environ)
API_URL = resolve_api_url(os.environ, DASHBOARD_ONLY)
NAV_ITEMS = [
    "Overview",
    "Model Details",
    "Monitoring & Drift",
    "Retraining History",
    "Predictions",
]

st.set_page_config(
    page_title="ModelGuard | MLOps Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
      --mg-bg: #0b1220; --mg-panel: #111c2d; --mg-panel-2: #152338;
      --mg-border: #26364d; --mg-text: #edf3fb; --mg-muted: #91a2b9;
      --mg-blue: #64a9ff; --mg-green: #42d3a2; --mg-amber: #f4be62;
      --mg-red: #ff7182;
    }
    [data-testid="stAppViewContainer"] { background: var(--mg-bg); color: var(--mg-text); }
    [data-testid="stHeader"] { background: rgba(11,18,32,.92); }
    [data-testid="stSidebar"] { background: #0e1828; border-right: 1px solid var(--mg-border); }
    [data-testid="stSidebar"] > div { padding-top: 1rem; }
    .block-container { max-width: 1440px; padding-top: 1.35rem; padding-bottom: 2.5rem; }
    h1, h2, h3 { letter-spacing: -.025em; }
    h1 { font-size: 1.72rem !important; margin-bottom: .15rem !important; }
    h2 { font-size: 1.12rem !important; margin-top: .5rem !important; }
    h3 { font-size: .96rem !important; }
    p, label, [data-testid="stCaptionContainer"] { color: var(--mg-muted); }
    [data-testid="stMetric"] { background: var(--mg-panel); border: 1px solid var(--mg-border);
      border-radius: 12px; padding: .85rem 1rem; min-height: 102px; }
    [data-testid="stMetricLabel"] { color: var(--mg-muted); font-size: .78rem; }
    [data-testid="stMetricValue"] { color: var(--mg-text); font-size: 1.42rem; }
    [data-testid="stVerticalBlockBorderWrapper"] { background: var(--mg-panel); border-color: var(--mg-border); border-radius: 12px; }
    [data-testid="stDataFrame"] { border: 1px solid var(--mg-border); border-radius: 10px; overflow: hidden; }
    div[data-testid="stAlert"] { border-radius: 10px; }
    .mg-brand { font-size: 1.12rem; font-weight: 750; color: var(--mg-text); letter-spacing: -.03em; }
    .mg-subbrand { color: var(--mg-muted); font-size: .76rem; margin: .15rem 0 1.1rem; }
    .mg-kicker { color: var(--mg-blue); text-transform: uppercase; letter-spacing: .12em;
      font-size: .66rem; font-weight: 700; margin-bottom: .2rem; }
    .mg-page-note { color: var(--mg-muted); font-size: .88rem; margin-bottom: 1.2rem; }
    .mg-badge { display: inline-flex; align-items: center; border: 1px solid currentColor;
      border-radius: 99px; padding: .22rem .58rem; font-size: .72rem; font-weight: 650;
      line-height: 1.25; white-space: nowrap; }
    .mg-healthy { color: var(--mg-green); background: rgba(66,211,162,.08); }
    .mg-warning { color: var(--mg-amber); background: rgba(244,190,98,.08); }
    .mg-drift { color: var(--mg-red); background: rgba(255,113,130,.08); }
    .mg-accepted { color: var(--mg-green); background: rgba(66,211,162,.08); }
    .mg-rejected { color: #b0bfd2; background: rgba(145,162,185,.08); }
    .mg-neutral { color: var(--mg-blue); background: rgba(100,169,255,.08); }
    .mg-event { border-bottom: 1px solid var(--mg-border); padding: .72rem 0; }
    .mg-event:last-child { border-bottom: 0; }
    .mg-event-title { color: var(--mg-text); font-size: .86rem; font-weight: 600; }
    .mg-event-meta { color: var(--mg-muted); font-size: .75rem; margin-top: .22rem; }
    .mg-divider { border-top: 1px solid var(--mg-border); margin: .9rem 0 1.15rem; }
    button[kind="primary"] { background: #347fd0; border-color: #347fd0; }
    a { color: var(--mg-blue) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _json_request(url: str, timeout: float = 2.5, body: dict | None = None) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return {"ok": True, "status": response.status, "data": json.loads(response.read())}
    except HTTPError as exc:
        try:
            detail = json.loads(exc.read())
        except Exception:
            detail = {"error": str(exc)}
        return {"ok": False, "status": exc.code, "data": detail}
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "status": None, "data": {"error": str(exc)}}


def _safe_json(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _jsonl(path: Path) -> list[dict]:
    events: list[dict] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        events.append(item)
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return events


def _report_path(value: str | None) -> Path | None:
    if not value:
        return None
    candidate = Path(value)
    if candidate.is_file():
        return candidate
    # A report may have been written on another OS/container path while its
    # basename is still available in the shared reports volume.
    for folder in (DRIFT_REPORTS_DIR, MONITORING_REPORTS_DIR):
        local_candidate = folder / candidate.name
        if local_candidate.is_file():
            return local_candidate
    return None


def _load_monitoring() -> list[dict]:
    try:
        events = load_monitoring_history()
    except Exception:
        events = _jsonl(MONITORING_REPORTS_DIR / "monitoring_log.jsonl")
    enriched = []
    for event in events:
        item = dict(event)
        report_file = _report_path(item.get("report_json"))
        report = _safe_json(report_file) if report_file else {}
        item["report_data"] = report
        item["report_file"] = str(report_file) if report_file else None
        if report:
            for key in (
                "feature_drift_scores", "n_drifted", "n_features",
                "n_current_samples", "n_reference_samples", "threshold_used",
            ):
                if key in report:
                    item[key] = report[key]
        enriched.append(item)
    return enriched


def _load_retraining() -> list[dict]:
    try:
        return load_retrain_history()
    except Exception:
        return _jsonl(MONITORING_REPORTS_DIR / "retraining_log.jsonl")


def _prediction_frame() -> pd.DataFrame:
    rows = _jsonl(PREDICTION_LOG_FILE)
    frame = pd.DataFrame(rows)
    if not frame.empty and "timestamp" in frame:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    if not frame.empty and "model_version" in frame:
        frame["model_version"] = frame["model_version"].astype(str)
    for column in ("prediction", "probability_yes"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _query_registry_snapshot() -> dict:
    try:
        import mlflow
        from mlflow import MlflowClient

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        # A successful server/client query is the health check; a local file's
        # existence alone does not imply that the tracking store is usable.
        client.search_experiments(max_results=1)
        versions = reg.get_all_versions(model_name=MLFLOW_MODEL_NAME)
        for row in versions:
            try:
                version_run = client.get_run(row["run_id"])
                row["metrics"] = dict(version_run.data.metrics)
                row["params"] = dict(version_run.data.params)
                row["run_tags"] = dict(version_run.data.tags)
            except Exception as exc:
                row["run_error"] = str(exc)
        champion_version = reg.get_champion_version()
        if champion_version is None:
            return {"ok": True, "versions": versions, "champion": None, "error": "No champion alias is set."}
        current = next(
            (row for row in versions if str(row.get("version")) == str(champion_version)),
            None,
        )
        if current is None:
            return {"ok": True, "versions": versions, "champion": None,
                    "error": f"Champion alias points to v{champion_version}, but that version was not returned."}
        return {"ok": True, "versions": versions,
                "champion": current, "champion_version": str(champion_version)}
    except Exception as exc:
        return {"ok": False, "versions": [], "champion": None, "error": str(exc)}


def _load_registry_snapshot() -> dict:
    """Query MLflow only in the normal full-stack deployment."""
    return apply_registry_mode(DASHBOARD_ONLY, _query_registry_snapshot)


def _service_snapshot() -> dict:
    return collect_service_snapshot(DASHBOARD_ONLY, API_URL, _json_request)


def _matching_test_metrics(version: str, run_id: str, history: list[dict], run_metrics: dict) -> dict:
    """Return only test metrics whose model version/run can be verified."""
    for event in reversed(history):
        if str(event.get("new_champion_version", "")) == str(version):
            metrics = event.get("challenger_metrics")
            if isinstance(metrics, dict) and metrics.get("split", "").endswith("test_report"):
                return {"metrics": metrics, "source": "retraining audit", "run_id": event.get("challenger_run_id")}
        if str(event.get("champion_version", "")) == str(version):
            metrics = event.get("champion_metrics")
            if isinstance(metrics, dict) and metrics.get("split", "").endswith("test_report"):
                return {"metrics": metrics, "source": "retraining audit", "run_id": run_id}

    # Newly trained versions log their test split metrics on the exact MLflow run.
    test_prefixes = {
        "roc_auc": "test_roc_auc", "pr_auc": "test_pr_auc",
        "f1_macro": "test_f1_macro", "accuracy": "test_accuracy",
        "precision_macro": "test_precision_macro", "recall_macro": "test_recall_macro",
    }
    if all(key in run_metrics for key in test_prefixes.values()):
        return {"metrics": {key: run_metrics[metric] for key, metric in test_prefixes.items()},
                "source": "MLflow run metrics", "run_id": run_id}

    # Accept a report file only when it contains an explicit identity match.
    legacy = _safe_json(ARTIFACTS_DIR / "champion_test_metrics.json")
    if str(legacy.get("run_id", "")) == str(run_id) and str(legacy.get("model_version", "")) == str(version):
        return {"metrics": legacy, "source": "version-linked evaluation artifact", "run_id": run_id}
    return {"metrics": {}, "source": None, "run_id": None,
            "unlinked_artifact": bool(legacy)}


def _version_test_map(versions: list[dict], history: list[dict]) -> dict[str, dict]:
    mapped: dict[str, dict] = {}
    for row in versions:
        version = str(row.get("version"))
        metrics = row.get("metrics", {})
        mapped[version] = _matching_test_metrics(version, row.get("run_id", ""), history, metrics)
    for event in history:
        champion_version = event.get("champion_version")
        if champion_version is not None and isinstance(event.get("champion_metrics"), dict):
            if event["champion_metrics"].get("split", "").endswith("test_report"):
                mapped.setdefault(str(champion_version), {
                    "metrics": event["champion_metrics"], "source": "retraining audit",
                    "run_id": event.get("champion_run_id"),
                })
        new_version = event.get("new_champion_version")
        if new_version is not None and isinstance(event.get("challenger_metrics"), dict):
            if event["challenger_metrics"].get("split", "").endswith("test_report"):
                mapped[str(new_version)] = {
                    "metrics": event["challenger_metrics"], "source": "retraining audit",
                    "run_id": event.get("challenger_run_id"),
                }
    return mapped


def _badge(label: str, tone: str = "neutral") -> None:
    safe_label = html.escape(str(label))
    tone = tone if tone in {"healthy", "warning", "drift", "accepted", "rejected", "neutral"} else "neutral"
    st.markdown(f'<span class="mg-badge mg-{tone}">{safe_label}</span>', unsafe_allow_html=True)


def _page_header(title: str, subtitle: str) -> None:
    st.markdown('<div class="mg-kicker">Model operations</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="mg-page-note">{html.escape(subtitle)}</div>', unsafe_allow_html=True)


def _event_card(title: str, detail: str, badge: str, tone: str) -> None:
    st.markdown(
        '<div class="mg-event">'
        f'<div class="mg-event-title">{html.escape(title)}</div>'
        f'<div class="mg-event-meta">{html.escape(detail)}</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    _badge(badge, tone)


def _format_time(value: object) -> str:
    if not value:
        return "Time unavailable"
    try:
        parsed = pd.to_datetime(value, unit="ms", utc=True) if isinstance(value, (int, float)) else pd.to_datetime(value, utc=True)
        return parsed.strftime("%b %d, %Y · %H:%M UTC")
    except Exception:
        return str(value)


def _status_text(response: dict, key: str = "status") -> tuple[str, str]:
    if not response.get("ok"):
        error = response.get("data", {}).get("error", "Service is unreachable")
        return "Unavailable", str(error)
    data = response.get("data", {})
    if key == "status":
        value = str(data.get("status", "unknown")).lower()
        return ("Healthy" if value == "ok" else "Degraded"), str(data.get("champion_version", ""))
    return "Connected", ""


def _metric_cards(metrics: dict, split: str, note: str | None = None) -> None:
    st.caption(f"Evaluation split: **{split}**" + (f" · {note}" if note else ""))
    definitions = [
        ("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"),
        ("Macro F1", "f1_macro"), ("Accuracy", "accuracy"),
    ]
    cols = st.columns(4)
    for col, (label, key) in zip(cols, definitions):
        value = metrics.get(key)
        col.metric(label, f"{float(value):.4f}" if value is not None else "—")


def _current_version_metrics(champion: dict | None) -> dict:
    return champion.get("metrics", {}) if champion else {}


def _version_table(versions: list[dict], test_map: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for item in sorted(versions, key=lambda value: int(value.get("version", 0)), reverse=True):
        metrics = item.get("metrics", {})
        test = test_map.get(str(item.get("version")), {}).get("metrics", {})
        rows.append({
            "Version": f"v{item.get('version')}",
            "Alias": "Champion" if "champion" in item.get("aliases", []) else "—",
            "Model": item.get("tags", {}).get("model_type", "Unknown").replace("_", " ").title(),
            "Validation ROC-AUC": metrics.get("val_roc_auc"),
            "Validation PR-AUC": metrics.get("val_pr_auc"),
            "Validation Macro F1": metrics.get("val_f1_macro"),
            "Test ROC-AUC": test.get("roc_auc"),
            "Created": _format_time(item.get("creation_timestamp")),
        })
    return pd.DataFrame(rows)


def _latest_events(history: list[dict], key: str, count: int = 4) -> list[dict]:
    return sorted(history, key=lambda item: str(item.get("timestamp", "")), reverse=True)[:count]


def _categorical_values() -> dict[str, list[str]]:
    try:
        frame = pd.read_csv(RAW_DATA_FILE, sep=DATA_SEPARATOR, usecols=CATEGORICAL_FEATURES)
        return {column: sorted(frame[column].dropna().astype(str).unique().tolist()) for column in CATEGORICAL_FEATURES}
    except Exception:
        return {}


# Sidebar and data snapshot
with st.sidebar:
    st.markdown('<div class="mg-brand">◈ ModelGuard</div>', unsafe_allow_html=True)
    st.markdown('<div class="mg-subbrand">MLOps monitoring console</div>', unsafe_allow_html=True)
    page = st.radio("Workspace", NAV_ITEMS, label_visibility="collapsed")
    st.markdown('<div class="mg-divider"></div>', unsafe_allow_html=True)
    st.caption("Registry")
    st.code("Disabled in this deployment" if DASHBOARD_ONLY else MLFLOW_TRACKING_URI, language=None)
    st.caption("Model")
    st.code(MLFLOW_MODEL_NAME, language=None)
    if st.button("Refresh data", width="stretch"):
        st.rerun()

registry = _load_registry_snapshot()
monitoring = _load_monitoring()
retraining = _load_retraining()
predictions = _prediction_frame()
services = _service_snapshot()
versions = registry.get("versions", [])
champion = registry.get("champion")
champion_version = str(registry.get("champion_version", ""))
test_map = _version_test_map(versions, retraining)
current_test = test_map.get(champion_version, {"metrics": {}, "source": None})

if DASHBOARD_ONLY:
    st.info(DASHBOARD_ONLY_MESSAGE)


if page == "Overview":
    _page_header(
        "Overview",
        "Packaged model information and available local activity."
        if DASHBOARD_ONLY else "Current champion, evaluation results and live service status.",
    )
    if DASHBOARD_ONLY:
        st.caption("Model details and evaluation metrics require live MLflow registry access and are not queried here.")
    elif not registry.get("ok"):
        st.error(f"MLflow registry unavailable: {registry.get('error', 'connection failed')}")
    elif not champion:
        st.warning(registry.get("error", "No champion is registered."))

    if champion:
        identity, val_score = st.columns([1.35, 2.65])
        with identity:
            with st.container(border=True):
                st.caption("ACTIVE CHAMPION")
                st.markdown(f"### v{champion_version} · {champion.get('tags', {}).get('model_type', 'model').replace('_', ' ').title()}")
                st.caption(MLFLOW_MODEL_NAME)
                api_version = str(services["model"].get("data", {}).get("champion_version", ""))
                if services["model"].get("ok") and api_version != champion_version:
                    _badge(f"API serves v{api_version or 'unknown'}", "warning")
                else:
                    _badge("Serving version aligned" if services["model"].get("ok") else "API unavailable", "healthy" if services["model"].get("ok") else "warning")
        with val_score:
            st.markdown("#### Champion evaluation")
            run_metrics = _current_version_metrics(champion)
            val_values = {
                "roc_auc": run_metrics.get("val_roc_auc"),
                "pr_auc": run_metrics.get("val_pr_auc"),
                "f1_macro": run_metrics.get("val_f1_macro"),
                "accuracy": run_metrics.get("val_accuracy"),
            }
            if any(value is not None for value in val_values.values()):
                _metric_cards(val_values, "Validation", "logged on the champion's MLflow run")
            else:
                st.info("Validation metrics are not available for this champion run.")

        if current_test.get("metrics"):
            st.markdown("#### Final test evaluation")
            _metric_cards(current_test["metrics"], "Held-out test", f"version-linked via {current_test.get('source')}")
        else:
            st.info("No held-out test metrics are linked to this champion version. The dashboard is withholding unlinked artifacts.")

    st.markdown("#### Service status")
    api_label, api_detail = _status_text(services["health"])
    mlflow_label = "Disabled" if DASHBOARD_ONLY else "Connected" if registry.get("ok") else "Unavailable"
    api_metrics = services["metrics"].get("data", {}) if services["metrics"].get("ok") else {}
    logger_ok = services["metrics"].get("ok") and "predictions_logged_file" in api_metrics
    status_cols = st.columns(3)
    with status_cols[0]:
        with st.container(border=True):
            st.caption("PREDICTION API")
            _badge("Disabled" if DASHBOARD_ONLY else api_label, "neutral" if DASHBOARD_ONLY else "healthy" if api_label == "Healthy" else "warning")
            st.caption("Live predictions are disabled." if DASHBOARD_ONLY else f"{API_URL} · served v{api_detail or '—'}")
    with status_cols[1]:
        with st.container(border=True):
            st.caption("MLFLOW TRACKING / REGISTRY")
            _badge(mlflow_label, "neutral" if DASHBOARD_ONLY else "healthy" if registry.get("ok") else "warning")
            st.caption("Live registry access is disabled." if DASHBOARD_ONLY else "Live tracking-store query")
    with status_cols[2]:
        with st.container(border=True):
            st.caption("PREDICTION LOGGING")
            _badge("Local file only" if DASHBOARD_ONLY else "Writing" if logger_ok else "Unavailable", "neutral" if DASHBOARD_ONLY else "healthy" if logger_ok else "warning")
            st.caption(f"{len(predictions)} locally available prediction records" if DASHBOARD_ONLY and not predictions.empty else "No local prediction records are available." if DASHBOARD_ONLY else f"{api_metrics.get('predictions_logged_file', 0)} persisted predictions")

    st.markdown("#### Recent activity")
    left, right = st.columns(2)
    with left:
        st.markdown("**Monitoring**")
        if monitoring:
            for event in _latest_events(monitoring, "timestamp"):
                drift = bool(event.get("dataset_drift"))
                drifted = event.get("drifted_features", [])
                _event_card(
                    str(event.get("batch_label", "Monitoring run")),
                    f"{_format_time(event.get('timestamp'))} · {len(drifted)} feature(s) flagged · retrain trigger {'met' if event.get('should_retrain') else 'not met'}",
                    "Drift detected" if drift else "No drift",
                    "drift" if drift else "healthy",
                )
        else:
            st.caption(
                "No monitoring log is available; monitoring events are not generated by this dashboard-only deployment."
                if DASHBOARD_ONLY else "No monitoring events have been recorded."
            )
    with right:
        st.markdown("**Retraining decisions**")
        if retraining:
            for event in _latest_events(retraining, "timestamp"):
                decision = event.get("decision", event.get("status", "unknown"))
                tone = "accepted" if decision == "promoted" else "rejected" if decision == "rejected" else "warning"
                version_text = event.get("new_champion_version") or event.get("retained_champion_version") or event.get("champion_version", "—")
                _event_card(
                    f"{str(decision).replace('_', ' ').title()} · v{version_text}",
                    f"{_format_time(event.get('timestamp'))} · {event.get('model_type', 'challenger')}",
                    "Promotion accepted" if decision == "promoted" else "Promotion rejected" if decision == "rejected" else str(decision).title(),
                    tone,
                )
        else:
            st.caption(
                "No retraining history file is available in this dashboard-only deployment."
                if DASHBOARD_ONLY else "No retraining decisions have been recorded."
            )

    valid_points = []
    for event in sorted(retraining, key=lambda item: str(item.get("timestamp", ""))):
        if event.get("comparison_protocol") != "baseline_validation_v1":
            continue
        if event.get("decision") == "promoted":
            score = event.get("challenger_validation_roc_auc")
            version = event.get("new_champion_version")
        else:
            score = event.get("champion_validation_roc_auc")
            version = event.get("champion_version")
        if score is not None and version is not None:
            valid_points.append({"Timestamp": _format_time(event.get("timestamp")), "Champion version": f"v{version}", "Validation ROC-AUC": float(score)})
    if len(valid_points) >= 2:
        st.markdown("#### Comparable champion trend")
        st.caption("Only retraining events using the fixed baseline validation protocol are plotted.")
        trend = pd.DataFrame(valid_points).set_index("Timestamp")
        st.line_chart(trend[["Validation ROC-AUC"]], height=220)


elif page == "Model Details":
    _page_header("Model details", "Registry lineage, active-run metrics and version comparisons.")
    if DASHBOARD_ONLY:
        st.info("Model details, versions, and MLflow metrics are unavailable because live registry access is disabled in this deployment.")
    elif not registry.get("ok"):
        st.error(f"Cannot query MLflow: {registry.get('error', 'connection failed')}")
    elif not champion:
        st.info("No champion alias is currently assigned in the configured registry.")
    else:
        meta, run_metrics = champion, champion.get("metrics", {})
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Champion", f"v{champion_version}")
        col_b.metric("Model type", meta.get("tags", {}).get("model_type", "Unknown").replace("_", " ").title())
        col_c.metric("Validation ROC-AUC", f"{run_metrics['val_roc_auc']:.4f}" if run_metrics.get("val_roc_auc") is not None else "—")
        st.markdown("#### Evaluation on this version")
        val_values = {"roc_auc": run_metrics.get("val_roc_auc"), "pr_auc": run_metrics.get("val_pr_auc"), "f1_macro": run_metrics.get("val_f1_macro"), "accuracy": run_metrics.get("val_accuracy")}
        if any(value is not None for value in val_values.values()):
            _metric_cards(val_values, "Validation", "from the registered champion's MLflow run")
        else:
            st.info("This MLflow run does not contain validation metrics.")
        if current_test.get("metrics"):
            _metric_cards(current_test["metrics"], "Held-out test", f"linked to champion v{champion_version}")
        else:
            st.warning("No held-out test evaluation is verifiably linked to this version.")
        with st.expander("Run and registry metadata"):
            st.json({
                "tracking_uri": MLFLOW_TRACKING_URI,
                "model_name": MLFLOW_MODEL_NAME,
                "version": champion_version,
                "aliases": meta.get("aliases", []),
                "run_id": meta.get("run_id"),
                "version_tags": meta.get("tags", {}),
                "run_params": meta.get("params", {}),
                "run_tags": meta.get("run_tags", {}),
            })

    st.markdown("#### Registered version history")
    if DASHBOARD_ONLY:
        st.caption("No registry version history is loaded in dashboard-only mode.")
    elif versions:
        table = _version_table(versions, test_map)
        st.dataframe(table, width="stretch", hide_index=True)
        selected = st.selectbox("Inspect version", [str(row.get("version")) for row in sorted(versions, key=lambda item: int(item.get("version", 0)), reverse=True)])
        selected_row = next((row for row in versions if str(row.get("version")) == str(selected)), {})
        with st.expander(f"Technical details · v{selected}"):
            st.json({key: value for key, value in selected_row.items() if key not in {"metrics", "params", "run_tags"}})
    else:
        st.caption("No registered versions were returned by MLflow.")


elif page == "Monitoring & Drift":
    _page_header("Monitoring & drift", "Reference-to-current distribution checks and retraining trigger status.")
    if not monitoring:
        st.info(
            "No monitoring history file is available in this dashboard-only deployment."
            if DASHBOARD_ONLY else "No monitoring events are recorded yet. Run the monitoring script or pipeline against a batch."
        )
    else:
        latest = max(monitoring, key=lambda item: str(item.get("timestamp", "")))
        report = latest.get("report_data", {})
        drifted = latest.get("drifted_features", []) or []
        threshold = float(report.get("threshold_used", DRIFT_THRESHOLD))
        fraction = latest.get("drift_fraction")
        current_n = report.get("n_current_samples", latest.get("n_rows"))
        reference_n = report.get("n_reference_samples")
        summary_cols = st.columns(4)
        summary_cols[0].metric("Latest batch", latest.get("batch_label", "—"))
        summary_cols[1].metric("Current rows", f"{current_n:,}" if current_n else "—")
        summary_cols[2].metric("Reference rows", f"{reference_n:,}" if reference_n else "—")
        summary_cols[3].metric("Affected features", f"{len(drifted)} / {latest.get('n_features', len(FEATURE_COLUMNS))}")
        status_tone = "drift" if latest.get("dataset_drift") else "healthy"
        _badge("Drift detected" if latest.get("dataset_drift") else "No dataset drift", status_tone)
        trigger = bool(latest.get("should_retrain"))
        st.markdown(f"Retraining trigger: **{'met' if trigger else 'not met'}** · configured affected-feature threshold: **{threshold:.0%}** · latest run at {_format_time(latest.get('timestamp'))}.")
        if latest.get("trigger_reason"):
            st.caption(latest["trigger_reason"])

        scores = report.get("feature_drift_scores", latest.get("feature_drift_scores", {}))
        if scores:
            st.markdown("#### Feature comparison")
            engine = str(latest.get("drift_engine", report.get("engine", "drift detector"))).replace("_", " ").title()
            score_label = f"{engine} score"
            score_df = pd.DataFrame([
                {"Feature": name.replace("_", " ").title(), score_label: float(score), "Drift detected": name in drifted}
                for name, score in scores.items()
            ])
            score_df = score_df.sort_values(score_label, ascending=False)
            st.bar_chart(score_df.set_index("Feature")[[score_label]], height=300)
            st.dataframe(score_df, width="stretch", hide_index=True)
        else:
            st.info("Feature-level scores were not included with this monitoring result.")

        report_path = _report_path(latest.get("report_html"))
        if report_path:
            try:
                with report_path.open("rb") as handle:
                    report_bytes = handle.read()
                st.download_button("Download Evidently HTML report", report_bytes,
                                   file_name=report_path.name, mime="text/html")
            except OSError:
                st.caption("The HTML report path is recorded, but the file is unavailable in this environment.")
        elif latest.get("report_html"):
            st.caption("A report path was recorded, but the HTML file is not mounted here.")

        st.markdown("#### Monitoring history")
        rows = []
        for event in sorted(monitoring, key=lambda item: str(item.get("timestamp", "")), reverse=True):
            rows.append({
                "Time": _format_time(event.get("timestamp")),
                "Batch": event.get("batch_label", "—"),
                "Rows": event.get("n_rows", "—"),
                "Drifted": f"{len(event.get('drifted_features', []) or [])} / {event.get('n_features', '?')}",
                "Drift": "Detected" if event.get("dataset_drift") else "Not detected",
                "Retrain trigger": "Met" if event.get("should_retrain") else "Not met",
                "Engine": event.get("drift_engine", "—"),
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        with st.expander("Raw event details"):
            st.json({key: value for key, value in latest.items() if key != "report_data"})


elif page == "Retraining History":
    _page_header("Retraining history", "Candidate evaluations, promotion decisions and their comparison protocol.")
    if not retraining:
        st.info(
            "No retraining history file is available in this dashboard-only deployment."
            if DASHBOARD_ONLY else "No retraining decisions have been recorded. Monitoring by itself does not start retraining."
        )
    else:
        def _decision(event: dict) -> str:
            return str(event.get("decision", event.get("status", "unknown"))).lower()

        promoted_count = sum(_decision(event) == "promoted" for event in retraining)
        rejected_count = sum(_decision(event) == "rejected" for event in retraining)
        skipped_count = sum(_decision(event) in {"skipped", "error"} for event in retraining)
        counts = st.columns(3)
        counts[0].metric("Evaluations", len(retraining))
        counts[1].metric("Promoted", promoted_count)
        counts[2].metric("Rejected / skipped", rejected_count + skipped_count)

        legacy = [event for event in retraining if event.get("comparison_protocol") != "baseline_validation_v1"]
        if legacy:
            st.warning(f"{len(legacy)} legacy event(s) used a different validation protocol; their scores are shown for history but excluded from comparable trends.")

        st.markdown("#### Decision timeline")
        for event in sorted(retraining, key=lambda item: str(item.get("timestamp", "")), reverse=True):
            decision = _decision(event)
            tone = "accepted" if decision == "promoted" else "rejected" if decision == "rejected" else "warning"
            title_cols = st.columns([3, 1])
            title_cols[0].markdown(f"**{_format_time(event.get('timestamp'))}** · {event.get('model_type', 'candidate').replace('_', ' ').title()}")
            title_cols[1].markdown(f'<span class="mg-badge mg-{tone}">{html.escape("Promotion accepted" if decision == "promoted" else "Promotion rejected" if decision == "rejected" else decision.title())}</span>', unsafe_allow_html=True)
            protocol = event.get("comparison_protocol")
            if protocol != "baseline_validation_v1":
                st.caption("Legacy / incomparable evaluation protocol")
            comp_a, comp_b, comp_c = st.columns([1, 1, 1.5])
            comp_a.metric(f"Champion v{event.get('champion_version', '—')} · validation ROC-AUC", f"{float(event['champion_validation_roc_auc']):.4f}" if event.get("champion_validation_roc_auc") is not None else "—")
            comp_b.metric("Challenger · validation ROC-AUC", f"{float(event['challenger_validation_roc_auc']):.4f}" if event.get("challenger_validation_roc_auc") is not None else "—")
            comp_c.caption(event.get("comparison_reason", event.get("reason", "No decision reason was recorded.")))
            with st.expander("Evaluation and run details"):
                st.write({
                    "Champion version": event.get("champion_version"),
                    "New champion version": event.get("new_champion_version"),
                    "Challenger run ID": event.get("challenger_run_id"),
                    "Comparison protocol": protocol or "legacy / unspecified",
                    "Champion test metrics": event.get("champion_metrics", {}),
                    "Challenger test metrics": event.get("challenger_metrics", {}),
                })
            st.markdown('<div class="mg-divider"></div>', unsafe_allow_html=True)


elif page == "Predictions":
    _page_header(
        "Predictions",
        "Available local prediction records. Live prediction requests require the API service."
        if DASHBOARD_ONLY else "Recent API activity and an interactive request using the model input schema.",
    )
    api_metrics = services["metrics"].get("data", {}) if services["metrics"].get("ok") else {}
    api_health_label, _ = _status_text(services["health"])
    if DASHBOARD_ONLY:
        st.info("Live prediction API is disabled in dashboard-only mode.")
    elif api_health_label != "Healthy":
        st.warning(f"Prediction API is {api_health_label.lower()}. Check the service card on Overview before submitting.")
    total = len(predictions)
    yes_count = int((pd.to_numeric(predictions.get("prediction", pd.Series(dtype=float)), errors="coerce") == 1).sum()) if total else 0
    if not predictions.empty:
        count_cols = st.columns(3)
        count_cols[0].metric("Persisted requests", f"{total:,}")
        count_cols[1].metric("Predicted yes", f"{yes_count:,}")
        count_cols[2].metric("Predicted no", f"{total - yes_count:,}")
    elif DASHBOARD_ONLY:
        st.info("No prediction log is available in this deployment. Prediction counts are not inferred.")
    if not predictions.empty:
        st.markdown("#### Probability distribution")
        if "probability_yes" in predictions:
            probabilities = pd.to_numeric(predictions["probability_yes"], errors="coerce").dropna()
            if len(probabilities) > 1 and probabilities.nunique() > 1:
                histogram = pd.cut(probabilities, bins=10, include_lowest=True).value_counts(sort=False)
                histogram_frame = pd.DataFrame(
                    {"Requests": histogram.to_numpy()},
                    index=[f"{interval.left:.2f}–{interval.right:.2f}" for interval in histogram.index],
                )
                st.bar_chart(histogram_frame, height=220)
            else:
                st.caption(f"Latest probability of yes: {probabilities.iloc[0]:.3f}" if len(probabilities) else "No probability values recorded.")
        table_cols = [column for column in ("timestamp", "model_version", "label", "probability_yes") if column in predictions]
        display = predictions[table_cols].copy()
        if "timestamp" in display:
            display["timestamp"] = display["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        st.dataframe(display.tail(50).iloc[::-1], width="stretch", hide_index=True)
    else:
        st.caption(
            "No local prediction log records are available."
            if DASHBOARD_ONLY else "No prediction requests have been logged yet."
        )

    st.markdown("#### Run a prediction")
    if DASHBOARD_ONLY:
        st.caption("Unavailable: live predictions require the Prediction API, which is not running in dashboard-only mode.")
    categories = _categorical_values()
    with st.form("modelguard_prediction_form", clear_on_submit=False):
        values: dict[str, object] = {}
        numeric_cols = st.columns(3)
        numeric_defaults = {"age": 35, "balance": 0, "day": 15, "campaign": 1, "pdays": -1, "previous": 0}
        for index, feature in enumerate(NUMERICAL_FEATURES):
            with numeric_cols[index % len(numeric_cols)]:
                values[feature] = st.number_input(feature.replace("_", " ").title(), value=numeric_defaults.get(feature, 0), step=1)
        categorical_cols = st.columns(3)
        for index, feature in enumerate(CATEGORICAL_FEATURES):
            with categorical_cols[index % len(categorical_cols)]:
                choices = categories.get(feature, [])
                if choices:
                    values[feature] = st.selectbox(feature.replace("_", " ").title(), choices, key=f"prediction_{feature}")
                else:
                    values[feature] = st.text_input(feature.replace("_", " ").title(), key=f"prediction_{feature}")
        submitted = st.form_submit_button("Predict", type="primary", disabled=DASHBOARD_ONLY)
    if submitted and not DASHBOARD_ONLY:
        missing = [feature for feature in FEATURE_COLUMNS if feature not in values or values[feature] is None]
        if missing:
            st.error(f"Missing input fields: {', '.join(missing)}")
        else:
            result = _json_request(f"{API_URL}/predict", timeout=15, body=values)
            if result.get("ok"):
                response = result["data"]
                st.success(f"Prediction: {str(response.get('label', 'unknown')).upper()} · model v{response.get('model_version', '—')}")
                probability = response.get("probability_yes")
                if probability is not None:
                    st.metric("Probability of yes", f"{float(probability):.1%}")
            else:
                detail = result.get("data", {}).get("error", "Prediction request failed")
                st.error(f"API rejected the request ({result.get('status') or 'unavailable'}): {detail}")
