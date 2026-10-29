# ModelGuard

ModelGuard is a reproducible drift-aware MLOps demonstration for the UCI Bank
Marketing dataset. It trains a Logistic Regression and Random Forest pipeline,
tracks models in MLflow, serves the registered champion through Flask, monitors
input drift with Evidently, and evaluates retraining candidates before
promotion. The `duration` field is excluded because it is known only after a
call and would leak information unavailable at prediction time.

## Requirements and setup

Use Python 3.11 or newer. The checked-in dependencies were verified with
Python 3.13.5 in the development environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

Place the UCI `bank-full.csv` file at `data/raw/bank-full.csv`. It must be a
semicolon-delimited CSV with target column `y`.

## Train and serve

Train and register both baseline models. Selection uses validation ROC-AUC;
the final test split is reported after model selection.

```powershell
$env:PYTHONPATH = "src"
python -m modelguard.train --model both
```

The best model is registered as `bank-marketing-classifier` with the `champion`
alias. MLflow logs scikit-learn models with skops serialization and a narrow
trusted type list for `numpy.dtype`.

Start the prediction API in one terminal and the dashboard in another:

```powershell
$env:PYTHONPATH = "src"
python -m modelguard.api
```

```powershell
$env:PYTHONPATH = "src"
streamlit run dashboard/app.py --server.port 8501
```

The API listens on port 5000. Its `/health`, `/model-info`, and `/predict`
endpoints expose the active model status, registry metadata, and predictions.
Predictions are written to `artifacts/predictions.jsonl`.

## Monitoring and retraining

Training writes a monitoring reference set to `data/reference/reference.csv`.
It also writes a fixed validation set to `artifacts/validation_set/`; promotion
comparisons use this same validation set for both champion and challenger.
Rerun baseline training if this file is missing. The API checks MLflow for a
new champion at most every 5 seconds by default and swaps to it only after the
exact registered version loads successfully; no process restart is required.
Monitor a batch with at least 500 rows:

```powershell
$env:PYTHONPATH = "src"
python scripts/run_monitoring.py --batch data/simulated/drifted_combined.csv --label drift-check
```

Evidently JSON and HTML reports are stored in `reports/drift/`; monitoring
events are appended to `reports/monitoring/monitoring_log.jsonl`. A drift
result recommends retraining. To execute it:

```powershell
$env:PYTHONPATH = "src"
python scripts/run_retraining.py --batch data/simulated/drifted_combined.csv --model random_forest
```

When a batch is supplied without `--confirmed-drift`, the retraining command
runs monitoring first and proceeds only when that batch triggers drift.
Retraining requires labeled input to augment the reference data. Promotion is
decided by validation ROC-AUC on the challenger cycle's validation split; the
fixed final test split is used only for reporting after that decision. Each
decision is recorded in `reports/monitoring/retraining_log.jsonl`. A rejected
challenger leaves the champion alias unchanged.
Historical retraining log entries without `comparison_protocol` used the older
re-split-reference comparison and should not be compared with the baseline or
new fixed-validation scores; the dashboard flags those entries.

Generate the deterministic drift scenarios and run the full interactive
demonstration with:

```powershell
$env:PYTHONPATH = "src"
python scripts/simulate_drift.py --scenario all
python scripts/demo.py --scenario all
```

The integrated accepted/rejected outcomes depend on the trained data and
acceptance threshold. For deterministic policy-only examples (synthetic metric
inputs; no models trained or registry changes made), run:

```powershell
$env:PYTHONPATH = "src"
python scripts/demo_decisions.py
```

## Tests and infrastructure

```powershell
$env:PYTHONPATH = "src"
python -m pytest tests/ -v
```

Docker Compose defines MLflow, API, dashboard, and an opt-in trainer. Populate
the shared registry first, then start the serving services:

```powershell
docker compose up --build -d mlflow
docker compose --profile train run --build --rm trainer
docker compose up --build -d api dashboard
```

Persistent MLflow data, model artifacts, API logs, and reports use named
volumes. The API image expects a champion already registered in that shared
MLflow store.

### Render dashboard-only deployment

`Dockerfile.render` runs only Streamlit on port 8501 to fit the free-tier memory
limit. Set `MODEL_GUARD_DASHBOARD_ONLY=true` for this deployment. In this mode
the dashboard does not query the Prediction API or MLflow registry and does
not submit predictions. It can display genuine prediction, monitoring, and
retraining records only when their files are present in the container; otherwise
it shows an unavailable or empty state. Do not set the API or MLflow service
URLs to localhost in this mode. The normal Docker Compose stack leaves this
setting unset and continues to use its configured `api` and `mlflow` endpoints.

Kubernetes examples are in `k8s/`. They require a working cluster, persistent
volume support, and a locally available `modelguard:latest` image. The manifests
have not been deployed in this environment.
