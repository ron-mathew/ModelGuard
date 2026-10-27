"""Real local-MLflow promotion test with controlled synthetic data.

This exercises real model serialization, model registration, alias promotion,
the retraining decision, and API refresh. The synthetic fixture only proves
that this code path moves versions safely; its scores are not model-quality
evidence for the Bank Marketing dataset.
"""

from __future__ import annotations

import json
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

from modelguard import api, registry, retrain
from modelguard.config import FEATURE_COLUMNS, MLFLOW_MODEL_PIP_REQUIREMENTS
from modelguard.preprocessing import build_full_pipeline


def _controlled_data(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(214)
    data: dict[str, object] = {
        "age": rng.integers(18, 76, size=n),
        "balance": rng.integers(-500, 8000, size=n),
        "day": rng.integers(1, 29, size=n),
        "campaign": rng.integers(1, 6, size=n),
        "pdays": rng.choice([-1, *range(1, 100)], size=n),
        "previous": rng.integers(0, 5, size=n),
    }
    categories = {
        "job": ["admin.", "technician", "management"],
        "marital": ["married", "single", "divorced"],
        "education": ["primary", "secondary", "tertiary"],
        "default": ["yes", "no"],
        "housing": ["yes", "no"],
        "loan": ["yes", "no"],
        "contact": ["cellular", "telephone", "unknown"],
        "month": ["jan", "may", "oct"],
        "poutcome": ["unknown", "success", "failure"],
    }
    for name, values in categories.items():
        data[name] = rng.choice(values, size=n)
    frame = pd.DataFrame(data, columns=FEATURE_COLUMNS)
    # A deterministic, learnable target makes the controlled challenger
    # outperform a constant DummyClassifier on validation data.
    frame["y"] = (frame["age"] >= 46).astype("int64")
    return frame


def _log_model(model, X: pd.DataFrame, y: pd.Series, run_name: str) -> str:
    with mlflow.start_run(run_name=run_name) as run:
        model.fit(X, y)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=X.head(1),
            serialization_format="skops",
            skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
            pip_requirements=MLFLOW_MODEL_PIP_REQUIREMENTS,
        )
        return run.info.run_id


def test_real_retraining_promotion_refreshes_running_api(tmp_path, monkeypatch):
    tracking_uri = f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}"
    original_tracking_uri = mlflow.get_tracking_uri()
    model_name = "modelguard-promotion-integration"
    experiment_name = "modelguard-promotion-integration"
    original_cache = {
        key: api._model_cache[key]
        for key in ("pipeline", "version", "loaded_at", "last_checked_monotonic")
    }

    monkeypatch.setattr(registry, "MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setattr(registry, "MLFLOW_MODEL_NAME", model_name)
    monkeypatch.setattr(retrain, "MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setattr(retrain, "MLFLOW_MODEL_NAME", model_name)
    monkeypatch.setattr(retrain, "MLFLOW_EXPERIMENT_NAME", experiment_name)
    monkeypatch.setattr(retrain, "RANDOM_FOREST_PARAMS", {
        "n_estimators": 20,
        "max_depth": 5,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": 214,
        "n_jobs": 1,
    })
    monkeypatch.setattr(retrain, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(retrain, "RETRAIN_LOG_FILE", tmp_path / "retraining.jsonl")
    monkeypatch.setattr(
        retrain, "RETRAINED_BATCHES_FILE", tmp_path / "retrained_batches.json"
    )
    monkeypatch.setattr(retrain, "REFERENCE_DATA_FILE", tmp_path / "reference.csv")
    monkeypatch.setattr(api, "MLFLOW_MODEL_NAME", model_name)
    monkeypatch.setattr(api, "MODEL_REFRESH_INTERVAL_SECONDS", 0.0)

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.create_experiment(
            experiment_name,
            artifact_location=(tmp_path / "artifacts").as_uri(),
        )
        mlflow.set_experiment(experiment_name)

        frame = _controlled_data()
        X = frame[FEATURE_COLUMNS]
        y = frame["y"]
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=214,
            stratify=y,
        )
        X_validation, X_test, y_validation, y_test = train_test_split(
            X_holdout,
            y_holdout,
            test_size=0.50,
            random_state=215,
            stratify=y_holdout,
        )
        pd.concat([X_train, y_train.rename("y")], axis=1).to_csv(
            retrain.REFERENCE_DATA_FILE, index=False
        )
        # The live API starts on a real, registered but intentionally weak
        # champion (constant prediction); this is controlled test data only.
        champion = build_full_pipeline(DummyClassifier(strategy="most_frequent"))
        champion_run_id = _log_model(
            champion, X_train, y_train, "controlled_champion"
        )
        champion_version = registry.register_model(
            champion_run_id, model_name=model_name
        )
        registry.set_alias(model_name, champion_version, "champion")
        assert str(champion_version) == "1"

        test_dir = tmp_path / "artifacts" / "test_set"
        test_dir.mkdir(parents=True)
        X_test.to_csv(test_dir / "X_test.csv", index=False)
        y_test.to_csv(test_dir / "y_test.csv", index=False)
        validation_dir = tmp_path / "artifacts" / "validation_set"
        validation_dir.mkdir()
        X_validation.to_csv(validation_dir / "X_validation.csv", index=False)
        y_validation.to_csv(validation_dir / "y_validation.csv", index=False)

        api._model_cache.update({
            "pipeline": None,
            "version": None,
            "loaded_at": None,
            "last_checked_monotonic": 0.0,
        })
        client = api.create_app(preload_model=True).test_client()
        assert client.get("/model-info").get_json()["champion_version"] == "1"

        result = retrain.run_retraining(
            drift_already_confirmed=True,
            model_type="random_forest",
            force=True,
        )

        assert result["decision"] == "promoted"
        assert str(result["champion_version"]) == "1"
        assert str(registry.get_champion_version()) == "2"
        promoted = client.get("/model-info").get_json()
        assert promoted["champion_version"] == "2"

        record = X_validation.iloc[0].to_dict()
        response = client.post("/predict", json=record)
        assert response.status_code == 200
        assert response.get_json()["model_version"] == "2"

        with (tmp_path / "retraining.jsonl").open(encoding="utf-8") as fh:
            audit = json.loads(fh.readlines()[-1])
        assert audit["decision"] == "promoted"
        assert audit["challenger_validation_roc_auc"] > audit[
            "champion_validation_roc_auc"
        ]
    finally:
        mlflow.set_tracking_uri(original_tracking_uri)
        api._model_cache.update(original_cache)
