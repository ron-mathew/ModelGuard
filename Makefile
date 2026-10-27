.PHONY: help setup train api dashboard test clean simulate-drift monitor retrain demo

PYTHON := python
SRC_DIR := src
PYTHONPATH_EXPORT := set PYTHONPATH=$(SRC_DIR) &&

help:
	@echo "ModelGuard — Available Commands"
	@echo "================================"
	@echo "  make setup          Install Python dependencies"
	@echo "  make prepare-data   Validate dataset and generate reference data"
	@echo "  make train          Train baseline models (LR + RF)"
	@echo "  make mlflow-ui      Start MLflow tracking UI (port 5001)"
	@echo "  make api            Start prediction API (port 5000)"
	@echo "  make dashboard      Start Streamlit dashboard (port 8501)"
	@echo "  make test           Run automated test suite"
	@echo "  make simulate-drift Generate drifted datasets"
	@echo "  make monitor        Run drift monitoring on combined scenario"
	@echo "  make retrain        Run conditional retraining"
	@echo "  make demo           Run all demo scenarios"
	@echo "  make docker-up      Start all services with Docker Compose"
	@echo "  make docker-down    Stop all Docker services"
	@echo "  make clean          Remove generated artifacts"

setup:
	pip install -r requirements.txt

prepare-data:
	set PYTHONPATH=src && $(PYTHON) scripts/prepare_data.py

train:
	set PYTHONPATH=src && $(PYTHON) -m modelguard.train --model both

mlflow-ui:
	mlflow server \
		--backend-store-uri sqlite:///mlruns/mlflow.db \
		--default-artifact-root ./mlruns/artifacts \
		--host 0.0.0.0 \
		--port 5001

api:
	set PYTHONPATH=src && $(PYTHON) src/modelguard/api.py

dashboard:
	set PYTHONPATH=src && streamlit run dashboard/app.py --server.port 8501

test:
	set PYTHONPATH=src && $(PYTHON) -m pytest tests/ -v --tb=short

simulate-drift:
	set PYTHONPATH=src && $(PYTHON) scripts/simulate_drift.py --scenario all

monitor:
	set PYTHONPATH=src && $(PYTHON) scripts/run_monitoring.py \
		--batch data/simulated/drifted_combined.csv \
		--label demo_combined \
		--force

retrain:
	set PYTHONPATH=src && $(PYTHON) scripts/run_retraining.py --force

demo:
	set PYTHONPATH=src && $(PYTHON) scripts/demo.py --scenario all

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down

clean:
	del /s /q artifacts\*.json artifacts\*.txt artifacts\*.csv 2>nul || true
	del /s /q reports\drift\*.html reports\drift\*.json 2>nul || true
	del /s /q reports\monitoring\*.jsonl 2>nul || true
	del /s /q data\simulated\*.csv 2>nul || true
