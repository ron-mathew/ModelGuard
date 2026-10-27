# ModelGuard Docker Image
FROM python:3.11-slim

LABEL maintainer="ModelGuard"
LABEL description="Drift-Aware MLOps Pipeline"

# Set working directory
WORKDIR /app

# curl is used by the container health check.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    API_HOST=0.0.0.0 \
    API_PORT=5000 \
    MLFLOW_TRACKING_URI=sqlite:////app/mlruns/mlflow.db \
    MLFLOW_ARTIFACT_ROOT=/app/mlruns/artifacts

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY data/ ./data/
COPY scripts/ ./scripts/
COPY dashboard/ ./dashboard/

# Create necessary directories
RUN mkdir -p /app/mlruns/artifacts \
             /app/artifacts \
             /app/reports/drift \
             /app/reports/monitoring \
             /app/data/simulated \
             /app/data/reference

# Expose ports
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Default command: run the prediction API
CMD ["python", "-m", "modelguard.api"]
