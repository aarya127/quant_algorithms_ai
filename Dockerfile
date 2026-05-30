FROM python:3.11-slim

LABEL org.opencontainers.image.title="Invest.ai Backend" \
      org.opencontainers.image.description="Institutional quant terminal + ML inference server" \
      org.opencontainers.image.version="1.0.0"

WORKDIR /app

# ── System deps needed by LightGBM (OpenMP) ──────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps (layer-cached) ────────────────────────────────────────────────
COPY backend/requirements.txt .

# CPU-only PyTorch (~180 MB vs the default 2.5 GB CUDA wheel)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# All remaining backend + ML inference dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ── FinBERT weights (bake in so first request doesn't block on 440 MB download)
RUN python3 -c "\
from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
AutoTokenizer.from_pretrained('ProsusAI/finbert'); \
AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert'); \
print('FinBERT pre-downloaded')"

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

WORKDIR /app/backend

# Ensure runtime data directories exist (overridden by PVC mounts in Kubernetes)
RUN mkdir -p \
    /app/algorithms/machine_learning_algorithms/data_pipelines \
    /app/algorithms/machine_learning_algorithms/supervised/model_registry \
    /app/algorithms/machine_learning_algorithms/supervised/output/monitoring \
    /app/mlruns

EXPOSE 8080

# Smoke-test: verify imports cleanly before shipping
RUN python3 -c "import flask, joblib, sklearn, xgboost, lightgbm, mlflow; print('imports OK')"

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python3 -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-8080}/health')"

COPY backend/entrypoint.sh .
RUN chmod +x entrypoint.sh

CMD ["./entrypoint.sh"]
