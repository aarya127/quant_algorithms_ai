#!/usr/bin/env bash
# daily_pipeline.sh — end-to-end ML pipeline for a single ticker.
#
# Usage:
#   ./scripts/daily_pipeline.sh [SYMBOL]      # default: NVDA
#
# Cron (runs Mon–Fri at 18:00 local time, after market close):
#   0 18 * * 1-5 /Users/aaryas127/quant_algorithms_ai/scripts/daily_pipeline.sh >> /Users/aaryas127/quant_algorithms_ai/logs/cron.log 2>&1
#
# Steps:
#   1. run_pipeline  — incremental feature extraction (yfinance + sentiment)
#   2. clean         — drop dead columns, forward-fill, median-fill
#   3. normalize     — StandardScaler features, define targets
#   4. unsupervised  — K-Means regimes + Isolation Forest anomalies
#   5. supervised    — walk-forward CV + holdout, registry promotion gate
#   6. reload        — signal Flask backend to reload model artifacts

set -euo pipefail

REPO=/Users/aaryas127/quant_algorithms_ai
PYTHON=/opt/anaconda3/bin/python
SYMBOL=${1:-NVDA}
PIPES=$REPO/algorithms/machine_learning_algorithms/data_pipelines

LOG_DIR=$REPO/logs
mkdir -p "$LOG_DIR"
LOG=$LOG_DIR/pipeline_${SYMBOL}_$(date +%Y%m%d).log
exec >> "$LOG" 2>&1

echo "=== $(date)  symbol=$SYMBOL ==="
cd "$REPO"

echo "[1/5] run_pipeline (incremental extract)"
"$PYTHON" "$PIPES/run_pipeline.py" "$SYMBOL"

echo "[2/5] clean"
"$PYTHON" "$PIPES/clean.py" "$SYMBOL"

echo "[3/5] normalize"
"$PYTHON" "$PIPES/normalize.py" "$SYMBOL"

echo "[4/5] unsupervised (regimes + anomalies)"
"$PYTHON" "$REPO/algorithms/machine_learning_algorithms/unsupervised/unsupervised.py" "$SYMBOL"

echo "[5/5] supervised (walk-forward CV, no tuning)"
USE_TUNING=0 "$PYTHON" \
    "$REPO/algorithms/machine_learning_algorithms/supervised/supervised.py" "$SYMBOL"

echo "[6/6] reload registry (best-effort)"
curl -s --max-time 5 -X POST http://localhost:5001/api/ml/reload-registry >/dev/null || true

echo "=== Done $(date) ==="
