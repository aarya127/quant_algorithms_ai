#!/usr/bin/env bash
# daily_retrain.sh — full supervised ML pipeline retrain + prediction log
#
# Runs supervised.py (retrain all models) then logs the latest prediction.
# Designed to be called by cron once per day after market close.
#
# Usage:
#   ./scripts/daily_predict.sh [TICKER]          # defaults to NVDA
#
# Register cron (5 PM ET = 22:00 UTC, Mon–Fri):
#   crontab -e
#   0 22 * * 1-5 /Users/aaryas127/quant_algorithms_ai/scripts/daily_predict.sh NVDA \
#       >> /tmp/daily_retrain.log 2>&1

set -euo pipefail

TICKER="${1:-NVDA}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON=/opt/anaconda3/bin/python
SUPERVISED="$PROJECT_DIR/algorithms/machine_learning_algorithms/supervised"

echo "========================================"
echo " Daily retrain — $(date '+%Y-%m-%d %H:%M:%S')"
echo " Ticker : $TICKER"
echo "========================================"

# Step 1: full pipeline retrain
echo "[1/2] Running supervised pipeline (USE_TUNING=1)..."
cd "$SUPERVISED"
USE_TUNING=1 "$PYTHON" supervised.py "$TICKER"
echo "      Pipeline complete."

# Step 2: log latest prediction
echo "[2/2] Logging latest prediction..."
cd "$PROJECT_DIR"
"$PYTHON" - <<PYEOF
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), "backend"))
ticker = "$TICKER"

from predictor import predict_latest, check_drift

result = predict_latest(ticker)
print(f"  Date       : {result['date']}")
print(f"  Signal     : {result['signal'].upper()}")
print(f"  Confidence : {result['confidence'].upper()}")
print(f"  Anomaly    : {'YES' if result['anomaly_flag'] else 'No'}")
for k, v in result.get('predictions', {}).items():
    print(f"  {k:<16}: {v:+.4f}" if isinstance(v, float) else f"  {k:<16}: {v}")

drift = check_drift(ticker)
print(f"  Drift      : {drift['overall_status'].upper()} ({drift['n_drift_flags']}/{drift['n_features_checked']} features)")
PYEOF

echo "Done — $(date '+%Y-%m-%d %H:%M:%S')"

