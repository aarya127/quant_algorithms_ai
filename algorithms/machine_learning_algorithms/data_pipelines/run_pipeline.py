"""
run_pipeline.py — Extract, transform, and save a feature matrix to CSV.

Usage:
    python run_pipeline.py [SYMBOL] [PERIOD]

    SYMBOL : ticker symbol (default: NVDA)
    PERIOD : yfinance period string — 3mo, 6mo, 1y, 2y (default: 3mo)

Output:
    <SYMBOL>_features.csv saved in this directory.
    Load it afterwards with load_features.py.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Allow running from any cwd
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from algorithms.machine_learning_algorithms.data_pipelines import DataTransformer  # noqa: E402

SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
PERIOD = sys.argv[2] if len(sys.argv) > 2 else "3mo"

print(f"=== pipeline: {SYMBOL}  period={PERIOD} ===\n")

dt = DataTransformer()
df = dt.build_feature_matrix(SYMBOL, period=PERIOD)

out = Path(__file__).parent / f"{SYMBOL}_features.csv"
df.to_csv(out)
print(f"Saved → {out}  ({df.shape[0]} rows, {df.shape[1]} cols)")
print(f"Load with: python load_features.py {SYMBOL}")
