"""
load_features.py — Load a saved feature-matrix CSV and show a summary.

Usage:
    python load_features.py [SYMBOL]

    SYMBOL : ticker symbol (default: NVDA)
             Looks for <SYMBOL>_features.csv in this directory.

Returns:
    df  — pandas DataFrame (date-indexed, all columns as typed)
"""

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent

SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
csv_path = HERE / f"{SYMBOL}_features.csv"

if not csv_path.exists():
    print(f"ERROR: {csv_path} not found. Run run_pipeline.py {SYMBOL} first.")
    sys.exit(1)

df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)

print(f"Loaded {csv_path.name}  →  {df.shape[0]} rows × {df.shape[1]} cols")
print(f"Date range: {df.index[0].date()} → {df.index[-1].date()}\n")

print("--- First 5 rows (key columns) ---")
preview_cols = [
    "Close", "Volume", "RSI_14", "MACD", "bb_pct",
    "SMA_20", "SMA_50", "news_count", "opt_atm_iv",
    "opt_put_call_oi", "earn_days_to_next",
]
available = [c for c in preview_cols if c in df.columns]
print(df[available].head().to_string())

print("\n--- Full column list (with null %) ---")
for i, col in enumerate(df.columns):
    null_pct = df[col].isna().mean() * 100
    print(f"  {i+1:3d}. {col:<40s}  null={null_pct:5.1f}%")

print("\n--- dtypes ---")
print(df.dtypes.value_counts().to_string())

# df is available for import when this module is used programmatically:
#   from load_features import df
