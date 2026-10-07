"""
normalize.py — Normalization step for the cleaned feature matrix.

Usage:
    python normalize.py [SYMBOL]

What it does:
    1. Defines all targets (regression + classification) and appends them
    2. Saves:
         <SYMBOL>_features_normalized.csv   — raw features + targets
         <SYMBOL>_targets.csv               — targets only, for easy loading

Features are NOT scaled or imputed here. A scaler fit on the whole file would
leak the holdout's statistics into training, and would tie every saved model to
the run it came from. The unsupervised step and each supervised fold impute and
scale from their own training rows, and each saved model carries its own
serving scaler. (The file name is kept for the downstream contract.)

Targets defined:
    Regression:
        target_1d       next-day log return
        target_5d       5-day forward cumulative log return
        target_vol_5d   std of next 5 daily log returns (realized vol forecast)

    Classification:
        target_dir_1d   direction: 1 (up >0.5%), 0 (flat), -1 (down <-0.5%)
        target_large_move  1 if |next-day return| > 2*rolling_std, else 0
        target_regime   forward volatility regime: 0=low, 1=mid, 2=high —
                        tercile of target_vol_5d against the expanding
                        history of past values (no full-sample cut points)
"""

import sys
import warnings
from pathlib import Path

import pandas as pd

from targets import TARGET_COLS, add_targets

warnings.filterwarnings("ignore")

SYMBOL = sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA"
HERE   = Path(__file__).parent

src = HERE / f"{SYMBOL}_features_clean.csv"
if not src.exists():
    print(f"ERROR: {src} not found — run clean.py first.")
    sys.exit(1)

df = pd.read_csv(src, index_col=0, parse_dates=True)
df.index = pd.to_datetime(df.index).tz_localize(None)
df.index.name = "Date"

print(f"=== normalize: {SYMBOL} ===")
print(f"Input : {df.shape[0]} rows × {df.shape[1]} cols\n")

# 1. Define targets (definitions live in targets.py)
df = add_targets(df)

print("[1] Targets defined:")
for tc in TARGET_COLS:
    non_null = df[tc].notna().sum()
    if tc in ("target_dir_1d", "target_large_move", "target_regime"):
        vc = df[tc].value_counts().sort_index().to_dict()
        print(f"      {tc:<22}  {non_null} rows  classes={vc}")
    else:
        print(f"      {tc:<22}  {non_null} rows  "
              f"mean={df[tc].mean():.5f}  std={df[tc].std():.5f}")

# 2. Save outputs
out_norm    = HERE / f"{SYMBOL}_features_normalized.csv"
out_targets = HERE / f"{SYMBOL}_targets.csv"

# Raw features + targets
df.to_csv(out_norm)
print(f"\n[2] Saved:")
print(f"    {out_norm}  ({df.shape[0]} rows × {df.shape[1]} cols)")

# Targets only (unscaled) — easy loading for modeling scripts
targets_df = df[TARGET_COLS].copy()
targets_df.to_csv(out_targets)
print(f"    {out_targets}  ({targets_df.shape[0]} rows × {targets_df.shape[1]} cols)")

# 3. Class balance report
print("\n[3] Classification target balance:")
for tc in ("target_dir_1d", "target_large_move", "target_regime"):
    vc = df[tc].value_counts(normalize=True).sort_index() * 100
    parts = "  ".join(f"{int(k) if k == int(k) else k}: {v:.1f}%" for k, v in vc.items())
    print(f"    {tc:<22}  {parts}")

print(f"\n=== normalize complete ===")
