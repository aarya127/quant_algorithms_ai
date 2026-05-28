"""
normalize.py — Normalization step for the cleaned feature matrix.

Usage:
    python normalize.py [SYMBOL]

What it does:
    1. Defines all targets (regression + classification) and appends them
    2. Splits features from targets — scalers are fit on features only
    3. Applies StandardScaler column-wise to all numeric features
       (targets are left in their natural units — do NOT scale targets)
    4. Saves:
         <SYMBOL>_features_normalized.csv   — scaled features + unscaled targets
         <SYMBOL>_scaler.pkl                — fitted StandardScaler (for inverse transforms)
         <SYMBOL>_targets.csv               — targets only (unscaled), for easy loading

Targets defined:
    Regression:
        target_1d       next-day log return
        target_5d       5-day forward cumulative log return
        target_vol_5d   std of next 5 daily log returns (realized vol forecast)

    Classification:
        target_dir_1d   direction: 1 (up >0.5%), 0 (flat), -1 (down <-0.5%)
        target_large_move  1 if |next-day return| > 2*rolling_std, else 0
        target_regime   volatility regime: 0=low, 1=mid, 2=high
                        based on 20d realized vol percentile rank
"""

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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

# 1. Define targets
log_ret = df["log_return"]

# Regression
df["target_1d"]     = log_ret.shift(-1)
df["target_5d"]     = log_ret.shift(-1).rolling(5).sum().shift(-4)   # sum of next 5 days
df["target_vol_5d"] = log_ret.shift(-1).rolling(5).std().shift(-4)   # vol of next 5 days

# Classification — direction (3-class)
FLAT_THRESHOLD = 0.005   # ±0.5% dead zone → labelled flat
df["target_dir_1d"] = np.where(
    df["target_1d"] >  FLAT_THRESHOLD,  1,
    np.where(df["target_1d"] < -FLAT_THRESHOLD, -1, 0)
)

# Classification — large move flag
rolling_std = log_ret.rolling(20).std()
df["target_large_move"] = (df["target_1d"].abs() > 2 * rolling_std.shift(-1)).astype(int)

# Classification — volatility regime (tercile of 20d realized vol)
vol_rank = df["realized_vol_20d"].rank(pct=True)
df["target_regime"] = pd.cut(vol_rank, bins=[0, 1/3, 2/3, 1.0],
                              labels=[0, 1, 2]).astype(float)

TARGET_COLS = [
    "target_1d", "target_5d", "target_vol_5d",
    "target_dir_1d", "target_large_move", "target_regime",
]

print("[1] Targets defined:")
for tc in TARGET_COLS:
    non_null = df[tc].notna().sum()
    if tc in ("target_dir_1d", "target_large_move", "target_regime"):
        vc = df[tc].value_counts().sort_index().to_dict()
        print(f"      {tc:<22}  {non_null} rows  classes={vc}")
    else:
        print(f"      {tc:<22}  {non_null} rows  "
              f"mean={df[tc].mean():.5f}  std={df[tc].std():.5f}")

# 2. Separate features from targets
# Non-numeric / metadata columns — excluded from scaling
META_COLS = ["symbol", "fund_quarter_end", "opt_as_of"]
meta_present = [c for c in META_COLS if c in df.columns]

feature_cols = [
    c for c in df.columns
    if c not in TARGET_COLS + meta_present
    and pd.api.types.is_numeric_dtype(df[c])
]

print(f"\n[2] Feature columns to scale: {len(feature_cols)}")
print(f"    Targets kept unscaled    : {len(TARGET_COLS)}")
if meta_present:
    print(f"    Meta cols excluded       : {meta_present}")

# 3. Fit StandardScaler on feature columns
# Drop rows where ALL targets are NaN (the last few rows from forward shifts)
# but keep the scaler fit on all available feature data
scaler = StandardScaler()
scaler.fit(df[feature_cols].fillna(df[feature_cols].median()))   # median fill for fit only

df_scaled = df.copy()
df_scaled[feature_cols] = scaler.transform(
    df[feature_cols].fillna(df[feature_cols].median())
)

print("\n[3] StandardScaler applied:")
print(f"    Feature means (post-scale) ≈ {df_scaled[feature_cols].mean().mean():.6f}  (expect ~0)")
print(f"    Feature stds  (post-scale) ≈ {df_scaled[feature_cols].std().mean():.6f}   (expect ~1)")

# 4. Save outputs
out_norm    = HERE / f"{SYMBOL}_features_normalized.csv"
out_scaler  = HERE / f"{SYMBOL}_scaler.pkl"
out_targets = HERE / f"{SYMBOL}_targets.csv"

# Normalized features + targets
df_scaled.to_csv(out_norm)
print(f"\n[4] Saved:")
print(f"    {out_norm}  ({df_scaled.shape[0]} rows × {df_scaled.shape[1]} cols)")

# Scaler object
with open(out_scaler, "wb") as f:
    pickle.dump({"scaler": scaler, "feature_cols": feature_cols}, f)
print(f"    {out_scaler}")

# Targets only (unscaled) — easy loading for modeling scripts
targets_df = df[TARGET_COLS].copy()
targets_df.to_csv(out_targets)
print(f"    {out_targets}  ({targets_df.shape[0]} rows × {targets_df.shape[1]} cols)")

# 5. Class balance report
print("\n[5] Classification target balance:")
for tc in ("target_dir_1d", "target_large_move", "target_regime"):
    vc = df[tc].value_counts(normalize=True).sort_index() * 100
    parts = "  ".join(f"{int(k) if k == int(k) else k}: {v:.1f}%" for k, v in vc.items())
    print(f"    {tc:<22}  {parts}")

print(f"\n=== normalize complete ===")
