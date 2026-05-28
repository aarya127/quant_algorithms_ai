"""
clean.py — Data cleaning step for a feature matrix CSV.

Usage:
    python clean.py [SYMBOL]

Strategy:
    1. Drop columns with >50% nulls (dead data sources on free tier)
    2. ffill → bfill for time-series columns (carries last known value forward)
    3. Median fill as backstop for any remaining nulls
    4. Report every decision made

Output:
    <SYMBOL>_features_clean.csv  (same folder)
"""

import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

SYMBOL = sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA"
HERE = Path(__file__).parent

src = HERE / f"{SYMBOL}_features.csv"
dst = HERE / f"{SYMBOL}_features_clean.csv"

if not src.exists():
    print(f"ERROR: {src} not found — run run_pipeline.py first.")
    sys.exit(1)

df = pd.read_csv(src, index_col=0, parse_dates=True)
df.index = pd.to_datetime(df.index).tz_localize(None)
df.index.name = "Date"

print(f"=== cleaning: {SYMBOL} ===")
print(f"Input : {df.shape[0]} rows × {df.shape[1]} cols\n")

original_cols = list(df.columns)
drop_log   = []
ffill_log  = []
median_log = []

# 1. Drop columns above null threshold
NULL_DROP_THRESHOLD = 0.50   # drop if >50% of rows are null

null_pct = df.isnull().mean()
cols_to_drop = null_pct[null_pct > NULL_DROP_THRESHOLD].index.tolist()
for col in cols_to_drop:
    drop_log.append((col, f"{null_pct[col]:.0%} null"))

df.drop(columns=cols_to_drop, inplace=True)
print(f"[1] Dropped {len(cols_to_drop)} column(s) with >{NULL_DROP_THRESHOLD:.0%} nulls:")
for col, reason in drop_log:
    print(f"      ✗  {col:<30}  ({reason})")

# 2. ffill → bfill for time-series / forward-carried columns
# These are columns where "last known value" is the correct imputation:
#   - fundamentals update quarterly → carry forward between earnings
#   - sentiment: carry the most recent sentiment until a new article appears
#   - options / insider: snapshots that don't change daily
FFILL_COLS = [
    # Fundamentals (quarterly release, valid until next quarter)
    "fund_cash", "fund_current_ratio", "fund_debt_to_equity", "fund_eps_ttm",
    "fund_fcf_ttm", "fund_gross_margin", "fund_gross_profit_ttm",
    "fund_net_income_ttm", "fund_net_margin", "fund_operating_margin",
    "fund_rev_ttm", "fund_roa", "fund_roe", "fund_total_assets",
    "fund_total_debt", "fund_rev_growth_yoy",
    # Sentiment (last known signal is valid until new articles arrive)
    "news_sent_marketaux", "news_sent_marketaux_7d",
    "news_sent_av", "news_sent_av_7d",
    "news_sent_score", "news_sent_7d",
    # Options / insider snapshots
    "opt_atm_iv", "opt_put_call_oi", "opt_total_oi", "opt_max_pain",
    "insdr_change", "insdr_mspr",
    # Macro (available every trading day but ffill over holidays)
    "vix_level", "yield_10y", "yield_curve_slope",
]

FFILL_LIMIT  = 20   # max consecutive days to carry forward
BFILL_LIMIT  = 5    # max days to back-fill at start of series

for col in FFILL_COLS:
    if col not in df.columns:
        continue
    before = df[col].isnull().sum()
    if before == 0:
        continue
    df[col] = df[col].ffill(limit=FFILL_LIMIT).bfill(limit=BFILL_LIMIT)
    after = df[col].isnull().sum()
    if before != after:
        ffill_log.append((col, before, after))

print(f"\n[2] ffill→bfill applied to {len(ffill_log)} column(s):")
for col, before, after in ffill_log:
    print(f"      ↑  {col:<30}  {before} → {after} nulls remaining")

# 3. Median fill as backstop for any remaining numeric nulls
numeric_cols = df.select_dtypes(include="number").columns
remaining_null = df[numeric_cols].isnull().any()
cols_needing_median = remaining_null[remaining_null].index.tolist()

for col in cols_needing_median:
    before = df[col].isnull().sum()
    med    = df[col].median()
    df[col] = df[col].fillna(med)
    median_log.append((col, before, round(med, 6)))

if median_log:
    print(f"\n[3] Median fill backstop ({len(median_log)} column(s)):")
    for col, cnt, med in median_log:
        print(f"      ~  {col:<30}  {cnt} nulls → median={med}")
else:
    print(f"\n[3] No remaining nulls after ffill — median fill not needed.")

# 4. Final null audit
remaining = df.isnull().sum()
remaining = remaining[remaining > 0]
print(f"\n[4] Remaining nulls after all steps:")
if remaining.empty:
    print("      None — dataset is fully imputed.")
else:
    for col, n in remaining.items():
        print(f"      !  {col:<30}  {n} nulls ({n/len(df):.1%})")

# 4b. Drop zero-variance (constant) columns
numeric_cols_now = df.select_dtypes(include="number").columns
zero_var = [c for c in numeric_cols_now if df[c].std() == 0]
if zero_var:
    print(f"\n[4b] Dropping {len(zero_var)} constant (zero-variance) column(s):")
    for col in zero_var:
        val = df[col].iloc[0]
        print(f"      ✗  {col:<30}  (constant = {val})")
        drop_log.append((col, f"constant={val}"))
    df.drop(columns=zero_var, inplace=True)
else:
    print("\n[4b] No zero-variance columns found.")

# 5. Save
df.to_csv(dst)
print(f"\nOutput: {dst}  ({df.shape[0]} rows × {df.shape[1]} cols)")
print(f"Dropped {len(cols_to_drop)} col(s), kept {df.shape[1]} of {len(original_cols)} original.")
