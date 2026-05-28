"""
run_pipeline.py — Extract, transform, and save a feature matrix to CSV.

Usage:
    python run_pipeline.py [SYMBOL] [PERIOD]

    SYMBOL : ticker symbol (default: NVDA)
    PERIOD : yfinance period string — 3mo, 6mo, 1y, 2y (default: 3mo)
             Pass 'full' to force a complete rebuild regardless of existing CSV.

Incremental mode:
    If <SYMBOL>_features.csv already exists, the pipeline detects the last
    date in it and only fetches + appends new trading days.  All sentiment
    API data is permanently cached so re-fetching overlapping ranges is free.
    New rows are appended and the CSV is de-duplicated (latest values win).

Full rebuild:
    Pass 'full' as PERIOD, or delete the CSV, to regenerate from scratch.

Output:
    <SYMBOL>_features.csv saved in this directory.
    Load it afterwards with load_features.py.
"""

import datetime
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

# Allow running from any cwd
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from algorithms.machine_learning_algorithms.data_pipelines import DataTransformer  # noqa: E402

SYMBOL = sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA"
PERIOD = sys.argv[2]         if len(sys.argv) > 2 else "1y"

out = Path(__file__).parent / f"{SYMBOL}_features.csv"

dt = DataTransformer()
today = datetime.date.today().isoformat()

# Detect incremental opportunity
_force_full = PERIOD.lower() == "full"
_existing: pd.DataFrame | None = None

if out.exists() and not _force_full:
    try:
        _existing = pd.read_csv(out, index_col=0, parse_dates=True)
        _existing.index = pd.to_datetime(_existing.index).tz_localize(None)
        _last_date = _existing.index.max().date()
        _next_date = _last_date + datetime.timedelta(days=1)

        if _next_date >= datetime.date.today():
            print(f"=== {SYMBOL}: already up to date (last row: {_last_date}) ===")
            print(f"CSV unchanged → {out}  ({len(_existing)} rows, {len(_existing.columns)} cols)")
            sys.exit(0)

        print(f"=== pipeline [INCREMENTAL]: {SYMBOL}  {_next_date} → {today} ===\n")
        df_new = dt.build_feature_matrix(SYMBOL, start=_next_date.isoformat(), end=today)

    except Exception as exc:
        print(f"Warning: could not read existing CSV ({exc}), falling back to full rebuild.")
        _existing = None

if _existing is None:
    # Full rebuild using the requested period
    actual_period = "1y" if _force_full else PERIOD
    print(f"=== pipeline [FULL]: {SYMBOL}  period={actual_period} ===\n")
    df_new = dt.build_feature_matrix(SYMBOL, period=actual_period)

if df_new is None or df_new.empty:
    print("No new data returned — CSV unchanged.")
    sys.exit(0)

# Merge & de-duplicate
if _existing is not None and not df_new.empty:
    # Align columns: add any new columns to the existing frame as NaN
    for col in df_new.columns:
        if col not in _existing.columns:
            _existing[col] = float("nan")
    for col in _existing.columns:
        if col not in df_new.columns:
            df_new[col] = float("nan")

    combined = pd.concat([_existing, df_new])
    # Keep last occurrence of each date (new data wins over old)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    added = len(combined) - len(_existing)
    print(f"\nAppended {added} new trading day(s).  Total: {len(combined)} rows.")
else:
    combined = df_new
    print(f"\nFull build: {len(combined)} rows.")

combined.to_csv(out)
print(f"Saved → {out}  ({combined.shape[0]} rows, {combined.shape[1]} cols)")
print(f"Load with: python load_features.py {SYMBOL}")
