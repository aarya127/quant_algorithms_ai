"""
targets.py — the six canonical prediction targets, defined once.

normalize.py appends them to the feature matrix; tests import add_targets()
directly so they check the real definitions.
"""
import numpy as np
import pandas as pd

TARGET_COLS = [
    "target_1d", "target_5d", "target_vol_5d",
    "target_dir_1d", "target_large_move", "target_regime",
]

FLAT_THRESHOLD     = 0.005   # ±0.5% dead zone → direction labelled flat
REGIME_MIN_HISTORY = 60      # rows before the expanding regime cut points are meaningful


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Append the targets to `df` (needs a `log_return` column) and return it.

    Row t's targets describe what happens after t's close, so they may use
    t+1..t+5 returns; nothing else here may look past t.
    """
    log_ret = df["log_return"]

    # Regression
    df["target_1d"]     = log_ret.shift(-1)
    df["target_5d"]     = log_ret.shift(-1).rolling(5).sum().shift(-4)   # sum of next 5 days
    df["target_vol_5d"] = log_ret.shift(-1).rolling(5).std().shift(-4)   # vol of next 5 days

    # Direction (3-class). np.where alone would label the last row — whose
    # next-day return is unknown — as flat.
    df["target_dir_1d"] = np.where(
        df["target_1d"] >  FLAT_THRESHOLD,  1,
        np.where(df["target_1d"] < -FLAT_THRESHOLD, -1, 0)
    ).astype(float)
    df.loc[df["target_1d"].isna(), "target_dir_1d"] = np.nan

    # Large move. The threshold uses vol known at t; a shift(-1) here would put
    # r(t+1), the very return being tested, into its own threshold.
    rolling_std = log_ret.rolling(20).std()
    df["target_large_move"] = (df["target_1d"].abs() > 2 * rolling_std).astype(float)
    df.loc[df["target_1d"].isna() | rolling_std.isna(), "target_large_move"] = np.nan

    # Forward volatility regime: tercile of the next 5 days' vol, ranked only
    # against values up to that row. (It used to be the tercile of same-day
    # realized_vol_20d — itself a feature — ranked over the full sample, which a
    # classifier could reproduce from its inputs: F1 ≈ 0.98.)
    vol_rank = df["target_vol_5d"].expanding(min_periods=REGIME_MIN_HISTORY).rank(pct=True)
    df["target_regime"] = pd.cut(vol_rank, bins=[0, 1/3, 2/3, 1.0],
                                 labels=[0, 1, 2]).astype(float)
    return df
