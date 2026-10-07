"""
baselines.py — Target-specific naive baselines (Phase 2).
"""
import numpy as np
import pandas as pd


def target_baseline(y_tr, n_val, target):
    """Return a naive forecast array of length n_val for the given target."""
    if target == "target_1d":
        return np.zeros(n_val)                               # predict zero return
    elif target == "target_5d":
        return np.full(n_val, float(np.mean(y_tr[-20:])))   # rolling mean last 20
    elif target == "target_vol_5d":
        return np.full(n_val, float(np.median(y_tr[-5:])))  # recent median vol
    elif target == "target_dir_1d":
        maj = int(pd.Series(y_tr.astype(int)).mode()[0])
        return np.full(n_val, float(maj))                   # majority class
    elif target == "target_large_move":
        return np.zeros(n_val)                              # always no large move
    elif target == "target_regime":
        return np.full(n_val, float(int(y_tr[-1])))         # last observed regime
    return np.zeros(n_val)


def naive_signal(df, target):
    """Feature-based naive forecast for the given rows, used for the regression
    baseline's IC. The constant forecasts above have no rank correlation, so a
    beat-the-baseline check on IC could never fire."""
    if target == "target_vol_5d" and "realized_vol_20d" in df.columns:
        return df["realized_vol_20d"].values.astype(float)   # vol persists
    if "log_return" in df.columns:
        return df["log_return"].values.astype(float)         # momentum
    return None
