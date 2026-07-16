"""
data.py — Walk-forward splits and per-fold leakage-free data preparation.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from imblearn.over_sampling import SMOTE

from config import LASSO_ALPHA, SMOTE_K


def make_walk_forward_splits(n_total, init_train, step, holdout, embargo=0):
    """Return list of (train_idx, val_idx) index pairs for walk-forward CV.

    `embargo` purges the last `embargo` rows of each training block (the ones whose
    forward-looking label overlaps the validation window), leaving a gap between
    train and val so validation metrics aren't inflated by look-ahead leakage.
    """
    available = n_total - holdout
    folds, train_end = [], init_train
    while train_end < available:
        val_end    = min(train_end + step, available)
        train_stop = max(0, train_end - embargo)   # purge leaked tail rows
        folds.append((list(range(0, train_stop)), list(range(train_end, val_end))))
        train_end += step
    return folds


def prepare_fold_data(df, train_idx, val_idx, features, target, use_smote=False):
    """
    Leakage-free preprocessing for a single fold.

    Steps (all fit on train only):
      1. NaN imputation via train medians
      2. StandardScaler
      3. Lasso feature selection
      4. SMOTE oversampling (train only, optional)

    Returns: X_tr, X_val, y_tr, y_val, sel_features, scaler, mask, col_med
    """
    X_raw_tr  = df.loc[train_idx, features].values.astype(float)
    X_raw_val = df.loc[val_idx,   features].values.astype(float)
    y_tr      = df.loc[train_idx, target].values.astype(float)
    y_val     = df.loc[val_idx,   target].values.astype(float)

    # imputation — train medians only
    col_med = np.nanmedian(X_raw_tr, axis=0)
    for j in range(X_raw_tr.shape[1]):
        X_raw_tr[np.isnan(X_raw_tr[:, j]), j]  = col_med[j]
        X_raw_val[np.isnan(X_raw_val[:, j]), j] = col_med[j]

    # scale — fit on train only
    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_raw_tr)
    X_val_s = scaler.transform(X_raw_val)

    # Lasso feature selection — fit on train only
    try:
        lasso = Lasso(alpha=LASSO_ALPHA, max_iter=5000)
        lasso.fit(X_tr_s, y_tr)
        mask = np.abs(lasso.coef_) > 0
        if mask.sum() < 3:
            mask = np.zeros(len(features), dtype=bool)
            mask[np.argsort(np.abs(lasso.coef_))[-10:]] = True
    except Exception:
        mask = np.ones(len(features), dtype=bool)

    sel   = [features[i] for i in range(len(features)) if mask[i]]
    X_tr  = X_tr_s[:, mask]
    X_val = X_val_s[:, mask]

    # SMOTE — train only, for imbalanced binary target
    if use_smote:
        n_min = pd.Series(y_tr.astype(int)).value_counts().min()
        k = min(SMOTE_K, n_min - 1)
        if k >= 1:
            try:
                sm = SMOTE(k_neighbors=k, random_state=42)
                X_tr, y_tr = sm.fit_resample(X_tr, y_tr.astype(int))
                y_tr = y_tr.astype(float)
            except Exception:
                pass

    return X_tr, X_val, y_tr, y_val, sel, scaler, mask, col_med
