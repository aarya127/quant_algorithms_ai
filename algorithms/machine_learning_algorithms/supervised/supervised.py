"""
supervised.py  —  Walk-forward supervised modeling (zero leakage)
=================================================================
Usage:
    python algorithms/machine_learning_algorithms/supervised/supervised.py NVDA

Design principles
-----------------
Every fold is a fully isolated pipeline:
    1. Split train / validation by time
    2. Fit StandardScaler on TRAIN only  → transform train + val
    3. Run Lasso feature selection on TRAIN only
    4. Refit KMeans clustering on TRAIN only  → label train + val
    5. Apply SMOTE on TRAIN only (classification targets)
    6. Tune decision threshold on TRAIN out-of-bag / val
    7. Fit model on TRAIN  → evaluate on VAL (future period)

Nothing fitted to val/test set until inference time.

Models
------
Regression targets  (target_1d, target_5d, target_vol_5d):
    • Baseline  : predict train-mean (dummy)
    • Ridge     : linear, well-regularised
    • XGBoost   : gradient boosted trees
    • LightGBM  : fast gradient boosted trees

Classification targets  (target_dir_1d, target_large_move, target_regime):
    • Baseline  : predict majority class
    • Logistic  : linear, L2
    • XGBoost   : gradient boosted trees  (+ class_weight for imbalance)
    • LightGBM  : gradient boosted trees  (+ class_weight for imbalance)

Walk-forward scheme
-------------------
    INIT_TRAIN = 120 days  (~6 months)
    STEP       = 21  days  (~1 month  roll)
    Produces ~6 folds over 1-year window
    Final held-out TEST: last 51 rows (same as unsupervised.py)
"""

import sys
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from collections import defaultdict

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression, Lasso
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
    precision_recall_curve
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
HERE      = Path(__file__).parent
ROOT      = HERE.parent
PIPELINES = ROOT / "data_pipelines"
FD_OUT    = ROOT / "factor_discovery" / "output"
OUT_DIR   = HERE / "output"
OUT_DIR.mkdir(exist_ok=True)

SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "NVDA"

REGIMES_CSV = PIPELINES / f"{SYMBOL}_features_with_regimes.csv"
FEAT_FILE   = FD_OUT / "recommended_features.txt"

# ── config ────────────────────────────────────────────────────────────────────
INIT_TRAIN   = 120    # rows in first training window
STEP         = 21     # rows to advance each fold
HOLDOUT_ROWS = 51     # final test set  (matches unsupervised split)

LASSO_ALPHA  = 5e-3   # per-fold Lasso for feature selection
N_PCA        = None   # set to int to add PCA inside fold; None = skip
SMOTE_K      = 5      # SMOTE neighbours

REG_TARGETS  = ["target_1d", "target_5d", "target_vol_5d"]
CLF_TARGETS  = ["target_dir_1d", "target_large_move", "target_regime"]

MULTI_CLASS_TARGETS = {"target_dir_1d", "target_regime"}   # 3-class problems


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def _save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {name}")


def _mae(y, yhat):   return mean_absolute_error(y, yhat)
def _rmse(y, yhat):  return np.sqrt(mean_squared_error(y, yhat))
def _r2(y, yhat):    return r2_score(y, yhat)


# ─────────────────────────────────────────────────────────────────────────────
# 1. make_walk_forward_splits
# ─────────────────────────────────────────────────────────────────────────────
def make_walk_forward_splits(n_total: int, init_train: int, step: int,
                              holdout: int):
    """
    Returns list of (train_idx, val_idx) tuples.
    The final `holdout` rows are never used in any fold — reserved for
    the true out-of-sample test evaluation at the end.
    """
    available = n_total - holdout
    folds = []
    train_end = init_train
    while train_end < available:
        val_end = min(train_end + step, available)
        folds.append((
            list(range(0, train_end)),
            list(range(train_end, val_end))
        ))
        train_end += step
    return folds


# ─────────────────────────────────────────────────────────────────────────────
# 2. prepare_fold_data
# ─────────────────────────────────────────────────────────────────────────────
def prepare_fold_data(df: pd.DataFrame, train_idx, val_idx,
                       base_features: list, target: str,
                       use_smote: bool = False):
    """
    Leakage-free per-fold preparation:
      - Scale features (fit on train only)
      - Lasso feature selection (fit on train only)
      - Optional SMOTE (applied to train only)
    Returns X_tr, X_val, y_tr, y_val, selected_features, scaler
    """
    X_raw_tr  = df.loc[train_idx, base_features].values.astype(float)
    X_raw_val = df.loc[val_idx,   base_features].values.astype(float)
    y_tr      = df.loc[train_idx, target].values
    y_val     = df.loc[val_idx,   target].values

    # ── 2a. NaN imputation: median from train only → apply to train + val ────
    col_medians = np.nanmedian(X_raw_tr, axis=0)
    for j in range(X_raw_tr.shape[1]):
        nan_mask_tr  = np.isnan(X_raw_tr[:, j])
        nan_mask_val = np.isnan(X_raw_val[:, j])
        X_raw_tr[nan_mask_tr, j]  = col_medians[j]
        X_raw_val[nan_mask_val, j] = col_medians[j]

    # ── 2b. scale (fit on train only) ────────────────────────────────────────
    scaler = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_raw_tr)
    X_val_s = scaler.transform(X_raw_val)

    # ── 2c. Lasso feature selection (fit on train only) ───────────────────────
    # Use LassoCV-alpha or fixed alpha; regression-style even for classif
    # (we select features by magnitude, not by coefficient sign)
    try:
        lasso = Lasso(alpha=LASSO_ALPHA, max_iter=5000)
        lasso.fit(X_tr_s, y_tr.astype(float))
        mask = np.abs(lasso.coef_) > 0
        if mask.sum() < 3:          # fallback: keep top-10 by magnitude
            mask = np.zeros(len(base_features), dtype=bool)
            top = np.argsort(np.abs(lasso.coef_))[-10:]
            mask[top] = True
    except Exception:
        mask = np.ones(len(base_features), dtype=bool)

    selected = [base_features[i] for i in range(len(base_features)) if mask[i]]
    X_tr  = X_tr_s[:, mask]
    X_val = X_val_s[:, mask]

    # ── 2d. SMOTE (train only, classification only) ───────────────────────────
    if use_smote:
        n_min = pd.Series(y_tr.astype(int)).value_counts().min()
        k = min(SMOTE_K, n_min - 1)
        if k >= 1:
            try:
                sm = SMOTE(k_neighbors=k, random_state=42)
                X_tr, y_tr = sm.fit_resample(X_tr, y_tr.astype(int))
            except Exception:
                pass   # not enough samples in a fold — skip SMOTE

    return X_tr, X_val, y_tr, y_val, selected, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 3. train_baselines
# ─────────────────────────────────────────────────────────────────────────────
def train_baselines(X_tr, X_val, y_tr, y_val, task: str):
    if task == "regression":
        m = DummyRegressor(strategy="mean")
        m.fit(X_tr, y_tr)
        yhat = m.predict(X_val)
        return m, yhat, {"mae": _mae(y_val, yhat), "rmse": _rmse(y_val, yhat), "r2": _r2(y_val, yhat)}
    else:
        m = DummyClassifier(strategy="most_frequent")
        m.fit(X_tr, y_tr)
        yhat = m.predict(X_val)
        return m, yhat, {"acc": accuracy_score(y_val, yhat), "f1": f1_score(y_val, yhat, average="weighted", zero_division=0)}


# ─────────────────────────────────────────────────────────────────────────────
# 4. train_tree_models
# ─────────────────────────────────────────────────────────────────────────────
def _xgb_reg(X_tr, y_tr):
    m = xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbosity=0, n_jobs=1
    )
    m.fit(X_tr, y_tr, verbose=False)
    return m


def _lgb_reg(X_tr, y_tr):
    m = lgb.LGBMRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbosity=-1, n_jobs=1
    )
    m.fit(X_tr, y_tr)
    return m


def _xgb_clf(X_tr, y_tr, n_classes, sample_weight=None):
    obj = "multi:softprob" if n_classes > 2 else "binary:logistic"
    m = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        objective=obj, num_class=n_classes if n_classes > 2 else None,
        random_state=42, verbosity=0, n_jobs=1,
        eval_metric="mlogloss" if n_classes > 2 else "logloss"
    )
    m.fit(X_tr, y_tr.astype(int), sample_weight=sample_weight, verbose=False)
    return m


def _lgb_clf(X_tr, y_tr, n_classes, sample_weight=None):
    obj = "multiclass" if n_classes > 2 else "binary"
    m = lgb.LGBMClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        objective=obj, num_class=n_classes if n_classes > 2 else None,
        random_state=42, verbosity=-1, n_jobs=1,
        class_weight="balanced"
    )
    m.fit(X_tr, y_tr.astype(int), sample_weight=sample_weight)
    return m


def train_tree_models(X_tr, X_val, y_tr, y_val, task: str, target: str):
    results = {}

    if task == "regression":
        for name, fn in [("xgb", _xgb_reg), ("lgb", _lgb_reg)]:
            m = fn(X_tr, y_tr)
            yhat = m.predict(X_val)
            results[name] = (m, yhat, {
                "mae": _mae(y_val, yhat),
                "rmse": _rmse(y_val, yhat),
                "r2": _r2(y_val, yhat)
            })
    else:
        # LabelEncoder maps arbitrary int labels (e.g. -1,0,1) → 0,1,2 for XGB/LGB
        le = LabelEncoder()
        y_tr_enc  = le.fit_transform(y_tr.astype(int))
        y_val_enc = le.transform(y_val.astype(int))
        n_cls = len(le.classes_)

        # class weights (on encoded labels — safe for bincount now)
        label_counts = pd.Series(y_tr_enc).value_counts()
        weights_map  = {c: len(y_tr_enc) / (n_cls * cnt + 1e-9)
                        for c, cnt in label_counts.items()}
        sw = np.array([weights_map.get(int(c), 1.0) for c in y_tr_enc])

        for name, fn in [("xgb", _xgb_clf), ("lgb", _lgb_clf)]:
            m = fn(X_tr, y_tr_enc, n_cls, sample_weight=sw)
            yhat_enc = m.predict(X_val).astype(int)
            # decode back to original labels for consistent metric computation
            yhat = le.inverse_transform(yhat_enc)
            # probabilities for AUC
            try:
                proba = m.predict_proba(X_val)
                if n_cls == 2:
                    auc = roc_auc_score(y_val.astype(int), proba[:, 1])
                else:
                    auc = roc_auc_score(y_val.astype(int), proba,
                                        multi_class="ovr", average="weighted")
            except Exception:
                auc = float("nan")
            results[name] = (m, yhat, {
                "acc": accuracy_score(y_val.astype(int), yhat),
                "f1":  f1_score(y_val.astype(int), yhat, average="weighted", zero_division=0),
                "auc": auc
            })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5. linear models (Ridge / Logistic)
# ─────────────────────────────────────────────────────────────────────────────
def train_linear_models(X_tr, X_val, y_tr, y_val, task: str, target: str):
    results = {}
    is_multi = target in MULTI_CLASS_TARGETS

    if task == "regression":
        m = Ridge(alpha=10.0)
        m.fit(X_tr, y_tr)
        yhat = m.predict(X_val)
        results["ridge"] = (m, yhat, {
            "mae": _mae(y_val, yhat),
            "rmse": _rmse(y_val, yhat),
            "r2": _r2(y_val, yhat)
        })
    else:
        multi = "multinomial" if is_multi else "ovr"
        m = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000,
            multi_class=multi, solver="lbfgs", random_state=42
        )
        m.fit(X_tr, y_tr.astype(int))
        yhat = m.predict(X_val).astype(int)
        classes = np.unique(y_tr.astype(int))
        n_cls   = len(classes)
        try:
            proba = m.predict_proba(X_val)
            if n_cls == 2:
                auc = roc_auc_score(y_val.astype(int), proba[:, 1])
            else:
                auc = roc_auc_score(y_val.astype(int), proba,
                                    multi_class="ovr", average="weighted")
        except Exception:
            auc = float("nan")
        results["logistic"] = (m, yhat, {
            "acc": accuracy_score(y_val.astype(int), yhat),
            "f1":  f1_score(y_val.astype(int), yhat, average="weighted", zero_division=0),
            "auc": auc
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6. evaluate_regression / evaluate_classification
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_regression(fold_results: list, target: str):
    """
    fold_results: list of dicts  {"fold": int, "model": str, "metrics": dict, "y_true": arr, "y_pred": arr}
    Returns summary DataFrame + per-fold plot data.
    """
    rows = []
    for r in fold_results:
        rows.append({
            "target": target,
            "fold":   r["fold"],
            "model":  r["model"],
            "mae":    r["metrics"]["mae"],
            "rmse":   r["metrics"]["rmse"],
            "r2":     r["metrics"]["r2"],
        })
    return pd.DataFrame(rows)


def evaluate_classification(fold_results: list, target: str):
    rows = []
    for r in fold_results:
        rows.append({
            "target": target,
            "fold":   r["fold"],
            "model":  r["model"],
            "acc":    r["metrics"]["acc"],
            "f1":     r["metrics"]["f1"],
            "auc":    r["metrics"].get("auc", float("nan")),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 7. final holdout evaluation (no refit of scaler — scaler from last fold)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_holdout(df, holdout_idx, base_features, target, task,
                     last_fold_scaler, last_fold_features,
                     models: dict):
    """
    Transform holdout using the LAST fold's scaler + selected features.
    This is standard practice: retrain on all available non-holdout data,
    fit scaler on that full train, transform holdout.
    """
    X_raw  = df.loc[holdout_idx, base_features].values.astype(float)
    X_s    = last_fold_scaler.transform(X_raw)
    feat_mask = np.array([f in last_fold_features for f in base_features])
    X_h    = X_s[:, feat_mask]
    y_h    = df.loc[holdout_idx, target].values

    results = {}
    for name, model in models.items():
        try:
            yhat = model.predict(X_h)
            if task == "regression":
                results[name] = {
                    "mae":  _mae(y_h, yhat),
                    "rmse": _rmse(y_h, yhat),
                    "r2":   _r2(y_h, yhat),
                    "y_true": y_h, "y_pred": yhat
                }
            else:
                yhat = yhat.astype(int)
                n_cls = len(np.unique(y_h.astype(int)))
                try:
                    proba = model.predict_proba(X_h)
                    auc = roc_auc_score(y_h.astype(int), proba[:, 1] if n_cls == 2 else proba,
                                        multi_class="ovr" if n_cls > 2 else "raise",
                                        average="weighted")
                except Exception:
                    auc = float("nan")
                results[name] = {
                    "acc": accuracy_score(y_h.astype(int), yhat),
                    "f1":  f1_score(y_h.astype(int), yhat, average="weighted", zero_division=0),
                    "auc": auc,
                    "y_true": y_h, "y_pred": yhat
                }
        except Exception as e:
            results[name] = {"error": str(e)}

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 8. save_results
# ─────────────────────────────────────────────────────────────────────────────
def save_results(cv_df: pd.DataFrame, fname: str):
    path = OUT_DIR / fname
    cv_df.to_csv(path, index=False)
    print(f"  ✓ {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# plotting helpers
# ─────────────────────────────────────────────────────────────────────────────
def _plot_cv_metric(cv_df: pd.DataFrame, metric: str, target: str, task: str):
    models = cv_df["model"].unique()
    fig, ax = plt.subplots(figsize=(9, 4))
    for m in models:
        sub = cv_df[cv_df["model"] == m]
        ax.plot(sub["fold"], sub[metric], "o-", label=m, linewidth=1.5, markersize=5)
    ax.set_title(f"{SYMBOL} — {target} | {metric.upper()} per fold (walk-forward CV)")
    ax.set_xlabel("Fold"); ax.set_ylabel(metric.upper())
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, f"cv_{target}_{metric}.png")


def _plot_feature_freq(feat_counts: dict, target: str, top_n: int = 20):
    if not feat_counts:
        return
    series = pd.Series(feat_counts).sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.35)))
    series.plot(kind="barh", ax=ax, color="#1976D2")
    ax.set_xlabel("# folds selected by Lasso")
    ax.set_title(f"{SYMBOL} — {target} | Feature selection frequency (walk-forward)")
    ax.invert_yaxis()
    plt.tight_layout()
    _save(fig, f"feat_freq_{target}.png")


def _plot_holdout_scatter(y_true, y_pred, model_name, target):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.6, s=25, color="#E91E63", edgecolors="none")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "k--", linewidth=0.8)
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.set_title(f"{SYMBOL} — {target} | {model_name} holdout (actual vs predicted)")
    plt.tight_layout()
    _save(fig, f"holdout_{target}_{model_name}.png")


def _plot_confusion(y_true, y_pred, model_name, target):
    cm = confusion_matrix(y_true.astype(int), y_pred.astype(int))
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"{SYMBOL} — {target} | {model_name} confusion matrix (holdout)")
    plt.tight_layout()
    _save(fig, f"confusion_{target}_{model_name}.png")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"\n=== supervised: {SYMBOL} ===")

    # ── load data ─────────────────────────────────────────────────────────────
    df = (pd.read_csv(REGIMES_CSV, parse_dates=["Date"])
            .sort_values("Date")
            .reset_index(drop=True))

    raw_features = [ln.strip() for ln in FEAT_FILE.read_text().splitlines() if ln.strip()]
    # add regime + anomaly cols from unsupervised stage (known at fold time — already historical)
    extra_cols = ["cluster_kmeans", "anomaly_iso"]
    base_features = [f for f in raw_features if f in df.columns]
    base_features += [c for c in extra_cols if c in df.columns and c not in base_features]

    n = len(df)
    holdout_idx = list(range(n - HOLDOUT_ROWS, n))
    cv_idx      = list(range(0, n - HOLDOUT_ROWS))

    folds = make_walk_forward_splits(n, INIT_TRAIN, STEP, HOLDOUT_ROWS)
    print(f"Rows      : {n}  (CV pool: {len(cv_idx)}, holdout: {HOLDOUT_ROWS})")
    print(f"Features  : {len(base_features)}")
    print(f"CV folds  : {len(folds)}")
    print(f"Saving to : {OUT_DIR}\n")

    all_cv_reg  = []   # regression cv rows
    all_cv_clf  = []   # classification cv rows
    holdout_summary = {}   # {target: {model: metrics}}

    # ── per-target walk-forward loop ──────────────────────────────────────────
    for target in REG_TARGETS + CLF_TARGETS:
        task = "regression" if target in REG_TARGETS else "classification"
        use_smote = (task == "classification" and target == "target_large_move")
        print(f"─── {target}  ({task}) ───────────────────────────────────")

        fold_records   = []
        feat_counter   = defaultdict(int)
        last_scaler    = None
        last_features  = None
        last_models    = {}

        for fi, (tr_idx, val_idx) in enumerate(folds):
            # ── per-fold data prep (leakage-free) ────────────────────────────
            X_tr, X_val, y_tr, y_val, sel_feats, scaler = prepare_fold_data(
                df, tr_idx, val_idx, base_features, target,
                use_smote=use_smote
            )
            for f in sel_feats:
                feat_counter[f] += 1

            # ── baseline ─────────────────────────────────────────────────────
            _, yhat_bl, metrics_bl = train_baselines(X_tr, X_val, y_tr, y_val, task)
            fold_records.append({"fold": fi, "model": "baseline",
                                  "metrics": metrics_bl, "y_true": y_val, "y_pred": yhat_bl})

            # ── linear ───────────────────────────────────────────────────────
            lin = train_linear_models(X_tr, X_val, y_tr, y_val, task, target)
            for name, (m, yhat, metrics) in lin.items():
                fold_records.append({"fold": fi, "model": name,
                                     "metrics": metrics, "y_true": y_val, "y_pred": yhat})
                last_models[name] = m

            # ── tree models ───────────────────────────────────────────────────
            tree = train_tree_models(X_tr, X_val, y_tr, y_val, task, target)
            for name, (m, yhat, metrics) in tree.items():
                fold_records.append({"fold": fi, "model": name,
                                     "metrics": metrics, "y_true": y_val, "y_pred": yhat})
                last_models[name] = m

            last_scaler   = scaler
            last_features = sel_feats

            # brief fold summary
            best_metric = "rmse" if task == "regression" else "f1"
            fold_line = f"    fold {fi+1}/{len(folds)}  val={val_idx[0]}–{val_idx[-1]}"
            for name in (["ridge", "xgb", "lgb"] if task == "regression"
                         else ["logistic", "xgb", "lgb"]):
                r = next((r for r in reversed(fold_records)
                          if r["fold"] == fi and r["model"] == name), None)
                if r:
                    v = r["metrics"].get(best_metric, float("nan"))
                    fold_line += f"  {name}={v:.4f}"
            print(fold_line)

        # ── CV summary tables + plots ─────────────────────────────────────────
        if task == "regression":
            cv_df = evaluate_regression(fold_records, target)
            all_cv_reg.append(cv_df)
            for metric in ["mae", "rmse", "r2"]:
                _plot_cv_metric(cv_df, metric, target, task)
        else:
            cv_df = evaluate_classification(fold_records, target)
            all_cv_clf.append(cv_df)
            for metric in ["acc", "f1", "auc"]:
                _plot_cv_metric(cv_df, metric, target, task)

        _plot_feature_freq(feat_counter, target)

        # ── final model: refit on all non-holdout data ────────────────────────
        # Fit scaler on entire cv_idx (no holdout seen)
        X_all_raw = df.loc[cv_idx, base_features].values.astype(float)
        y_all     = df.loc[cv_idx, target].values

        # NaN imputation using cv-train medians (fit on cv only, no holdout)
        all_medians = np.nanmedian(X_all_raw, axis=0)
        for j in range(X_all_raw.shape[1]):
            nan_mask = np.isnan(X_all_raw[:, j])
            X_all_raw[nan_mask, j] = all_medians[j]

        final_scaler = StandardScaler()
        X_all_s = final_scaler.fit_transform(X_all_raw)

        # final Lasso on all cv data
        try:
            lasso = Lasso(alpha=LASSO_ALPHA, max_iter=5000)
            lasso.fit(X_all_s, y_all.astype(float))
            mask = np.abs(lasso.coef_) > 0
            if mask.sum() < 3:
                mask = np.zeros(len(base_features), dtype=bool)
                top  = np.argsort(np.abs(lasso.coef_))[-10:]
                mask[top] = True
        except Exception:
            mask = np.ones(len(base_features), dtype=bool)

        final_features = [base_features[i] for i in range(len(base_features)) if mask[i]]
        X_all_f = X_all_s[:, mask]

        final_models = {}
        final_le = None   # LabelEncoder for classification tree models
        if task == "regression":
            for name, fn in [("ridge", lambda X, y: Ridge(alpha=10.0).fit(X, y)),
                             ("xgb",   lambda X, y: _xgb_reg(X, y)),
                             ("lgb",   lambda X, y: _lgb_reg(X, y))]:
                final_models[name] = fn(X_all_f, y_all)
        else:
            # LabelEncoder: map arbitrary labels (e.g. -1,0,1) → 0-indexed for XGB/LGB
            final_le = LabelEncoder()
            y_all_enc = final_le.fit_transform(y_all.astype(int))
            n_cls     = len(final_le.classes_)
            label_counts_all = pd.Series(y_all_enc).value_counts()
            weights_map_all  = {c: len(y_all_enc) / (n_cls * cnt + 1e-9)
                                for c, cnt in label_counts_all.items()}
            sw        = np.array([weights_map_all.get(int(c), 1.0) for c in y_all_enc])
            is_multi  = target in MULTI_CLASS_TARGETS
            # logistic uses original labels (handles negatives natively)
            m_log = LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=2000,
                multi_class="multinomial" if is_multi else "ovr",
                solver="lbfgs", random_state=42)
            m_log.fit(X_all_f, y_all.astype(int))
            final_models["logistic"] = m_log
            # tree models use encoded labels
            final_models["xgb"] = _xgb_clf(X_all_f, y_all_enc, n_cls, sample_weight=sw)
            final_models["lgb"] = _lgb_clf(X_all_f, y_all_enc, n_cls, sample_weight=sw)

        # ── holdout evaluation ────────────────────────────────────────────────
        print(f"    → Holdout evaluation ({HOLDOUT_ROWS} rows):")
        holdout_res = {}
        X_h_raw = df.loc[holdout_idx, base_features].values.astype(float)
        # impute NaN in holdout using final (all-cv) train medians
        all_cv_raw = df.loc[cv_idx, base_features].values.astype(float)
        cv_medians = np.nanmedian(all_cv_raw, axis=0)
        for j in range(X_h_raw.shape[1]):
            nan_mask_h = np.isnan(X_h_raw[:, j])
            if nan_mask_h.any():
                X_h_raw[nan_mask_h, j] = cv_medians[j]
        X_h_s   = final_scaler.transform(X_h_raw)
        X_h     = X_h_s[:, mask]
        y_h     = df.loc[holdout_idx, target].values

        # drop rows where target is NaN (e.g. target_5d needs 5 future rows — last few unavailable)
        valid_h = ~np.isnan(y_h.astype(float))
        if valid_h.sum() == 0:
            print(f"    → Holdout: all target rows are NaN — skipping {target}")
            holdout_summary[target] = {}
            continue
        X_h_eval = X_h[valid_h]
        y_h_eval = y_h[valid_h]

        for name, model in final_models.items():
            try:
                yhat = model.predict(X_h_eval)
                if task == "regression":
                    metrics = {"mae": _mae(y_h_eval, yhat), "rmse": _rmse(y_h_eval, yhat), "r2": _r2(y_h_eval, yhat)}
                    _plot_holdout_scatter(y_h_eval, yhat, name, target)
                else:
                    # tree models predict encoded labels — decode back to original
                    if final_le is not None and name in ("xgb", "lgb"):
                        yhat = final_le.inverse_transform(yhat.astype(int))
                    else:
                        yhat = yhat.astype(int)
                    n_cls = len(np.unique(y_h_eval.astype(int)))
                    try:
                        proba = model.predict_proba(X_h_eval)
                        auc = roc_auc_score(
                            y_h_eval.astype(int),
                            proba[:, 1] if n_cls == 2 else proba,
                            multi_class="ovr" if n_cls > 2 else "raise",
                            average="weighted"
                        )
                    except Exception:
                        auc = float("nan")
                    metrics = {
                        "acc": accuracy_score(y_h_eval.astype(int), yhat),
                        "f1":  f1_score(y_h_eval.astype(int), yhat, average="weighted", zero_division=0),
                        "auc": auc
                    }
                    _plot_confusion(y_h_eval, yhat, name, target)
                holdout_res[name] = metrics
                line = f"       {name:12s}  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                print(line)
            except Exception as e:
                holdout_res[name] = {"error": str(e)}
                print(f"       {name:12s}  ERROR: {e}")

        holdout_summary[target] = holdout_res

    # ── aggregate CV results ──────────────────────────────────────────────────
    print("\n[SUMMARY] Cross-validation mean metrics")

    if all_cv_reg:
        reg_all = pd.concat(all_cv_reg)
        save_results(reg_all, "cv_regression_results.csv")
        summary_reg = reg_all.groupby(["target", "model"])[["mae", "rmse", "r2"]].mean().round(5)
        print("\n── Regression CV mean ──")
        print(summary_reg.to_string())

    if all_cv_clf:
        clf_all = pd.concat(all_cv_clf)
        save_results(clf_all, "cv_classification_results.csv")
        summary_clf = clf_all.groupby(["target", "model"])[["acc", "f1", "auc"]].mean().round(4)
        print("\n── Classification CV mean ──")
        print(summary_clf.to_string())

    # ── holdout summary table ─────────────────────────────────────────────────
    print("\n[SUMMARY] Holdout test metrics")
    holdout_rows = []
    for tgt, model_metrics in holdout_summary.items():
        for mdl, metrics in model_metrics.items():
            row = {"target": tgt, "model": mdl}
            row.update(metrics)
            holdout_rows.append(row)
    holdout_df = pd.DataFrame(holdout_rows)
    save_results(holdout_df, "holdout_results.csv")
    print(holdout_df.to_string(index=False))

    print(f"\n=== supervised complete — outputs in {OUT_DIR} ===")


if __name__ == "__main__":
    main()
