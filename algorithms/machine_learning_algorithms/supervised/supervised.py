"""
supervised.py  —  Walk-forward supervised modeling (zero leakage)
=================================================================
Usage:
    python algorithms/machine_learning_algorithms/supervised/supervised.py NVDA

Phase 2 — Target-specific baselines
    target_1d      → predict 0 (zero return)
    target_5d      → predict rolling mean of recent train returns
    target_vol_5d  → predict median of recent train vol
    target_dir_1d  → predict majority class
    target_large_move → always predict 0 (no large move)
    target_regime  → predict last seen regime

Phase 3 — Model ladder (per task):
    Regression:     Ridge, ElasticNet, HuberRegressor, RandomForest, XGBoost, LightGBM
    Classification: Logistic, CalibratedSVC, RandomForest, XGBoost, LightGBM

Phase 4 — Target-specific metrics:
    Regression: MAE, RMSE, R², directional accuracy, IC (Spearman rank corr)
    Classification: accuracy, balanced accuracy, F1-weighted, F1-macro
    target_large_move extras: precision, recall, PR-AUC, ROC-AUC
    target_regime extras: per-class F1 in confusion matrix

Feature set comparison:
    Version A: 34 recommended features only
    Version B: 34 features + cluster_kmeans + anomaly_iso + anomaly_score

Leakage controls — per fold:
    1. Split by time
    2. Median imputation fit on train only
    3. StandardScaler fit on train only
    4. Lasso feature selection fit on train only
    5. SMOTE applied to train only (target_large_move)
    6. Class weights / scale_pos_weight computed from train only
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from scipy.stats import spearmanr

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor, Lasso, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_curve,
)
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
INIT_TRAIN   = 120
STEP         = 21
HOLDOUT_ROWS = 51

LASSO_ALPHA  = 5e-3
SMOTE_K      = 5

REG_TARGETS  = ["target_1d", "target_5d", "target_vol_5d"]
CLF_TARGETS  = ["target_dir_1d", "target_large_move", "target_regime"]
BINARY_CLF   = {"target_large_move"}
MULTI_CLF    = {"target_dir_1d", "target_regime"}

REGIME_COLS  = ["cluster_kmeans", "anomaly_iso", "anomaly_score"]


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def _save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {name}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. walk-forward splits
# ─────────────────────────────────────────────────────────────────────────────
def make_walk_forward_splits(n_total, init_train, step, holdout):
    available = n_total - holdout
    folds, train_end = [], init_train
    while train_end < available:
        val_end = min(train_end + step, available)
        folds.append((list(range(0, train_end)), list(range(train_end, val_end))))
        train_end += step
    return folds


# ─────────────────────────────────────────────────────────────────────────────
# 2. per-fold leakage-free data prep
# ─────────────────────────────────────────────────────────────────────────────
def prepare_fold_data(df, train_idx, val_idx, features, target, use_smote=False):
    X_raw_tr  = df.loc[train_idx, features].values.astype(float)
    X_raw_val = df.loc[val_idx,   features].values.astype(float)
    y_tr      = df.loc[train_idx, target].values.astype(float)
    y_val     = df.loc[val_idx,   target].values.astype(float)

    # NaN imputation — medians from train only
    col_med = np.nanmedian(X_raw_tr, axis=0)
    for j in range(X_raw_tr.shape[1]):
        X_raw_tr[np.isnan(X_raw_tr[:, j]), j]  = col_med[j]
        X_raw_val[np.isnan(X_raw_val[:, j]), j] = col_med[j]

    # scale — fit on train only
    scaler = StandardScaler()
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


# ─────────────────────────────────────────────────────────────────────────────
# 3. target-specific baselines  (Phase 2)
# ─────────────────────────────────────────────────────────────────────────────
def target_baseline(y_tr, n_val, target):
    if target == "target_1d":
        return np.zeros(n_val)                              # predict zero return
    elif target == "target_5d":
        return np.full(n_val, float(np.mean(y_tr[-20:])))  # rolling mean last 20
    elif target == "target_vol_5d":
        return np.full(n_val, float(np.median(y_tr[-5:]))) # recent median vol
    elif target == "target_dir_1d":
        maj = int(pd.Series(y_tr.astype(int)).mode()[0])
        return np.full(n_val, float(maj))                  # majority class
    elif target == "target_large_move":
        return np.zeros(n_val)                             # always no large move
    elif target == "target_regime":
        return np.full(n_val, float(int(y_tr[-1])))        # last observed regime
    return np.zeros(n_val)


# ─────────────────────────────────────────────────────────────────────────────
# 4. target-specific metrics  (Phase 4)
# ─────────────────────────────────────────────────────────────────────────────
def reg_metrics(y_true, y_pred):
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[valid], y_pred[valid]
    if len(yt) < 2:
        return {}
    ic, _ = spearmanr(yt, yp)
    return {
        "mae":     float(mean_absolute_error(yt, yp)),
        "rmse":    float(np.sqrt(mean_squared_error(yt, yp))),
        "r2":      float(r2_score(yt, yp)),
        "dir_acc": float(np.mean(np.sign(yt) == np.sign(yp))),
        "ic":      float(ic) if not np.isnan(ic) else 0.0,
    }


def clf_metrics(y_true, y_pred, proba, target):
    yt = y_true.astype(int)
    yp = y_pred.astype(int)
    m = {
        "acc":      float(accuracy_score(yt, yp)),
        "bal_acc":  float(balanced_accuracy_score(yt, yp)),
        "f1_w":     float(f1_score(yt, yp, average="weighted",  zero_division=0)),
        "f1_macro": float(f1_score(yt, yp, average="macro",     zero_division=0)),
    }
    if target == "target_large_move":
        m["precision"] = float(precision_score(yt, yp, zero_division=0))
        m["recall"]    = float(recall_score(yt, yp,    zero_division=0))
        if proba is not None and proba.ndim == 2 and proba.shape[1] >= 2:
            try:
                m["pr_auc"]  = float(average_precision_score(yt, proba[:, 1]))
                m["roc_auc"] = float(roc_auc_score(yt, proba[:, 1]))
            except Exception:
                pass
    elif proba is not None:
        n_cls = len(np.unique(yt))
        try:
            if n_cls == 2:
                m["roc_auc"] = float(roc_auc_score(yt, proba[:, 1]))
            else:
                m["roc_auc"] = float(roc_auc_score(
                    yt, proba, multi_class="ovr", average="weighted"))
        except Exception:
            pass
    return m


# ─────────────────────────────────────────────────────────────────────────────
# 5. model factories  (Phase 3)
# ─────────────────────────────────────────────────────────────────────────────
def reg_model_set():
    return {
        "ridge":       Ridge(alpha=10.0),
        "elasticnet":  ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
        "huber":       HuberRegressor(epsilon=1.5, max_iter=500),
        "rf":          RandomForestRegressor(n_estimators=200, max_depth=6,
                                             random_state=42, n_jobs=1),
        "xgb":         xgb.XGBRegressor(n_estimators=300, max_depth=4,
                                         learning_rate=0.05, subsample=0.8,
                                         colsample_bytree=0.8, reg_alpha=0.1,
                                         reg_lambda=1.0, random_state=42,
                                         verbosity=0, n_jobs=1),
        "lgb":         lgb.LGBMRegressor(n_estimators=300, max_depth=4,
                                          learning_rate=0.05, subsample=0.8,
                                          colsample_bytree=0.8, reg_alpha=0.1,
                                          reg_lambda=1.0, random_state=42,
                                          verbosity=-1, n_jobs=1),
    }


def clf_model_set(n_cls, is_binary_imb, spw, is_multi):
    multi = "multinomial" if is_multi else "ovr"
    return {
        "logistic": LogisticRegression(C=1.0, class_weight="balanced",
                                        max_iter=2000,
                                        solver="lbfgs", random_state=42),
        "svc_cal":  CalibratedClassifierCV(
                        LinearSVC(class_weight="balanced", max_iter=5000,
                                  random_state=42), cv=3),
        "rf":       RandomForestClassifier(n_estimators=200, max_depth=6,
                                           class_weight="balanced",
                                           random_state=42, n_jobs=1),
        "xgb":      xgb.XGBClassifier(
                        n_estimators=300, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        reg_alpha=0.1, reg_lambda=1.0,
                        objective="multi:softprob" if n_cls > 2 else "binary:logistic",
                        num_class=n_cls if n_cls > 2 else None,
                        scale_pos_weight=spw if (is_binary_imb and n_cls == 2) else 1.0,
                        random_state=42, verbosity=0, n_jobs=1,
                        eval_metric="mlogloss" if n_cls > 2 else "logloss"),
        "lgb":      lgb.LGBMClassifier(
                        n_estimators=300, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        reg_alpha=0.1, reg_lambda=1.0,
                        objective="multiclass" if n_cls > 2 else "binary",
                        num_class=n_cls if n_cls > 2 else None,
                        class_weight="balanced",
                        random_state=42, verbosity=-1, n_jobs=1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5b. native feature importance extraction (no SHAP required)
# ─────────────────────────────────────────────────────────────────────────────
def _get_feature_importances(model, name, sel_features):
    """Return a {feature: normalized_importance} dict for supported models."""
    try:
        if name in ("rf", "xgb", "lgb"):
            fi = np.array(model.feature_importances_, dtype=float)
        elif name in ("ridge", "elasticnet", "huber"):
            fi = np.abs(np.array(model.coef_, dtype=float))
        elif name == "logistic":
            coef = np.array(model.coef_, dtype=float)
            fi = np.abs(coef).mean(axis=0) if coef.ndim == 2 else np.abs(coef[0])
        else:
            return None         # svc_cal — calibrated SVC internals are complex; skip
        if len(fi) != len(sel_features):
            return None
        total = fi.sum()
        if total > 0:
            fi = fi / total
        return dict(zip(sel_features, fi.tolist()))
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 6. fold runner
# ─────────────────────────────────────────────────────────────────────────────
def run_folds(df, folds, features, target, task):
    use_smote     = (target == "target_large_move")
    is_binary_imb = (target in BINARY_CLF)
    is_multi      = (target in MULTI_CLF)

    fold_records = []
    feat_counter = defaultdict(int)

    for fi, (tr_idx, val_idx) in enumerate(folds):
        X_tr, X_val, y_tr, y_val, sel, scaler, mask, _ = prepare_fold_data(
            df, tr_idx, val_idx, features, target, use_smote=use_smote)
        for f in sel:
            feat_counter[f] += 1

        valid = ~np.isnan(y_val)
        y_bl  = target_baseline(y_tr, int(valid.sum()), target)

        if task == "regression":
            fold_records.append({"fold": fi, "model": "baseline",
                                 **reg_metrics(y_val[valid], y_bl)})
            for name, m in reg_model_set().items():
                try:
                    m.fit(X_tr, y_tr)
                    yhat = m.predict(X_val)
                    fold_records.append({"fold": fi, "model": name,
                                         **reg_metrics(y_val[valid], yhat[valid])})
                except Exception as e:
                    fold_records.append({"fold": fi, "model": name, "error": str(e)})
        else:
            le         = LabelEncoder()
            y_tr_enc   = le.fit_transform(y_tr.astype(int))
            y_val_orig = y_val.astype(int)
            n_cls      = len(le.classes_)

            lc   = pd.Series(y_tr_enc).value_counts()
            spw  = float(lc.get(0, 1) / max(lc.get(1, 1), 1)) if is_binary_imb else 1.0
            sw_m = {c: len(y_tr_enc)/(n_cls * cnt + 1e-9) for c, cnt in lc.items()}
            sw   = np.array([sw_m.get(int(c), 1.0) for c in y_tr_enc])

            # baseline (clamp to valid label range before inverse_transform)
            y_bl_i = np.clip(y_bl.astype(int),
                             int(le.classes_.min()), int(le.classes_.max()))
            fold_records.append({"fold": fi, "model": "baseline",
                                 **clf_metrics(y_val_orig, y_bl_i, None, target)})

            for name, m in clf_model_set(n_cls, is_binary_imb, spw, is_multi).items():
                try:
                    if name in ("xgb", "lgb"):
                        m.fit(X_tr, y_tr_enc, sample_weight=sw)
                        yhat = le.inverse_transform(
                            np.clip(m.predict(X_val).astype(int), 0, n_cls - 1))
                        proba = m.predict_proba(X_val)
                    else:
                        m.fit(X_tr, y_tr.astype(int))
                        yhat  = m.predict(X_val).astype(int)
                        proba = m.predict_proba(X_val) if hasattr(m, "predict_proba") else None
                    fold_records.append({"fold": fi, "model": name,
                                         **clf_metrics(y_val_orig, yhat, proba, target)})
                except Exception as e:
                    fold_records.append({"fold": fi, "model": name, "error": str(e)})

        # summary line
        km = "ic" if task == "regression" else "f1_w"
        parts = [f"fold {fi+1}/{len(folds)}  val={val_idx[0]}–{val_idx[-1]}"]
        for r in fold_records:
            if r["fold"] == fi and km in r:
                parts.append(f"{r['model']}={r[km]:.4f}")
        print("    " + "  ".join(parts))

    return fold_records, feat_counter


# ─────────────────────────────────────────────────────────────────────────────
# 7. final refit + holdout evaluation
# ─────────────────────────────────────────────────────────────────────────────
def run_holdout(df, cv_idx, holdout_idx, features, target, task):
    use_smote     = (target == "target_large_move")
    is_binary_imb = (target in BINARY_CLF)
    is_multi      = (target in MULTI_CLF)

    X_raw = df.loc[cv_idx, features].values.astype(float)
    y_all = df.loc[cv_idx, target].values.astype(float)

    col_med = np.nanmedian(X_raw, axis=0)
    for j in range(X_raw.shape[1]):
        X_raw[np.isnan(X_raw[:, j]), j] = col_med[j]

    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X_raw)

    try:
        lasso = Lasso(alpha=LASSO_ALPHA, max_iter=5000)
        lasso.fit(X_s, y_all)
        mask = np.abs(lasso.coef_) > 0
        if mask.sum() < 3:
            mask = np.zeros(len(features), dtype=bool)
            mask[np.argsort(np.abs(lasso.coef_))[-10:]] = True
    except Exception:
        mask = np.ones(len(features), dtype=bool)

    X_cv         = X_s[:, mask]
    sel_features = [features[i] for i in range(len(features)) if mask[i]]

    X_h_raw = df.loc[holdout_idx, features].values.astype(float)
    for j in range(X_h_raw.shape[1]):
        X_h_raw[np.isnan(X_h_raw[:, j]), j] = col_med[j]
    X_h   = scaler.transform(X_h_raw)[:, mask]
    y_h   = df.loc[holdout_idx, target].values.astype(float)

    valid_h = ~np.isnan(y_h)
    X_h_ev  = X_h[valid_h]
    y_h_ev  = y_h[valid_h]

    results = {}

    if task == "regression":
        y_bl = target_baseline(y_all, len(y_h_ev), target)
        results["baseline"] = {"metrics": reg_metrics(y_h_ev, y_bl),
                               "y_true": y_h_ev, "y_pred": y_bl}
        for name, m in reg_model_set().items():
            try:
                m.fit(X_cv, y_all)
                yhat = m.predict(X_h_ev)
                results[name] = {
                    "metrics": reg_metrics(y_h_ev, yhat),
                    "y_true": y_h_ev, "y_pred": yhat,
                    "importances": _get_feature_importances(m, name, sel_features),
                }
            except Exception as e:
                results[name] = {"error": str(e)}

    else:
        le       = LabelEncoder()
        y_cv_enc = le.fit_transform(y_all.astype(int))
        n_cls    = len(le.classes_)

        lc   = pd.Series(y_cv_enc).value_counts()
        spw  = float(lc.get(0, 1) / max(lc.get(1, 1), 1)) if is_binary_imb else 1.0
        sw_m = {c: len(y_cv_enc)/(n_cls * cnt + 1e-9) for c, cnt in lc.items()}
        sw   = np.array([sw_m.get(int(c), 1.0) for c in y_cv_enc])

        # after SMOTE X_cv grows — track resampled labels for all models
        y_cv_orig_fit = y_all.astype(int)   # default: no SMOTE
        if use_smote:
            n_min = int(pd.Series(y_cv_enc).value_counts().min())
            k = min(SMOTE_K, n_min - 1)
            if k >= 1:
                try:
                    sm = SMOTE(k_neighbors=k, random_state=42)
                    X_cv, y_cv_enc = sm.fit_resample(X_cv, y_cv_enc)
                    y_cv_orig_fit = le.inverse_transform(y_cv_enc)  # keep aligned
                    sw = None
                except Exception:
                    pass

        y_bl  = target_baseline(y_all, len(y_h_ev), target)
        y_bl_i = np.clip(y_bl.astype(int), int(le.classes_.min()), int(le.classes_.max()))
        results["baseline"] = {"metrics": clf_metrics(y_h_ev.astype(int), y_bl_i, None, target),
                               "y_true": y_h_ev, "y_pred": y_bl_i.astype(float)}

        for name, m in clf_model_set(n_cls, is_binary_imb, spw, is_multi).items():
            try:
                if name in ("xgb", "lgb"):
                    m.fit(X_cv, y_cv_enc, sample_weight=sw)
                    yhat_enc = m.predict(X_h_ev).astype(int)
                    yhat  = le.inverse_transform(np.clip(yhat_enc, 0, n_cls - 1)).astype(float)
                    proba = m.predict_proba(X_h_ev)
                else:
                    m.fit(X_cv, y_cv_orig_fit)   # uses SMOTE-resampled labels if applicable
                    yhat  = m.predict(X_h_ev).astype(float)
                    proba = m.predict_proba(X_h_ev) if hasattr(m, "predict_proba") else None
                results[name] = {
                    "metrics": clf_metrics(y_h_ev.astype(int), yhat.astype(int), proba, target),
                    "y_true": y_h_ev, "y_pred": yhat,
                    "proba": proba,
                    "importances": _get_feature_importances(m, name, sel_features),
                }
            except Exception as e:
                results[name] = {"error": str(e)}

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 8. plots
# ─────────────────────────────────────────────────────────────────────────────
def _primary_metric(task, target):
    if task == "regression":
        return "ic"
    return "pr_auc" if target == "target_large_move" else "f1_w"


def plot_cv_primary(cv_df, target, task, version):
    """One chart: primary metric across folds, all models."""
    metric = _primary_metric(task, target)
    if metric not in cv_df.columns:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    for model in cv_df["model"].unique():
        sub = cv_df[cv_df["model"] == model]
        style = dict(linewidth=1.8, markersize=6)
        if model == "baseline":
            style.update(color="gray", linestyle="--", linewidth=1.2)
        ax.plot(sub["fold"], sub[metric], "o-", label=model, **style)
    ax.set_title(f"{SYMBOL} v{version} — {target} | {metric} (walk-forward CV)")
    ax.set_xlabel("Fold"); ax.set_ylabel(metric)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, f"v{version}_cv_{target}.png")


def plot_feat_freq(feat_counter, target, version, top_n=20):
    if not feat_counter:
        return
    s = pd.Series(feat_counter).sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(9, max(4, len(s) * 0.35)))
    s.plot(kind="barh", ax=ax, color="#1976D2")
    ax.set_xlabel("# folds selected by Lasso")
    ax.set_title(f"{SYMBOL} v{version} — {target} | Feature selection frequency")
    ax.invert_yaxis(); plt.tight_layout()
    _save(fig, f"v{version}_feat_freq_{target}.png")


def plot_holdout_bar(h_res, target, task, version):
    """Horizontal bar: primary metric for all models on the holdout set."""
    metric = _primary_metric(task, target)
    rows = [(m, res["metrics"].get(metric))
            for m, res in h_res.items()
            if "metrics" in res and res["metrics"].get(metric) is not None]
    if not rows:
        return
    models, vals = zip(*rows)
    colors = ["#9E9E9E" if m == "baseline" else "#1976D2" for m in models]
    fig, ax = plt.subplots(figsize=(7, max(3, len(rows) * 0.45)))
    ax.barh(models, vals, color=colors, height=0.6, edgecolor="white")
    bl = h_res.get("baseline", {}).get("metrics", {}).get(metric)
    if bl is not None:
        ax.axvline(bl, color="red", linestyle="--", linewidth=1.2,
                   label=f"baseline={bl:.3f}")
        ax.legend(fontsize=8)
    ax.set_xlabel(metric)
    ax.set_title(f"{SYMBOL} v{version} — {target} | holdout {metric}")
    ax.invert_yaxis()
    plt.tight_layout()
    _save(fig, f"v{version}_holdout_{target}.png")


def plot_pr_combined(h_res, target, version):
    """All models' PR curves on one chart for the imbalanced binary target."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for model, res in h_res.items():
        if model == "baseline" or res.get("proba") is None:
            continue
        try:
            prec, rec, _ = precision_recall_curve(
                res["y_true"].astype(int), res["proba"][:, 1])
            ap = average_precision_score(
                res["y_true"].astype(int), res["proba"][:, 1])
            ax.plot(rec, prec, linewidth=1.5, label=f"{model} AP={ap:.3f}")
        except Exception:
            pass
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"{SYMBOL} v{version} — {target} | PR curves (holdout)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, f"v{version}_pr_curves_{target}.png")


def plot_best_holdout(h_res, target, task, version):
    """Scatter (regression) or confusion (classification) for the best model only."""
    metric = _primary_metric(task, target)
    best, best_val = None, -np.inf
    for model, res in h_res.items():
        if model == "baseline" or "metrics" not in res:
            continue
        v = res["metrics"].get(metric)
        if v is not None and v > best_val:
            best_val, best = v, model
    if best is None:
        return
    res = h_res[best]
    y_t, y_p = res["y_true"], res["y_pred"]
    if task == "regression":
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(y_t, y_p, alpha=0.6, s=25, color="#E91E63", edgecolors="none")
        lims = [min(y_t.min(), y_p.min()), max(y_t.max(), y_p.max())]
        ax.plot(lims, lims, "k--", linewidth=0.8)
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
        ax.set_title(f"{SYMBOL} v{version} — {target} | best: {best} ({metric}={best_val:.3f})")
        plt.tight_layout()
        _save(fig, f"v{version}_best_{target}.png")
    else:
        cm = confusion_matrix(y_t.astype(int), y_p.astype(int))
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"{SYMBOL} v{version} — {target} | best: {best} ({metric}={best_val:.3f})")
        plt.tight_layout()
        _save(fig, f"v{version}_best_{target}.png")


def plot_feature_importance(h_res, target, task, version, top_n=20):
    """Horizontal bar: native feature importances for the best model on holdout."""
    metric = _primary_metric(task, target)
    best, best_val = None, -np.inf
    for model, res in h_res.items():
        if model == "baseline" or "metrics" not in res:
            continue
        if res.get("importances") is None:
            continue
        v = res["metrics"].get(metric)
        if v is not None and v > best_val:
            best_val, best = v, model
    if best is None:
        return

    imp = h_res[best]["importances"]
    s = pd.Series(imp).sort_values(ascending=False).head(top_n)
    if s.empty:
        return

    fig, ax = plt.subplots(figsize=(9, max(4, len(s) * 0.42)))
    s.plot(kind="barh", ax=ax, color="#388E3C", edgecolor="white")
    ax.set_xlabel("Normalized importance")
    ax.set_title(
        f"{SYMBOL} v{version} — {target} | Feature importance  [{best}]  top-{top_n}"
    )
    ax.invert_yaxis()
    ax.grid(True, alpha=0.25, axis="x")
    plt.tight_layout()
    _save(fig, f"v{version}_feat_imp_{target}.png")


def plot_version_comparison(all_holdout, targets, task):
    key = "ic" if task == "regression" else "f1_w"
    rows = []
    for tgt in targets:
        for ver in ["A", "B"]:
            if (tgt, ver) not in all_holdout:
                continue
            for model, res in all_holdout[(tgt, ver)].items():
                if "error" in res or "metrics" not in res:
                    continue
                v = res["metrics"].get(key)
                if v is not None:
                    rows.append({"target": tgt, "version": ver, "model": model, key: v})
    if not rows:
        return
    df_p = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 5), sharey=False)
    if len(targets) == 1:
        axes = [axes]
    for ax, tgt in zip(axes, targets):
        sub = df_p[df_p["target"] == tgt]
        if sub.empty:
            continue
        pivot = sub.pivot(index="model", columns="version", values=key)
        pivot.plot(kind="bar", ax=ax, color=["#1976D2", "#E91E63"], width=0.65)
        ax.set_title(tgt, fontsize=9); ax.set_xlabel("")
        ax.set_ylabel(key); ax.legend(title="Version", fontsize=8)
        ax.tick_params(axis="x", labelrotation=40)
        ax.grid(True, alpha=0.2, axis="y")
    task_label = task
    fig.suptitle(f"{SYMBOL} — Version A vs B  |  {key} (holdout)", fontsize=11)
    plt.tight_layout()
    _save(fig, f"version_AB_{task_label}.png")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"\n=== supervised: {SYMBOL} ===")

    df = (pd.read_csv(REGIMES_CSV, parse_dates=["Date"])
            .sort_values("Date").reset_index(drop=True))

    raw_feats = [ln.strip() for ln in FEAT_FILE.read_text().splitlines() if ln.strip()]
    feat_A    = [f for f in raw_feats if f in df.columns]
    feat_B    = feat_A + [c for c in REGIME_COLS if c in df.columns and c not in feat_A]

    n           = len(df)
    holdout_idx = list(range(n - HOLDOUT_ROWS, n))
    cv_idx      = list(range(0, n - HOLDOUT_ROWS))
    folds       = make_walk_forward_splits(n, INIT_TRAIN, STEP, HOLDOUT_ROWS)

    print(f"Rows        : {n}  (CV: {len(cv_idx)}, holdout: {HOLDOUT_ROWS})")
    print(f"Features A  : {len(feat_A)}  |  B: {len(feat_B)}")
    print(f"CV folds    : {len(folds)}")
    print(f"Saving to   : {OUT_DIR}\n")

    all_cv_rows = []
    all_holdout = {}   # {(target, version): {model: result}}

    for version, features in [("A", feat_A), ("B", feat_B)]:
        print(f"\n{'═'*62}")
        print(f"  VERSION {version}  ({len(features)} features)")
        print(f"{'═'*62}")

        for target in REG_TARGETS + CLF_TARGETS:
            task = "regression" if target in REG_TARGETS else "classification"
            print(f"\n── {target}  [{task}]  v{version} ──────────────────────")

            fold_records, feat_counter = run_folds(df, folds, features, target, task)

            fd = pd.DataFrame(fold_records)
            fd.insert(0, "version", version)
            fd.insert(1, "target", target)
            all_cv_rows.append(fd)

            plot_cv_primary(fd, target, task, version)
            if version == "A":                          # feature freq once is enough
                plot_feat_freq(feat_counter, target, version)

            # holdout
            print(f"    → Holdout ({HOLDOUT_ROWS} rows):")
            h_res = run_holdout(df, cv_idx, holdout_idx, features, target, task)
            all_holdout[(target, version)] = h_res

            for model, res in h_res.items():
                if "error" in res:
                    print(f"       {model:14s}  ERROR: {res['error']}")
                    continue
                met  = res["metrics"]
                line = f"       {model:14s}  " + \
                       "  ".join(f"{k}={v:.4f}" for k, v in met.items()
                                 if isinstance(v, float))
                print(line)

            plot_holdout_bar(h_res, target, task, version)
            plot_best_holdout(h_res, target, task, version)
            plot_feature_importance(h_res, target, task, version)
            if target == "target_large_move":
                plot_pr_combined(h_res, target, version)

    # ── version comparison plots ──────────────────────────────────────────────
    plot_version_comparison(all_holdout, REG_TARGETS, "regression")
    plot_version_comparison(all_holdout, CLF_TARGETS, "classification")

    # ── save CSVs ─────────────────────────────────────────────────────────────
    all_cv_df = pd.concat(all_cv_rows, ignore_index=True)
    all_cv_df.to_csv(OUT_DIR / "cv_all_results.csv", index=False)
    print(f"\n  ✓ cv_all_results.csv")

    holdout_rows = []
    for (tgt, ver), model_res in all_holdout.items():
        for model, res in model_res.items():
            if "error" in res:
                continue
            row = {"target": tgt, "version": ver, "model": model}
            row.update(res.get("metrics", {}))
            holdout_rows.append(row)
    holdout_df = pd.DataFrame(holdout_rows)
    holdout_df.to_csv(OUT_DIR / "holdout_results.csv", index=False)
    print(f"  ✓ holdout_results.csv")

    # ── beat-the-baseline summary ─────────────────────────────────────────────
    print("\n\n" + "═"*70)
    print("  DOES THE MODEL BEAT THE NAIVE BASELINE?  (holdout, Version A)")
    print("═"*70)

    for task, targets in [("regression", REG_TARGETS), ("classification", CLF_TARGETS)]:
        key = "ic" if task == "regression" else "f1_w"
        higher_is_better = (key != "rmse")
        print(f"\n── {task.upper()} — primary metric: {key} ──")

        for tgt in targets:
            res_A = all_holdout.get((tgt, "A"), {})
            bl_val = res_A.get("baseline", {}).get("metrics", {}).get(key)
            if bl_val is None:
                continue
            print(f"\n  {tgt}  (baseline {key}={bl_val:.4f})")
            for model, res in res_A.items():
                if model == "baseline" or "error" in res:
                    continue
                mv = res.get("metrics", {}).get(key)
                if mv is None:
                    continue
                beat = mv > bl_val if higher_is_better else mv < bl_val
                sym  = "✓ BEATS" if beat else "✗ LOSES"
                diff = mv - bl_val
                print(f"    {sym}  {model:14s}  {key}={mv:.4f}  (Δ={diff:+.4f})")

        # also print version A vs B delta
        print(f"\n  Version B uplift over A (Δ{key}, holdout):")
        for tgt in targets:
            for model in ["xgb", "lgb", "rf"]:
                va = all_holdout.get((tgt, "A"), {}).get(model, {}).get("metrics", {}).get(key)
                vb = all_holdout.get((tgt, "B"), {}).get(model, {}).get("metrics", {}).get(key)
                if va is not None and vb is not None:
                    print(f"    {tgt:25s}  {model:8s}  A={va:.4f}  B={vb:.4f}  Δ={vb-va:+.4f}")

    print(f"\n=== supervised complete — {OUT_DIR} ===\n")


if __name__ == "__main__":
    main()
