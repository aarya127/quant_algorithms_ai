"""
pipeline.py — Walk-forward fold runner and final holdout evaluation.
"""
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Lasso
from imblearn.over_sampling import SMOTE

from config import LASSO_ALPHA, SMOTE_K, BINARY_CLF, MULTI_CLF, USE_TUNING, TUNE_ITER, TUNE_SPLITS
from data import prepare_fold_data
from baselines import target_baseline
from metrics import reg_metrics, clf_metrics
from models import (reg_model_set, clf_model_set, _get_feature_importances,
                    reg_param_dists, clf_param_dists, _tune_scoring, tune_model)


# ─────────────────────────────────────────────────────────────────────────────
# fold runner (cross-validation)
# ─────────────────────────────────────────────────────────────────────────────
def run_folds(df, folds, features, target, task):
    """
    Walk-forward cross-validation across all folds.

    Returns:
        fold_records: list of dicts (one per model per fold)
        feat_counter: dict of {feature: n_folds_selected_by_lasso}
    """
    use_smote     = (target == "target_large_move")
    is_binary_imb = (target in BINARY_CLF)
    is_multi      = (target in MULTI_CLF)

    fold_records = []
    feat_counter = defaultdict(int)
    param_dists  = (reg_param_dists() if task == "regression"
                    else clf_param_dists()) if USE_TUNING else {}
    scoring      = _tune_scoring(task, target) if USE_TUNING else None

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
                    if USE_TUNING and name in param_dists:
                        m, _ = tune_model(name, m, param_dists[name],
                                          X_tr, y_tr, scoring, TUNE_ITER, TUNE_SPLITS)
                    else:
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
            sw_m = {c: len(y_tr_enc) / (n_cls * cnt + 1e-9) for c, cnt in lc.items()}
            sw   = np.array([sw_m.get(int(c), 1.0) for c in y_tr_enc])

            y_bl_i = np.clip(y_bl.astype(int),
                             int(le.classes_.min()), int(le.classes_.max()))
            fold_records.append({"fold": fi, "model": "baseline",
                                 **clf_metrics(y_val_orig, y_bl_i, None, target)})

            for name, m in clf_model_set(n_cls, is_binary_imb, spw, is_multi).items():
                try:
                    if USE_TUNING and name in param_dists:
                        y_fit = y_tr_enc if name in ("xgb", "lgb") else y_tr.astype(int)
                        m, _ = tune_model(name, m, param_dists[name],
                                          X_tr, y_fit, scoring, TUNE_ITER, TUNE_SPLITS)
                    elif name in ("xgb", "lgb"):
                        m.fit(X_tr, y_tr_enc, sample_weight=sw)
                    else:
                        m.fit(X_tr, y_tr.astype(int))
                    if name in ("xgb", "lgb"):
                        yhat = le.inverse_transform(
                            np.clip(m.predict(X_val).astype(int), 0, n_cls - 1))
                        proba = m.predict_proba(X_val)
                    else:
                        yhat  = m.predict(X_val).astype(int)
                        proba = m.predict_proba(X_val) if hasattr(m, "predict_proba") else None
                    fold_records.append({"fold": fi, "model": name,
                                         **clf_metrics(y_val_orig, yhat, proba, target)})
                except Exception as e:
                    fold_records.append({"fold": fi, "model": name, "error": str(e)})

        km = "ic" if task == "regression" else "f1_w"
        parts = [f"fold {fi+1}/{len(folds)}  val={val_idx[0]}–{val_idx[-1]}"]
        for r in fold_records:
            if r["fold"] == fi and km in r:
                parts.append(f"{r['model']}={r[km]:.4f}")
        print("    " + "  ".join(parts))

    return fold_records, feat_counter


# ─────────────────────────────────────────────────────────────────────────────
# final refit + holdout evaluation
# ─────────────────────────────────────────────────────────────────────────────
def run_holdout(df, cv_idx, holdout_idx, features, target, task):
    """
    Refit each model on the full CV set, evaluate on the holdout.

    All preprocessing is fit on cv_idx only (no leakage into holdout).

    Returns: dict of {model_name: {"metrics": ..., "y_true": ..., ...}}
    """
    use_smote     = (target == "target_large_move")
    is_binary_imb = (target in BINARY_CLF)
    is_multi      = (target in MULTI_CLF)

    # ── preprocessing: fit on CV, apply to holdout ────────────────────────────
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

    results     = {}
    param_dists = (reg_param_dists() if task == "regression"
                   else clf_param_dists()) if USE_TUNING else {}
    h_scoring   = _tune_scoring(task, target) if USE_TUNING else None

    # ── regression ─────────────────────────────────────────────────────────────
    if task == "regression":
        y_bl = target_baseline(y_all, len(y_h_ev), target)
        results["baseline"] = {"metrics": reg_metrics(y_h_ev, y_bl),
                               "y_true": y_h_ev, "y_pred": y_bl}
        for name, m in reg_model_set().items():
            try:
                if USE_TUNING and name in param_dists:
                    m, _ = tune_model(name, m, param_dists[name],
                                      X_cv, y_all, h_scoring, TUNE_ITER, TUNE_SPLITS)
                else:
                    m.fit(X_cv, y_all)
                yhat = m.predict(X_h_ev)
                results[name] = {
                    "metrics":     reg_metrics(y_h_ev, yhat),
                    "y_true":      y_h_ev,
                    "y_pred":      yhat,
                    "importances": _get_feature_importances(m, name, sel_features),
                }
            except Exception as e:
                results[name] = {"error": str(e)}

    # ── classification ────────────────────────────────────────────────────────
    else:
        le       = LabelEncoder()
        y_cv_enc = le.fit_transform(y_all.astype(int))
        n_cls    = len(le.classes_)

        lc   = pd.Series(y_cv_enc).value_counts()
        spw  = float(lc.get(0, 1) / max(lc.get(1, 1), 1)) if is_binary_imb else 1.0
        sw_m = {c: len(y_cv_enc) / (n_cls * cnt + 1e-9) for c, cnt in lc.items()}
        sw   = np.array([sw_m.get(int(c), 1.0) for c in y_cv_enc])

        # SMOTE on full CV set (train-only for the holdout split)
        y_cv_orig_fit = y_all.astype(int)
        if use_smote:
            n_min = int(pd.Series(y_cv_enc).value_counts().min())
            k = min(SMOTE_K, n_min - 1)
            if k >= 1:
                try:
                    sm = SMOTE(k_neighbors=k, random_state=42)
                    X_cv, y_cv_enc = sm.fit_resample(X_cv, y_cv_enc)
                    y_cv_orig_fit = le.inverse_transform(y_cv_enc)
                    sw = None
                except Exception:
                    pass

        y_bl   = target_baseline(y_all, len(y_h_ev), target)
        y_bl_i = np.clip(y_bl.astype(int), int(le.classes_.min()), int(le.classes_.max()))
        results["baseline"] = {
            "metrics": clf_metrics(y_h_ev.astype(int), y_bl_i, None, target),
            "y_true":  y_h_ev,
            "y_pred":  y_bl_i.astype(float),
        }

        for name, m in clf_model_set(n_cls, is_binary_imb, spw, is_multi).items():
            try:
                if USE_TUNING and name in param_dists:
                    y_fit = y_cv_enc if name in ("xgb", "lgb") else y_cv_orig_fit
                    m, _ = tune_model(name, m, param_dists[name],
                                      X_cv, y_fit, h_scoring, TUNE_ITER, TUNE_SPLITS)
                elif name in ("xgb", "lgb"):
                    m.fit(X_cv, y_cv_enc, sample_weight=sw)
                else:
                    m.fit(X_cv, y_cv_orig_fit)
                if name in ("xgb", "lgb"):
                    yhat_enc = m.predict(X_h_ev).astype(int)
                    yhat  = le.inverse_transform(np.clip(yhat_enc, 0, n_cls - 1)).astype(float)
                    proba = m.predict_proba(X_h_ev)
                else:
                    yhat  = m.predict(X_h_ev).astype(float)
                    proba = m.predict_proba(X_h_ev) if hasattr(m, "predict_proba") else None
                results[name] = {
                    "metrics":     clf_metrics(y_h_ev.astype(int), yhat.astype(int), proba, target),
                    "y_true":      y_h_ev,
                    "y_pred":      yhat,
                    "proba":       proba,
                    "importances": _get_feature_importances(m, name, sel_features),
                }
            except Exception as e:
                results[name] = {"error": str(e)}

    return results
