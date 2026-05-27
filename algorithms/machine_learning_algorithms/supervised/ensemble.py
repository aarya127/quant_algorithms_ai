"""
ensemble.py — Stage 13A: mean, weighted, and regime-aware prediction ensembles.

Called from main.py after all holdout evaluations are complete.

Inputs : all_holdout  {(target, version): {model_name: result}}
         df, holdout_idx, ticker, out_dir
Outputs: output/predictions/{ticker}_ensemble_predictions.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

# ── base models included in each ensemble ─────────────────────────────────────
_REG_MODELS = ["ridge", "xgb", "lgb"]
_CLF_MODELS = ["logistic", "xgb", "lgb"]

# ── static weights (Stage 13A — no Sharpe weighting yet) ─────────────────────
#   calm regime  (regime == 0) → trust ML more  (XGB/LGB upweighted)
#   stress regime (regime != 0) → trust conservative model (Ridge upweighted)
_W_CALM   = {"ridge": 0.20, "xgb": 0.40, "lgb": 0.40}
_W_STRESS = {"ridge": 0.40, "xgb": 0.30, "lgb": 0.30}

# ── weighted ensemble (static, regime-agnostic) ───────────────────────────────
_W_STATIC = {"ridge": 0.20, "xgb": 0.40, "lgb": 0.40}


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _expand(y_subset, valid_mask):
    """Expand a valid-row-only array back to full holdout_n (NaN elsewhere)."""
    out = np.full(len(valid_mask), np.nan)
    out[valid_mask] = y_subset
    return out


def _get_pred(all_holdout, target, version, model, valid_mask):
    """Return a full-length float array (NaN where invalid), or None."""
    res = all_holdout.get((target, version), {}).get(model, {})
    if "y_pred" not in res:
        return None
    yp = np.asarray(res["y_pred"], dtype=float)
    n_valid = int(valid_mask.sum())
    full_n  = len(valid_mask)
    if len(yp) == n_valid:
        return _expand(yp, valid_mask)
    if len(yp) == full_n:
        return yp
    return None


def _get_proba(all_holdout, target, version, model, valid_mask):
    """Return (full_n, n_classes) proba array or None."""
    res = all_holdout.get((target, version), {}).get(model, {})
    if res.get("proba") is None:
        return None
    p = np.asarray(res["proba"])
    n_valid = int(valid_mask.sum())
    full_n  = len(valid_mask)
    if p.shape[0] == n_valid:
        out = np.full((full_n, p.shape[1]), np.nan)
        out[valid_mask] = p
        return out
    if p.shape[0] == full_n:
        return p
    return None


def _weighted_avg(preds, weights):
    """Weighted mean of aligned arrays; preds = {model: array}."""
    total_w = 0.0
    result  = None
    for m, w in weights.items():
        if m in preds and preds[m] is not None:
            arr    = np.asarray(preds[m], dtype=float)
            result = arr * w if result is None else result + arr * w
            total_w += w
    return result / total_w if (result is not None and total_w > 0) else None


def _vmask(df, holdout_idx, target):
    """Valid-row boolean mask for a target column."""
    if target in df.columns:
        return ~df.loc[holdout_idx, target].isna().values
    return np.ones(len(holdout_idx), dtype=bool)


# ─────────────────────────────────────────────────────────────────────────────
# main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_ensemble(all_holdout, df, holdout_idx, ticker, out_dir):
    """
    Build Stage 13A ensembles from holdout predictions and save to CSV.

    Parameters
    ----------
    all_holdout : dict  {(target, version): {model_name: result_dict}}
    df          : full DataFrame (provides dates + actual target values)
    holdout_idx : list[int]
    ticker      : str
    out_dir     : Path-like

    Returns
    -------
    pd.DataFrame  (also saved to out_dir/predictions/{ticker}_ensemble_predictions.csv)
    """
    pred_dir = Path(out_dir) / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    full_n = len(holdout_idx)
    dates  = (df.loc[holdout_idx, "Date"].values
              if "Date" in df.columns else np.arange(full_n))
    rows   = {"date": dates, "ticker": ticker}

    # ── prefer version B (includes regime features) ───────────────────────────
    version = "B"

    # ── Step 1: regime predictions (needed for regime-aware ensemble) ─────────
    regime_vm   = _vmask(df, holdout_idx, "target_regime")
    regime_full = None
    for m in ["xgb", "lgb", "rf"]:
        cand = _get_pred(all_holdout, "target_regime", version, m, regime_vm)
        if cand is None:
            cand = _get_pred(all_holdout, "target_regime", "A", m, regime_vm)
        if cand is not None:
            regime_full = np.nan_to_num(cand, nan=0).astype(int)
            rows["pred_regime"] = regime_full
            break

    # ── Step 2: regression ensembles ─────────────────────────────────────────
    for target in ["target_1d", "target_5d", "target_vol_5d"]:
        suffix = target.replace("target_", "")   # "1d", "5d", "vol_5d"
        vm     = _vmask(df, holdout_idx, target)

        # collect individual model predictions
        preds = {}
        for m in _REG_MODELS:
            p = _get_pred(all_holdout, target, version, m, vm)
            if p is None:
                p = _get_pred(all_holdout, target, "A", m, vm)
            if p is not None:
                preds[m] = p
                rows[f"pred_{m}_{suffix}"] = p

        if len(preds) < 2:
            continue

        # Ensemble 1 — simple mean
        stack = np.vstack([preds[m] for m in _REG_MODELS if m in preds])
        rows[f"pred_ensemble_mean_{suffix}"] = np.nanmean(stack, axis=0)

        # Ensemble 2 — static weighted average
        w = _weighted_avg(preds, _W_STATIC)
        if w is not None:
            rows[f"pred_ensemble_weighted_{suffix}"] = w

        # Ensemble 3 — regime-aware weighted average
        if regime_full is not None:
            calm_w   = _weighted_avg(preds, _W_CALM)
            stress_w = _weighted_avg(preds, _W_STRESS)
            if calm_w is not None and stress_w is not None:
                rows[f"pred_ensemble_regime_{suffix}"] = np.where(
                    regime_full == 0, calm_w, stress_w)

        # actuals (for metric computation in summary table)
        if target in df.columns:
            rows[f"actual_{suffix}"] = df.loc[holdout_idx, target].values

    # ── Step 3: classification probability ensembles ──────────────────────────
    for target, suffix in [("target_dir_1d", "dir_1d"),
                            ("target_large_move", "large_move")]:
        vm     = _vmask(df, holdout_idx, target)
        probas = []
        for m in _CLF_MODELS:
            p = _get_proba(all_holdout, target, version, m, vm)
            if p is None:
                p = _get_proba(all_holdout, target, "A", m, vm)
            if p is not None:
                probas.append(p)

        if not probas:
            continue
        avg_p = np.nanmean(np.stack(probas, axis=0), axis=0)
        rows[f"pred_ensemble_{suffix}"] = np.nanargmax(avg_p, axis=1).astype(float)
        if avg_p.shape[1] >= 2:
            rows[f"proba_ensemble_{suffix}"] = avg_p[:, 1]

    # ── Step 4: summary comparison table ─────────────────────────────────────
    print(f"\n{'═'*66}")
    print(f"  ENSEMBLE PREDICTIONS — {ticker}  (holdout n={full_n})")
    print(f"{'═'*66}")

    for target in ["target_1d", "target_5d", "target_vol_5d"]:
        suffix = target.replace("target_", "")
        actual_key = f"actual_{suffix}"
        if actual_key not in rows:
            continue
        act   = np.asarray(rows[actual_key], dtype=float)
        valid = ~np.isnan(act)
        if valid.sum() < 5:
            continue

        print(f"\n  {target}  (n_valid={valid.sum()}):")
        cols_to_show = (
            [f"pred_{m}_{suffix}" for m in _REG_MODELS] +
            [f"pred_ensemble_mean_{suffix}",
             f"pred_ensemble_weighted_{suffix}",
             f"pred_ensemble_regime_{suffix}"]
        )
        for col in cols_to_show:
            if col not in rows:
                continue
            pred = np.asarray(rows[col], dtype=float)
            v2   = valid & ~np.isnan(pred)
            if v2.sum() < 5:
                continue
            ic, _    = spearmanr(act[v2], pred[v2])
            dir_acc  = float(np.mean(np.sign(act[v2]) == np.sign(pred[v2])))
            label    = col.replace(f"_{suffix}", "").replace("pred_", "")
            ic_val   = ic if not np.isnan(ic) else 0.0
            print(f"    {label:28s}  IC={ic_val:+.4f}  dir_acc={dir_acc:.3f}")

    # ── Step 5: align + save CSV ──────────────────────────────────────────────
    aligned = {}
    for k, v in rows.items():
        arr = np.asarray(v)
        if arr.ndim == 1 and len(arr) == full_n:
            aligned[k] = arr
        elif arr.ndim == 0:
            aligned[k] = np.full(full_n, arr.item())

    ensemble_df = pd.DataFrame(aligned)
    out_path    = pred_dir / f"{ticker}_ensemble_predictions.csv"
    ensemble_df.to_csv(out_path, index=False)
    print(f"\n  ✓ {out_path.name}")
    print(f"{'═'*66}")

    return ensemble_df
