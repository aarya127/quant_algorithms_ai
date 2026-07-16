"""
main.py — Main entry point for the walk-forward supervised pipeline.
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from pathlib import Path

from config import (
    SYMBOL, REGIMES_CSV, FEAT_FILE, HOLDOUT_ROWS, INIT_TRAIN, STEP,
    REGIME_COLS, REG_TARGETS, CLF_TARGETS, OUT_DIR, target_embargo,
)
from data import make_walk_forward_splits
from pipeline import run_folds, run_holdout
from plots import (
    plot_cv_primary, plot_feat_freq, plot_holdout_bar,
    plot_best_holdout, plot_feature_importance,
    plot_pr_combined, plot_version_comparison,
)
from ensemble import run_ensemble
from registry import save_registry
from mlflow_tracker import setup_experiment, log_run

_REGISTRY_DIR = Path(__file__).parent / "model_registry"


def main():
    print(f"\n=== supervised: {SYMBOL} ===")
    setup_experiment(SYMBOL)

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
    all_holdout = {}   # {(target, version): {model_name: result}}

    for version, features in [("A", feat_A), ("B", feat_B)]:
        print(f"\n{'═'*62}")
        print(f"  VERSION {version}  ({len(features)} features)")
        print(f"{'═'*62}")

        for target in REG_TARGETS + CLF_TARGETS:
            task = "regression" if target in REG_TARGETS else "classification"
            # Per-target purge gap (= label horizon) prevents look-ahead leakage
            # from the forward-looking target into validation/holdout.
            embargo  = target_embargo(target)
            folds_t  = make_walk_forward_splits(n, INIT_TRAIN, STEP, HOLDOUT_ROWS, embargo=embargo)
            cv_idx_t = list(range(0, n - HOLDOUT_ROWS - embargo))
            print(f"\n── {target}  [{task}]  v{version} ── embargo={embargo} ──────────")

            fold_records, feat_counter = run_folds(df, folds_t, features, target, task)

            fd = pd.DataFrame(fold_records)
            fd.insert(0, "version", version)
            fd.insert(1, "target", target)
            all_cv_rows.append(fd)

            plot_cv_primary(fd, target, task, version)
            if version == "A":
                plot_feat_freq(feat_counter, target, version)

            print(f"    → Holdout ({HOLDOUT_ROWS} rows):")
            h_res = run_holdout(df, cv_idx_t, holdout_idx, features, target, task)
            all_holdout[(target, version)] = h_res

            # Log params, CV metrics, holdout metrics, and feature artifacts to MLflow
            sel_features = h_res.get("_meta", {}).get("sel_features", features)
            log_run(SYMBOL, target, version, task, fold_records, h_res, sel_features)

            for model, res in h_res.items():
                if model.startswith("_"):
                    continue
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

    # summary plots 
    plot_version_comparison(all_holdout, REG_TARGETS, "regression")
    plot_version_comparison(all_holdout, CLF_TARGETS, "classification")

    # save CSVs
    all_cv_df = pd.concat(all_cv_rows, ignore_index=True)
    all_cv_df.to_csv(OUT_DIR / "cv_all_results.csv", index=False)
    print(f"\n  ✓ cv_all_results.csv")

    holdout_rows = []
    for (tgt, ver), model_res in all_holdout.items():
        for model, res in model_res.items():
            if model.startswith("_") or "error" in res:
                continue
            row = {"target": tgt, "version": ver, "model": model}
            row.update(res.get("metrics", {}))
            holdout_rows.append(row)
    holdout_df = pd.DataFrame(holdout_rows)
    holdout_df.to_csv(OUT_DIR / "holdout_results.csv", index=False)
    print(f"  ✓ holdout_results.csv")

    # Stage 13A: ensemble layer 
    run_ensemble(all_holdout, df, holdout_idx, SYMBOL, OUT_DIR)

    # Stage 14A: local model registry
    save_registry(all_holdout, SYMBOL, _REGISTRY_DIR)

    # beat-the-baseline summary
    print("\n\n" + "═"*70)
    print("  DOES THE MODEL BEAT THE NAIVE BASELINE?  (holdout, Version A)")
    print("═"*70)

    for task, targets in [("regression", REG_TARGETS), ("classification", CLF_TARGETS)]:
        key = "ic" if task == "regression" else "f1_w"
        print(f"\n── {task.upper()} — primary metric: {key} ──")
        for tgt in targets:
            res_A  = all_holdout.get((tgt, "A"), {})
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
                sym  = "✓ BEATS" if mv > bl_val else "✗ LOSES"
                diff = mv - bl_val
                print(f"    {sym}  {model:14s}  {key}={mv:.4f}  (Δ={diff:+.4f})")

        print(f"\n  Version B uplift over A (Δ{key}, holdout):")
        for tgt in targets:
            for model in ["xgb", "lgb", "rf"]:
                va = all_holdout.get((tgt, "A"), {}).get(model, {}).get("metrics", {}).get(key)
                vb = all_holdout.get((tgt, "B"), {}).get(model, {}).get("metrics", {}).get(key)
                if va is not None and vb is not None:
                    print(f"    {tgt:25s}  {model:8s}  A={va:.4f}  B={vb:.4f}  Δ={vb-va:+.4f}")

    print(f"\n=== supervised complete — {OUT_DIR} ===\n")
