"""
mlflow_tracker.py — MLflow integration for the supervised pipeline.

Tracking store  : file://<repo>/mlruns/   (local, no server required)
Experiment      : one per ticker  (e.g. "NVDA")
Run             : one per (target, version, date)
                  e.g. "target_5d_vB_2026-05-30"

What is logged per run
----------------------
Tags    : ticker, target, version, task
Params  : init_train, step, holdout_rows, lasso_alpha, use_tuning,
          n_features, best_model
Metrics : cv_{model}_{metric}_mean / _std   — cross-validation summary
          holdout_{model}_{metric}           — holdout evaluation per model
          best_{metric}                      — best model's primary metric
Artifacts:
          features/{target}_v{version}_features.json   — Lasso-selected features
          importance/{target}_v{version}_importance.json — top-20 feature weights

To launch the MLflow UI:
    mlflow ui --backend-store-uri <repo>/mlruns --port 5002
    then open http://localhost:5002
"""

import os
import json
import tempfile
import datetime
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import mlflow
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

from config import INIT_TRAIN, STEP, HOLDOUT_ROWS, LASSO_ALPHA, USE_TUNING

# SQLite tracking store — required for MLflow 3.x traces/datasets/scorers features.
_REPO_ROOT   = Path(__file__).resolve().parents[3]   # quant_algorithms_ai/
TRACKING_URI = f"sqlite:///{_REPO_ROOT / 'mlflow.db'}"


def setup_experiment(ticker: str) -> bool:
    """
    Point MLflow at the local store and activate the ticker experiment.
    Returns True if MLflow is available, False otherwise (pipeline still runs).
    """
    if not _AVAILABLE:
        print("[mlflow] not installed — skipping experiment tracking")
        return False
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(ticker)
    return True


def log_run(
    ticker: str,
    target: str,
    version: str,
    task: str,
    fold_records: list,
    holdout_res: dict,
    sel_features: list,
) -> None:
    """
    Log one (target, version) trial to the active MLflow experiment.

    Parameters
    ----------
    ticker       : str   e.g. "NVDA"
    target       : str   e.g. "target_5d"
    version      : str   "A" or "B"
    task         : str   "regression" | "classification"
    fold_records : list  rows from run_folds()
    holdout_res  : dict  {model_name: result} from run_holdout()
    sel_features : list  Lasso-selected feature names
    """
    if not _AVAILABLE:
        return

    primary  = "ic" if task == "regression" else "f1_w"
    date_tag = datetime.date.today().isoformat()
    run_name = f"{target}_v{version}_{date_tag}"

    with mlflow.start_run(run_name=run_name):

        # ── Tags ──────────────────────────────────────────────────────────────
        mlflow.set_tags({
            "ticker":  ticker,
            "target":  target,
            "version": version,
            "task":    task,
        })

        # ── Params ────────────────────────────────────────────────────────────
        mlflow.log_params({
            "init_train":   INIT_TRAIN,
            "step":         STEP,
            "holdout_rows": HOLDOUT_ROWS,
            "lasso_alpha":  LASSO_ALPHA,
            "use_tuning":   int(USE_TUNING),
            "n_features":   len(sel_features),
        })

        # Dataset tracking — log the feature matrix so every run records what data trained it.
        # Visible in MLflow UI under Evaluation > Datasets.
        _features_csv = Path(__file__).resolve().parents[1] / "data_pipelines" / f"{ticker}_features_with_regimes.csv"
        if _features_csv.exists():
            try:
                _df_feat = pd.read_csv(_features_csv, index_col=0)
                _ds = mlflow.data.from_pandas(
                    _df_feat,
                    source=str(_features_csv),
                    name=f"{ticker}_features_with_regimes",
                    targets=target,
                )
                mlflow.log_input(_ds, context="training")
            except Exception:
                pass  # dataset logging is optional; never block a training run

        # ── CV metrics: mean & std of primary metric per model ────────────────
        fd = pd.DataFrame(fold_records)
        if not fd.empty and primary in fd.columns:
            for model_name, grp in fd.groupby("model"):
                vals = grp[primary].dropna()
                if len(vals) > 0:
                    mlflow.log_metric(
                        f"cv_{model_name}_{primary}_mean",
                        round(float(vals.mean()), 6),
                    )
                    if len(vals) > 1:
                        mlflow.log_metric(
                            f"cv_{model_name}_{primary}_std",
                            round(float(vals.std()), 6),
                        )

        # ── Holdout metrics per model ─────────────────────────────────────────
        best_model_name = None
        best_val        = -np.inf
        for model_name, res in holdout_res.items():
            if model_name.startswith("_") or "error" in res:
                continue
            for k, v in res.get("metrics", {}).items():
                if isinstance(v, float) and not np.isnan(v):
                    mlflow.log_metric(f"holdout_{model_name}_{k}", round(v, 6))
            val = res.get("metrics", {}).get(primary, -np.inf)
            if isinstance(val, float) and not np.isnan(val) and val > best_val:
                best_val        = val
                best_model_name = model_name

        if best_model_name is not None:
            mlflow.log_param("best_model", best_model_name)
            mlflow.log_metric(f"best_{primary}", round(float(best_val), 6))

            # Evaluation run — fills MLflow UI Evaluation > Evaluation runs.
            # Runs the default tabular evaluator on the best model's holdout predictions.
            _best_res = holdout_res.get(best_model_name, {})
            _y_true = _best_res.get("y_true")
            _y_pred = _best_res.get("y_pred")
            if _y_true is not None and _y_pred is not None and len(_y_true) > 0:
                try:
                    _eval_df = pd.DataFrame({
                        "prediction": _y_pred.astype(float),
                        "target":     _y_true.astype(float),
                    })
                    _mtype = "regressor" if task == "regression" else "classifier"
                    mlflow.evaluate(
                        data=_eval_df,
                        targets="target",
                        predictions="prediction",
                        model_type=_mtype,
                        evaluator_config={"log_model_explainability": False},
                    )
                except Exception:
                    pass  # evaluation logging is optional; never block a training run

        # ── Artifact: selected feature list ───────────────────────────────────
        prefix = f"{target}_v{version}"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix=prefix + "_features_", delete=False
        ) as fh:
            json.dump(sel_features, fh, indent=2)
            feat_tmp = fh.name
        mlflow.log_artifact(feat_tmp, artifact_path="features")
        os.unlink(feat_tmp)

        # ── Artifact: feature importance top-20 per model ─────────────────────
        importance: dict = {}
        for model_name, res in holdout_res.items():
            if model_name.startswith("_") or "error" in res:
                continue
            imp = res.get("importances")
            if imp:
                importance[model_name] = dict(
                    sorted(imp.items(), key=lambda x: -abs(x[1]))[:20]
                )
        if importance:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", prefix=prefix + "_importance_", delete=False
            ) as fh:
                json.dump(importance, fh, indent=2)
                imp_tmp = fh.name
            mlflow.log_artifact(imp_tmp, artifact_path="importance")
            os.unlink(imp_tmp)
