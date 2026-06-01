"""
predictor.py — Stage 14A/B: serve predictions from the local model registry.

Used by backend/app.py to power:
  GET /api/predict/<ticker>
  GET /api/drift/<ticker>
  GET /api/model/status
"""

import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

_HERE        = Path(__file__).parent                          # backend/
_PROJECT     = _HERE.parent                                   # project root
_SUPERVISED  = _PROJECT / "algorithms" / "machine_learning_algorithms" / "supervised"
_PIPELINES   = _PROJECT / "algorithms" / "machine_learning_algorithms" / "data_pipelines"
_REGISTRY    = _SUPERVISED / "model_registry"
_MONITORING  = _SUPERVISED / "output" / "monitoring"

# add supervised/ to path so we can import registry.py
sys.path.insert(0, str(_SUPERVISED))
from registry import load_registry   # noqa: E402

_DRIFT_Z_THRESH = 3.0
_MODEL_DRIFT_IC_DROP = 0.10   # flag when rolling IC falls this far below registered IC
_MODEL_DRIFT_MIN_OBS = 10     # minimum realized pairs needed before computing rolling IC


# Alerting

def _notify(subject: str, body: dict) -> None:
    """
    POST an alert to ALERT_WEBHOOK_URL (Slack incoming webhook, Make.com, etc.).
    Fires on data drift, model drift, or pipeline failures.
    Silently no-ops when the env var is unset or the HTTP call fails.
    """
    url = os.environ.get("ALERT_WEBHOOK_URL", "").strip()
    if not url:
        return
    try:
        import requests as _req
        _req.post(url, json={"text": subject, **body}, timeout=5)
    except Exception:
        pass


# internal helpers

def _features_df(ticker: str) -> pd.DataFrame:
    """Load the latest features CSV for ticker."""
    path = _PIPELINES / f"{ticker}_features_with_regimes.csv"
    if not path.exists():
        raise FileNotFoundError(f"Features CSV not found: {path}")
    return pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)


def _preprocess_row(row_df: pd.DataFrame, reg: dict) -> np.ndarray:
    """
    Apply saved col_med_sel imputation → serving_scaler → return shaped array.

    serving_scaler is fit on the Lasso-selected raw columns only, so it accepts
    exactly len(sel_features) inputs — avoiding the shape mismatch that would
    occur if we used the full-feature scaler stored in scaler.pkl.

    row_df  : DataFrame with raw feature columns (1+ rows)
    reg     : dict from load_registry()
    returns : 2-D float array (n_rows, n_selected_features)
    """
    features       = reg["features"]          # Lasso-selected names
    col_med_sel    = reg.get("col_med_sel")   # medians for selected cols
    serving_scaler = reg.get("serving_scaler")

    # keep only features the model was trained on (in order)
    present = [f for f in features if f in row_df.columns]
    X = row_df[present].values.astype(float)

    # impute NaN with training medians for selected features
    if col_med_sel is not None:
        med = np.asarray(col_med_sel, dtype=float)
        for j in range(X.shape[1]):
            nans = np.isnan(X[:, j])
            if nans.any():
                fill = float(med[j]) if j < len(med) else 0.0
                X[nans, j] = fill
    else:
        X = np.nan_to_num(X, nan=0.0)

    if serving_scaler is not None:
        X = serving_scaler.transform(X)

    return X


# public API

def predict_latest(ticker: str) -> dict:
    """
    Generate predictions for the most recent row in the features CSV.

    Returns a JSON-serialisable dict:
    {
      ticker, date, model_version,
      predicted_1d_return, predicted_5d_return, predicted_vol_5d,
      predicted_regime, predicted_dir_1d,
      signal, confidence, anomaly_flag
    }
    """
    df   = _features_df(ticker)
    row  = df.iloc[[-1]]
    date = str(df.iloc[-1]["Date"])[:10]

    out = {
        "ticker":        ticker,
        "date":          date,
        "model_version": None,
        "predictions":   {},
    }

    _targets = {
        "target_1d":     "predicted_1d_return",
        "target_5d":     "predicted_5d_return",
        "target_vol_5d": "predicted_vol_5d",
        "target_regime": "predicted_regime",
        "target_dir_1d": "predicted_dir_1d",
    }

    for target, label in _targets.items():
        try:
            reg = load_registry(ticker, _REGISTRY, target=target)
        except FileNotFoundError:
            continue

        X     = _preprocess_row(row, reg)
        if X.shape[1] == 0:
            continue
        model = reg["model"]
        meta  = reg["metadata"]
        task  = meta.get("task", "regression")

        if task == "regression":
            out["predictions"][label] = round(float(model.predict(X)[0]), 6)
        else:
            le  = reg.get("label_encoder")
            raw = model.predict(X)[0]
            cls = int(le.inverse_transform([int(raw)])[0]) if le is not None else int(raw)
            out["predictions"][label] = cls
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                out["predictions"][f"{label}_proba"] = [round(float(p), 4) for p in proba]

        if out["model_version"] is None:
            out["model_version"] = f"{meta['model_type']}_{target}_v1"

    # derive signal + confidence
    ret5 = out["predictions"].get("predicted_5d_return")
    vol  = out["predictions"].get("predicted_vol_5d")

    if ret5 is not None:
        out["signal"] = "long" if ret5 > 0.01 else ("short" if ret5 < -0.01 else "neutral")
    else:
        out["signal"] = "neutral"

    if vol is not None:
        out["confidence"] = "high" if vol < 0.02 else ("low" if vol > 0.04 else "medium")
    else:
        out["confidence"] = "medium"

    out["anomaly_flag"] = int(
        df.iloc[-1].get("anomaly_iso", 0) == -1
        if "anomaly_iso" in df.columns else 0
    )

    # log the prediction
    _log_prediction(out)

    return out


def check_drift(ticker: str) -> dict:
    """
    Compare the latest feature row against training-time distribution.

    Returns:
    {
      date, ticker, n_features_checked, n_drift_flags,
      overall_status,   # "ok" | "warning"
      drift_flags: [{feature, latest_value, training_mean, z_score, status}, ...]
    }
    """
    df   = _features_df(ticker)
    row  = df.iloc[-1]
    date = str(row.get("Date", ""))[:10]

    try:
        reg = load_registry(ticker, _REGISTRY, target="target_5d")
    except FileNotFoundError:
        return {"error": "registry not found — run the supervised pipeline first",
                "ticker": ticker}

    train_stats  = reg.get("train_stats", {})
    drift_flags  = []

    for feat, stats in train_stats.items():
        if feat not in row.index:
            continue
        val = row[feat]
        if pd.isna(val):
            continue
        val  = float(val)
        mean = stats.get("mean", 0.0)
        std  = max(stats.get("std", 1e-9), 1e-9)
        p05  = stats.get("p05", val)
        p95  = stats.get("p95", val)

        z            = (val - mean) / std
        out_of_range = (val < p05 or val > p95)
        if abs(z) > _DRIFT_Z_THRESH or out_of_range:
            drift_flags.append({
                "feature":       feat,
                "latest_value":  round(val, 6),
                "training_mean": round(mean, 6),
                "training_std":  round(std, 6),
                "z_score":       round(z, 2),
                "in_p5_p95":     not out_of_range,
                "status":        "drift",
            })

    report = {
        "date":                date,
        "ticker":              ticker,
        "n_features_checked":  len(train_stats),
        "n_drift_flags":       len(drift_flags),
        "overall_status":      "ok" if not drift_flags else "warning",
        "drift_flags":         drift_flags[:25],   # cap for readability
    }

    # persist report
    _MONITORING.mkdir(parents=True, exist_ok=True)
    (_MONITORING / f"drift_report_{ticker}_latest.json").write_text(
        json.dumps(report, indent=2))

    if drift_flags:
        _notify(
            f"[{ticker}] Data drift detected: {len(drift_flags)} feature(s) out of range",
            report,
        )

    return report


def model_status() -> dict:
    """
    Return a summary of all registered models across all tickers.
    """
    if not _REGISTRY.exists():
        return {"error": "model_registry not found — run the supervised pipeline first"}

    status = {}
    for ticker_dir in sorted(_REGISTRY.iterdir()):
        if not ticker_dir.is_dir():
            continue
        active_path = ticker_dir / "active.json"
        if not active_path.exists():
            continue
        active = json.loads(active_path.read_text())
        targets = {}
        for tgt, info in active.get("targets", {}).items():
            meta_path = Path(info["path"]) / "metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                targets[tgt] = {
                    "model_type":   meta["model_type"],
                    "metric":       meta["primary_metric"],
                    "metric_value": meta["metric_value"],
                    "created_at":   meta["created_at"],
                }
        status[ticker_dir.name] = {
            "created_at": active.get("created_at"),
            "targets":    targets,
        }
    return status


# prediction logging

_PRED_LOG = _MONITORING / "prediction_log.csv"


def _log_prediction(pred: dict) -> None:
    """Append prediction dict to running prediction_log.csv."""
    preds   = pred.get("predictions", {})
    flat    = {
        "date":             pred.get("date"),
        "ticker":           pred.get("ticker"),
        "model_version":    pred.get("model_version"),
        "pred_1d_return":   preds.get("predicted_1d_return"),
        "pred_5d_return":   preds.get("predicted_5d_return"),
        "pred_vol_5d":      preds.get("predicted_vol_5d"),
        "pred_regime":      preds.get("predicted_regime"),
        "pred_dir_1d":      preds.get("predicted_dir_1d"),
        "signal":           pred.get("signal"),
        "confidence":       pred.get("confidence"),
        "anomaly_flag":     pred.get("anomaly_flag"),
        # actuals filled in later when returns are realized
        "actual_1d_return": None,
        "actual_5d_return": None,
    }
    _MONITORING.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame([flat])
    if _PRED_LOG.exists():
        existing = pd.read_csv(_PRED_LOG, dtype=str)
        dup_mask = ~(
            (existing["date"]   == str(flat["date"])) &
            (existing["ticker"] == str(flat["ticker"]))
        )
        combined = pd.concat([existing[dup_mask], row_df], ignore_index=True)
    else:
        combined = row_df
    combined.to_csv(_PRED_LOG, index=False)


# Model drift — backfill actuals + rolling IC

def backfill_actuals(ticker: str) -> int:
    """
    Fill in actual_1d_return / actual_5d_return in prediction_log.csv using
    realized log returns from the features CSV.

    Called automatically by compute_model_drift() and check_drift().
    Returns the number of rows updated.
    """
    if not _PRED_LOG.exists():
        return 0

    log = pd.read_csv(_PRED_LOG, dtype=str)
    if log.empty or "ticker" not in log.columns:
        return 0

    try:
        feat = _features_df(ticker)
    except FileNotFoundError:
        return 0

    if "log_return" not in feat.columns:
        return 0

    feat = feat[["Date", "log_return"]].copy()
    feat["Date"] = pd.to_datetime(feat["Date"])
    feat = feat.sort_values("Date").reset_index(drop=True)

    n_updated = 0
    for idx, row in log.iterrows():
        if row.get("ticker") != ticker:
            continue
        # skip rows already filled
        a1 = str(row.get("actual_1d_return", "")).strip()
        if a1 not in ("", "None", "nan"):
            continue

        try:
            pred_date = pd.Timestamp(row["date"])
        except Exception:
            continue

        future = feat[feat["Date"] > pred_date].reset_index(drop=True)
        if future.empty:
            continue   # actual not yet available

        actual_1d = float(future.iloc[0]["log_return"])
        log.at[idx, "actual_1d_return"] = round(actual_1d, 6)

        if len(future) >= 5:
            actual_5d = float(future.iloc[:5]["log_return"].sum())
            log.at[idx, "actual_5d_return"] = round(actual_5d, 6)

        n_updated += 1

    if n_updated > 0:
        log.to_csv(_PRED_LOG, index=False)

    return n_updated


def compute_model_drift(ticker: str, window: int = 20) -> dict:
    """
    Compute rolling Spearman IC on realized prediction-actual pairs.

    Returns:
    {
      ticker, n_observations, n_with_actuals, rolling_window,
      rolling_ic_1d, rolling_ic_5d,
      registered_ic_1d, registered_ic_5d,   # IC stored in model registry
      drift_status,   # "ok" | "warning" | "degraded" | "insufficient_data"
      message
    }
    """
    backfill_actuals(ticker)   # always refresh before computing

    if not _PRED_LOG.exists():
        return {"ticker": ticker, "drift_status": "insufficient_data",
                "message": "No prediction log found"}

    log = pd.read_csv(_PRED_LOG, dtype=str)
    tlog = log[log["ticker"] == ticker].copy()

    for col in ["actual_1d_return", "actual_5d_return",
                "pred_1d_return", "pred_5d_return"]:
        tlog[col] = pd.to_numeric(tlog[col], errors="coerce")

    filled_1d = tlog.dropna(subset=["pred_1d_return", "actual_1d_return"])
    filled_5d = tlog.dropna(subset=["pred_5d_return", "actual_5d_return"])
    n_with_actuals = len(filled_1d)

    rolling_ic_1d = rolling_ic_5d = None

    if len(filled_1d) >= _MODEL_DRIFT_MIN_OBS:
        r1d = filled_1d.tail(window)
        ic, _ = spearmanr(r1d["pred_1d_return"], r1d["actual_1d_return"])
        rolling_ic_1d = round(float(ic), 4)

    if len(filled_5d) >= _MODEL_DRIFT_MIN_OBS:
        r5d = filled_5d.tail(window)
        ic, _ = spearmanr(r5d["pred_5d_return"], r5d["actual_5d_return"])
        rolling_ic_5d = round(float(ic), 4)

    # pull registered ICs from model registry
    registered_ic_1d = registered_ic_5d = None
    for tgt, attr in [("target_1d", "registered_ic_1d"),
                      ("target_5d", "registered_ic_5d")]:
        try:
            reg = load_registry(ticker, _REGISTRY, target=tgt)
            if attr == "registered_ic_1d":
                registered_ic_1d = reg["metadata"].get("metric_value")
            else:
                registered_ic_5d = reg["metadata"].get("metric_value")
        except FileNotFoundError:
            pass

    # classify drift status
    if n_with_actuals < _MODEL_DRIFT_MIN_OBS:
        drift_status = "insufficient_data"
        message = f"Only {n_with_actuals} realized pairs (need ≥{_MODEL_DRIFT_MIN_OBS})"
    else:
        drop_flags = []
        if rolling_ic_1d is not None and registered_ic_1d is not None:
            if rolling_ic_1d < registered_ic_1d - _MODEL_DRIFT_IC_DROP:
                drop_flags.append(
                    f"1d IC {rolling_ic_1d:.3f} vs registered {registered_ic_1d:.3f}")
        if rolling_ic_5d is not None and registered_ic_5d is not None:
            if rolling_ic_5d < registered_ic_5d - _MODEL_DRIFT_IC_DROP:
                drop_flags.append(
                    f"5d IC {rolling_ic_5d:.3f} vs registered {registered_ic_5d:.3f}")

        if drop_flags:
            drift_status = "degraded"
            message = "Performance degraded: " + "; ".join(drop_flags)
        elif rolling_ic_1d is not None and rolling_ic_1d < 0:
            drift_status = "warning"
            message = f"Negative rolling 1d IC ({rolling_ic_1d:.3f})"
        else:
            drift_status = "ok"
            message = "Model performance within expected range"

    report = {
        "ticker":           ticker,
        "n_observations":   len(tlog),
        "n_with_actuals":   n_with_actuals,
        "rolling_window":   window,
        "rolling_ic_1d":    rolling_ic_1d,
        "rolling_ic_5d":    rolling_ic_5d,
        "registered_ic_1d": registered_ic_1d,
        "registered_ic_5d": registered_ic_5d,
        "drift_status":     drift_status,
        "message":          message,
    }

    _MONITORING.mkdir(parents=True, exist_ok=True)
    (_MONITORING / f"model_drift_{ticker}_latest.json").write_text(
        json.dumps(report, indent=2))

    if drift_status in ("degraded", "warning"):
        _notify(
            f"[{ticker}] Model drift detected: {drift_status.upper()} — {message}",
            report,
        )

    return report
