"""
registry.py — Stage 14A: local model registry (save + load).

Directory layout:
  model_registry/
    {ticker}/
      active.json                  ← index of all registered targets
      {target}/
        model.pkl                  ← best model by primary metric
        scaler.pkl
        label_encoder.pkl          ← classification only
        features.json              ← Lasso-selected feature list
        col_med.json               ← per-column training medians (imputation)
        train_stats.json           ← mean/std/p5/p95 per feature (drift)
        metadata.json              ← model type, metrics, created_at
"""

import json
import datetime
import numpy as np
import joblib
from pathlib import Path

# targets registered (key targets for serving)
_SAVE_TARGETS = {
    "target_5d":      "regression",
    "target_1d":      "regression",
    "target_vol_5d":  "regression",
    "target_dir_1d":  "classification",
    "target_regime":  "classification",
}

_REG_PRIMARY = "ic"
_CLF_PRIMARY = "f1_w"

# Evaluation gate thresholds
# A new model must:
#   1. Clear an absolute floor (prevents deploying near-random models)
#   2. Beat the naive baseline by at least BASELINE_BEAT_RATIO
#   3. Improve over the current production model by at least MIN_DELTA
# Pass force=True to save_registry() to bypass checks (e.g. initial seed run).
_MIN_ABSOLUTE = {
    "regression":     0.02,   # IC > 0.02 (barely predictive, but not random)
    "classification": 0.30,   # F1_w > 0.30 (well above majority-class baseline)
}
_MIN_DELTA = {
    "regression":     0.005,  # IC must improve by ≥0.005 over production
    "classification": 0.005,  # F1_w must improve by ≥0.005 over production
}
_BASELINE_BEAT_RATIO = 1.05   # must be ≥5% better than naive baseline


def _promotion_gate(
    new_val: float,
    existing_val,           # float | None
    baseline_val,           # float | None
    task: str,
    target: str,
    primary: str,
) -> tuple[bool, str]:
    """
    Multi-stage promotion gate.
    Returns (passes: bool, log_line: str).
    """
    floor = _MIN_ABSOLUTE[task]
    delta = _MIN_DELTA[task]

    # Gate 1 — absolute floor
    if new_val < floor:
        return False, (
            f"  ✗ GATE[floor]  {target:20s}  "
            f"{primary}={new_val:.4f} < floor={floor:.3f}"
        )

    # Gate 2 — must beat naive baseline by ≥5%
    if baseline_val is not None and baseline_val > 0:
        required = baseline_val * _BASELINE_BEAT_RATIO
        if new_val < required:
            return False, (
                f"  ✗ GATE[baseline]  {target:20s}  "
                f"{primary}={new_val:.4f} baseline={baseline_val:.4f} need≥{required:.4f}"
            )

    # Gate 3 — must beat existing production model by MIN_DELTA
    if existing_val is not None:
        required = existing_val + delta
        if new_val < required:
            return False, (
                f"  ↩ GATE[improvement]  {target:20s}  "
                f"{primary}={new_val:.4f} existing={existing_val:.4f} need≥{required:.4f}"
            )

    return True, ""


def _best_model(model_res, primary_metric):
    """Return (model_name, metric_value) for the best non-baseline model."""
    best_name, best_val = None, -np.inf
    for name, res in model_res.items():
        if name in ("baseline", "_meta") or "error" in res:
            continue
        val = res.get("metrics", {}).get(primary_metric, -np.inf)
        if val > best_val:
            best_val, best_name = val, name
    return best_name, best_val


# save

def save_registry(all_holdout, ticker, registry_dir, force: bool = False):
    """
    Save best model per target to a local folder registry.

    Parameters
    ----------
    all_holdout  : dict  {(target, version): {model_name: result_dict}}
    ticker       : str
    registry_dir : Path-like  (e.g. supervised/model_registry)
    force        : bool  — skip evaluation gates (use for initial seeding only)
    """
    reg_root   = Path(registry_dir) / ticker
    version    = "B"                                  # prefer version B
    created_at = datetime.datetime.utcnow().isoformat() + "Z"
    active     = {}

    print(f"\n── model registry  ({ticker}) ────────────────────────────────")

    for target, task in _SAVE_TARGETS.items():
        primary   = _REG_PRIMARY if task == "regression" else _CLF_PRIMARY
        model_res = all_holdout.get((target, version), {})
        used_ver  = version
        if not model_res:
            model_res = all_holdout.get((target, "A"), {})
            used_ver  = "A"
        if not model_res:
            continue

        meta_block = model_res.get("_meta", {})
        best_name, best_val = _best_model(model_res, primary)
        if best_name is None:
            continue
        best_res = model_res[best_name]
        if "model_obj" not in best_res:
            print(f"  ✗ {target}  — model_obj missing (pipeline not updated?)")
            continue

        # per-target directory
        tgt_dir = reg_root / target
        tgt_dir.mkdir(parents=True, exist_ok=True)

        # Evaluation gate
        existing_meta_path = tgt_dir / "metadata.json"
        existing_val   = None
        existing_meta  = {}
        if existing_meta_path.exists():
            try:
                existing_meta = json.loads(existing_meta_path.read_text())
                existing_val  = float(existing_meta.get("metric_value", -np.inf))
            except Exception:
                pass  # unreadable metadata → treat as first registration

        # Baseline score from model_res (the naive predictor trained alongside)
        baseline_res = model_res.get("baseline", {})
        baseline_val = baseline_res.get("metrics", {}).get(primary) if baseline_res else None

        if not force:
            passes, gate_msg = _promotion_gate(
                new_val=best_val,
                existing_val=existing_val,
                baseline_val=baseline_val,
                task=task,
                target=target,
                primary=primary,
            )
            if not passes:
                print(gate_msg)
                if existing_val is not None:
                    # Keep the existing (better) model in active.json
                    active[target] = {
                        "path":         str(tgt_dir),
                        "model_type":   existing_meta.get("model_type", "unknown"),
                        "metric_value": existing_val,
                        "created_at":   existing_meta.get("created_at", created_at),
                    }
                continue

        joblib.dump(best_res["model_obj"], tgt_dir / "model.pkl")

        if meta_block.get("scaler") is not None:
            joblib.dump(meta_block["scaler"], tgt_dir / "scaler.pkl")

        if meta_block.get("serving_scaler") is not None:
            joblib.dump(meta_block["serving_scaler"], tgt_dir / "serving_scaler.pkl")

        col_med_sel = meta_block.get("col_med_sel")
        if col_med_sel is not None:
            (tgt_dir / "col_med_sel.json").write_text(
                json.dumps(col_med_sel.tolist(), indent=2))

        if "label_encoder" in best_res:
            joblib.dump(best_res["label_encoder"], tgt_dir / "label_encoder.pkl")

        sel_features = meta_block.get("sel_features", [])
        (tgt_dir / "features.json").write_text(json.dumps(sel_features, indent=2))

        col_med = meta_block.get("col_med")
        if col_med is not None:
            (tgt_dir / "col_med.json").write_text(
                json.dumps(col_med.tolist(), indent=2))

        train_stats = meta_block.get("train_stats", {})
        (tgt_dir / "train_stats.json").write_text(json.dumps(train_stats, indent=2))

        metrics_clean = {
            k: float(v)
            for k, v in best_res.get("metrics", {}).items()
            if isinstance(v, (int, float)) and not np.isnan(v)
        }
        metadata = {
            "ticker":           ticker,
            "target":           target,
            "task":             task,
            "model_type":       best_name,
            "feature_version":  used_ver,
            "created_at":       created_at,
            "primary_metric":   primary,
            "metric_value":     round(float(best_val), 6),
            "n_features":       len(sel_features),
            "metrics":          metrics_clean,
        }
        (tgt_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        active[target] = {
            "path":         str(tgt_dir),
            "model_type":   best_name,
            "metric_value": round(float(best_val), 6),
            "created_at":   created_at,
        }
        print(f"  ✓ {target:20s}  [{best_name:10s}  {primary}={best_val:.4f}]")

    (reg_root / "active.json").write_text(
        json.dumps({"ticker": ticker, "created_at": created_at,
                    "targets": active}, indent=2))
    print(f"  ✓ active.json  ({len(active)} targets registered)")


# load

def load_registry(ticker, registry_dir, target="target_5d"):
    """
    Load model + preprocessing artifacts for a given ticker + target.

    Returns dict with keys:
        model, scaler, features, col_med, train_stats, metadata,
        label_encoder (classification only)
    """
    tgt_dir = Path(registry_dir) / ticker / target
    if not tgt_dir.exists():
        raise FileNotFoundError(f"Registry not found: {tgt_dir}")

    result = {
        "model":    joblib.load(tgt_dir / "model.pkl"),
        "features": json.loads((tgt_dir / "features.json").read_text()),
        "metadata": json.loads((tgt_dir / "metadata.json").read_text()),
        "train_stats": (
            json.loads((tgt_dir / "train_stats.json").read_text())
            if (tgt_dir / "train_stats.json").exists() else {}
        ),
        "scaler": (
            joblib.load(tgt_dir / "scaler.pkl")
            if (tgt_dir / "scaler.pkl").exists() else None
        ),
        "serving_scaler": (
            joblib.load(tgt_dir / "serving_scaler.pkl")
            if (tgt_dir / "serving_scaler.pkl").exists() else None
        ),
        "col_med": (
            json.loads((tgt_dir / "col_med.json").read_text())
            if (tgt_dir / "col_med.json").exists() else None
        ),
        "col_med_sel": (
            json.loads((tgt_dir / "col_med_sel.json").read_text())
            if (tgt_dir / "col_med_sel.json").exists() else None
        ),
    }
    le_path = tgt_dir / "label_encoder.pkl"
    if le_path.exists():
        result["label_encoder"] = joblib.load(le_path)
    return result
