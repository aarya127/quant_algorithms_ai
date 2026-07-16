"""
config.py — Walk-forward supervised modeling: constants and paths.
"""
import sys
from pathlib import Path

# paths
HERE      = Path(__file__).parent
ROOT      = HERE.parent
PIPELINES = ROOT / "data_pipelines"
FD_OUT    = ROOT / "factor_discovery" / "output"
OUT_DIR   = HERE / "output"
OUT_DIR.mkdir(exist_ok=True)

SYMBOL      = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
REGIMES_CSV = PIPELINES / f"{SYMBOL}_features_with_regimes.csv"
FEAT_FILE   = FD_OUT / "recommended_features.txt"

# walk-forward config
INIT_TRAIN   = 120
STEP         = 21
HOLDOUT_ROWS = 51

# Purge/embargo gap (rows) between train and validation/holdout, per target.
# Targets are forward-looking, so the last `horizon` training rows carry labels
# computed from data inside the eval window. Dropping them prevents look-ahead
# leakage that would otherwise inflate CV/holdout metrics (and the promotion gate).
#   target_1d / dir_1d / large_move  → label uses t+1              → 1
#   target_5d / vol_5d               → label uses t+1..t+5         → 5
#   target_regime                    → conservative (vol-based)    → 5
TARGET_HORIZON = {
    "target_1d":         1,
    "target_dir_1d":     1,
    "target_large_move": 1,
    "target_5d":         5,
    "target_vol_5d":     5,
    "target_regime":     5,
}


def target_embargo(target):
    """Rows to purge between train and eval for `target` (conservative default 5)."""
    return TARGET_HORIZON.get(target, 5)

# model config
LASSO_ALPHA  = 5e-3
SMOTE_K      = 5

# target groups
REG_TARGETS  = ["target_1d", "target_5d", "target_vol_5d"]
CLF_TARGETS  = ["target_dir_1d", "target_large_move", "target_regime"]
BINARY_CLF   = {"target_large_move"}
MULTI_CLF    = {"target_dir_1d", "target_regime"}
REGIME_COLS  = ["cluster_kmeans", "anomaly_iso", "anomaly_score"]

# Stage 7C — hyperparameter tuning
# Set env var USE_TUNING=0 for a fast no-tuning run (default params, ~30s).
# Set USE_TUNING=1 (or leave unset) for full RandomizedSearchCV (~10 min).
import os as _os
USE_TUNING   = _os.environ.get("USE_TUNING", "1") != "0"
TUNE_ITER    = 25     # iterations per model per fold (guide: 25–50)
TUNE_SPLITS  = 3      # inner TimeSeriesSplit folds
