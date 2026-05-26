"""
config.py — Walk-forward supervised modeling: constants and paths.
"""
import sys
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
HERE      = Path(__file__).parent
ROOT      = HERE.parent
PIPELINES = ROOT / "data_pipelines"
FD_OUT    = ROOT / "factor_discovery" / "output"
OUT_DIR   = HERE / "output"
OUT_DIR.mkdir(exist_ok=True)

SYMBOL      = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
REGIMES_CSV = PIPELINES / f"{SYMBOL}_features_with_regimes.csv"
FEAT_FILE   = FD_OUT / "recommended_features.txt"

# ── walk-forward config ───────────────────────────────────────────────────────
INIT_TRAIN   = 120
STEP         = 21
HOLDOUT_ROWS = 51

# ── model config ──────────────────────────────────────────────────────────────
LASSO_ALPHA  = 5e-3
SMOTE_K      = 5

# ── target groups ─────────────────────────────────────────────────────────────
REG_TARGETS  = ["target_1d", "target_5d", "target_vol_5d"]
CLF_TARGETS  = ["target_dir_1d", "target_large_move", "target_regime"]
BINARY_CLF   = {"target_large_move"}
MULTI_CLF    = {"target_dir_1d", "target_regime"}
REGIME_COLS  = ["cluster_kmeans", "anomaly_iso", "anomaly_score"]

# ── Stage 7C — hyperparameter tuning ─────────────────────────────────────────
USE_TUNING   = True   # RandomizedSearchCV with TimeSeriesSplit inner CV
TUNE_ITER    = 25     # iterations per model per fold (guide: 25–50)
TUNE_SPLITS  = 3      # inner TimeSeriesSplit folds
