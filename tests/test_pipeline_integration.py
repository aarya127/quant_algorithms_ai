"""
tests/test_pipeline_integration.py

Integration tests: full clean → normalize pipeline on a synthetic mini-dataset.
No external data fetching, no network. Uses realistic feature shapes.
"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path


# Synthetic data fixture

N = 80   # enough rows for rolling-20 windows and 5-day forward targets


def make_synthetic_features(n: int = N, seed: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic feature DataFrame covering all column patterns
    present in the real NVDA feature CSV:
      - Core OHLCV / technical indicators (always populated)
      - Sparse fundamentals (quarterly, ~80% null → below drop threshold)
      - A dead feed (>50% null → should be dropped)
      - A constant column (zero variance → should be dropped)
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")

    log_ret       = rng.normal(0, 0.012, n)
    close         = 100 * np.exp(np.cumsum(log_ret))
    realized_vol  = pd.Series(log_ret).rolling(20).std().bfill().values

    df = pd.DataFrame({
        # Core features (always available)
        "log_return":        log_ret,
        "Close":             close,
        "Volume":            rng.integers(1_000_000, 5_000_000, n).astype(float),
        "realized_vol_20d":  realized_vol,
        "SMA_20":            pd.Series(close).rolling(20).mean().bfill().values,
        "SMA_200":           pd.Series(close).rolling(50).mean().bfill().values,
        "RSI":               rng.uniform(30, 70, n),
        "MACD":              rng.normal(0, 0.5, n),
        "OBV":               np.cumsum(rng.normal(0, 1e6, n)),
        # Macro (daily but ffill over weekends)
        "vix_level":         rng.uniform(15, 35, n),
        "yield_10y":         rng.uniform(3.5, 5.0, n),
        "yield_curve_slope": rng.uniform(-0.5, 1.0, n),
        # Sparse fundamentals (60% populated → survives drop threshold, ffill fills rest)
        "fund_eps_ttm":      np.where(rng.random(n) < 0.60, rng.normal(5, 1, n), np.nan),
        "fund_roe":          np.where(rng.random(n) < 0.60, rng.uniform(0.1, 0.4, n), np.nan),
        "insdr_change":      np.where(rng.random(n) < 0.55, rng.normal(0, 1, n), np.nan),
        # Dead feed: >50% null — must be dropped by clean step
        "dead_feed":         np.where(rng.random(n) < 0.08, rng.random(n), np.nan),
        # Zero-variance column — must be dropped by clean step
        "constant_col":      np.full(n, 1.0),
    }, index=idx)
    df.index.name = "Date"
    return df


# Pipeline helpers (mirror clean.py / normalize.py)

def run_clean(df: pd.DataFrame, null_threshold: float = 0.50) -> pd.DataFrame:
    df = df.copy()
    null_pct = df.isnull().mean()
    df.drop(columns=null_pct[null_pct > null_threshold].index.tolist(), inplace=True)
    df = df.ffill(limit=20).bfill(limit=5)
    for col in df.select_dtypes(include="number").columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    zero_var = [c for c in df.select_dtypes(include="number").columns if df[c].std() == 0]
    df.drop(columns=zero_var, inplace=True)
    return df


FLAT_THRESHOLD = 0.005


def run_normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    log_ret = df["log_return"]
    df["target_1d"]         = log_ret.shift(-1)
    df["target_5d"]         = log_ret.shift(-1).rolling(5).sum().shift(-4)
    df["target_vol_5d"]     = log_ret.shift(-1).rolling(5).std().shift(-4)
    df["target_dir_1d"]     = np.where(
        df["target_1d"] > FLAT_THRESHOLD,  1,
        np.where(df["target_1d"] < -FLAT_THRESHOLD, -1, 0),
    )
    rolling_std              = log_ret.rolling(20).std()
    df["target_large_move"]  = (
        df["target_1d"].abs() > 2 * rolling_std.shift(-1)
    ).astype(float)
    vol_rank                 = df["realized_vol_20d"].rank(pct=True)
    df["target_regime"]      = pd.cut(
        vol_rank, bins=[0, 1/3, 2/3, 1.0], labels=[0, 1, 2]
    ).astype(float)
    return df


# Clean step tests

class TestCleanStep:
    def test_dead_feed_removed(self):
        df = run_clean(make_synthetic_features())
        assert "dead_feed" not in df.columns, "dead_feed (>50% null) must be dropped"

    def test_constant_column_removed(self):
        df = run_clean(make_synthetic_features())
        assert "constant_col" not in df.columns, "zero-variance column must be dropped"

    def test_no_nulls_in_output(self):
        df = run_clean(make_synthetic_features())
        remaining = df.isnull().sum()
        assert remaining.sum() == 0, f"Nulls remain:\n{remaining[remaining > 0]}"

    def test_sparse_fundamentals_kept_and_filled(self):
        """fund_eps_ttm is 80% null — below drop threshold, must be kept + filled."""
        df = run_clean(make_synthetic_features())
        assert "fund_eps_ttm" in df.columns
        assert df["fund_eps_ttm"].isnull().sum() == 0

    def test_core_features_all_present(self):
        df = run_clean(make_synthetic_features())
        required = ["log_return", "Close", "realized_vol_20d", "vix_level", "RSI", "MACD"]
        for col in required:
            assert col in df.columns, f"Core feature '{col}' was incorrectly dropped"

    def test_row_count_unchanged(self):
        raw = make_synthetic_features()
        clean = run_clean(raw)
        assert len(clean) == len(raw)

    def test_feature_values_finite(self):
        df = run_clean(make_synthetic_features())
        numeric = df.select_dtypes(include="number")
        assert np.isfinite(numeric.values).all(), "Non-finite values after clean"


# Normalize step tests

class TestNormalizeStep:
    def test_all_targets_created(self):
        df = run_normalize(run_clean(make_synthetic_features()))
        for target in [
            "target_1d", "target_5d", "target_vol_5d",
            "target_dir_1d", "target_large_move", "target_regime",
        ]:
            assert target in df.columns, f"Missing target column: {target}"

    def test_direction_labels_valid(self):
        df = run_normalize(run_clean(make_synthetic_features()))
        unique = set(df["target_dir_1d"].dropna().unique())
        assert unique.issubset({-1, 0, 1}), f"Invalid direction labels: {unique}"

    def test_vol_target_non_negative(self):
        df = run_normalize(run_clean(make_synthetic_features()))
        vol = df["target_vol_5d"].dropna()
        assert (vol >= 0).all()

    def test_regime_labels_in_expected_set(self):
        df = run_normalize(run_clean(make_synthetic_features()))
        regimes = set(df["target_regime"].dropna().astype(int).unique())
        assert regimes.issubset({0, 1, 2}), f"Unexpected regime labels: {regimes}"

    def test_all_three_regimes_appear(self):
        """With 80 rows the vol percentile should produce all three terciles."""
        df = run_normalize(run_clean(make_synthetic_features()))
        regimes = df["target_regime"].dropna().astype(int).unique()
        assert len(regimes) == 3, f"Expected 3 regimes, got: {sorted(regimes)}"

    def test_row_count_unchanged(self):
        raw = make_synthetic_features()
        df = run_normalize(run_clean(raw))
        assert len(df) == len(raw)

    def test_large_move_is_binary(self):
        df = run_normalize(run_clean(make_synthetic_features()))
        vals = set(df["target_large_move"].dropna().unique())
        assert vals.issubset({0.0, 1.0}), f"large_move not binary: {vals}"


# End-to-end pipeline correctness

class TestEndToEnd:
    def test_pipeline_produces_usable_ml_dataset(self):
        """
        A minimal sanity check: after clean + normalize, we should be able
        to train a trivial sklearn model without errors.
        """
        from sklearn.linear_model import Ridge

        df = run_normalize(run_clean(make_synthetic_features(n=80)))
        feature_cols = [c for c in df.columns if not c.startswith("target_")]
        target_col   = "target_1d"

        ml_df = df[feature_cols + [target_col]].dropna()
        assert len(ml_df) >= 20, "Too few rows for a meaningful ML check"

        X = ml_df[feature_cols].values
        y = ml_df[target_col].values
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(y)

    def test_deterministic_output_across_runs(self):
        """Same seed → same synthetic data → same clean output."""
        df1 = run_clean(make_synthetic_features(seed=99))
        df2 = run_clean(make_synthetic_features(seed=99))
        pd.testing.assert_frame_equal(df1, df2)
