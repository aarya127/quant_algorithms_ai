"""
tests/test_targets.py

Unit tests for the real target definitions (data_pipelines/targets.py).
Validates label correctness, no-lookahead integrity, and numerical sanity.
"""
import numpy as np
import pandas as pd
import pytest

from targets import FLAT_THRESHOLD, REGIME_MIN_HISTORY, add_targets


def build_targets(log_ret: pd.Series) -> pd.DataFrame:
    return add_targets(pd.DataFrame({"log_return": log_ret}))


# Regression targets

class TestRegressionTargets:
    def test_target_1d_is_next_day_return(self):
        returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
        df = build_targets(returns)
        assert df["target_1d"].iloc[0] == pytest.approx(0.02)

    def test_target_1d_last_row_is_nan(self):
        returns = pd.Series([0.01, 0.02, 0.03])
        df = build_targets(returns)
        assert pd.isna(df["target_1d"].iloc[-1])

    def test_target_5d_sum_of_next_five(self):
        """Each return is 0.01 → 5-day sum should be 0.05."""
        returns = pd.Series([0.01] * 20)
        df = build_targets(returns)
        val = df["target_5d"].dropna().iloc[0]
        assert val == pytest.approx(0.05, rel=1e-5)

    def test_target_vol_5d_non_negative(self):
        rng = np.random.default_rng(0)
        returns = pd.Series(rng.normal(0, 0.01, 100))
        df = build_targets(returns)
        vol = df["target_vol_5d"].dropna()
        assert (vol >= 0).all()

    def test_target_vol_5d_zero_for_constant_returns(self):
        """Constant returns → zero realized vol."""
        returns = pd.Series([0.01] * 30)
        df = build_targets(returns)
        vol = df["target_vol_5d"].dropna()
        assert (vol.abs() < 1e-10).all()


# Direction target (3-class)

class TestDirectionTarget:
    def test_strong_up_labelled_1(self):
        returns = pd.Series([0.0, 0.02, 0.0])   # row 0 sees +2% tomorrow
        df = build_targets(returns)
        assert df["target_dir_1d"].iloc[0] == 1

    def test_strong_down_labelled_neg1(self):
        returns = pd.Series([0.0, -0.02, 0.0])
        df = build_targets(returns)
        assert df["target_dir_1d"].iloc[0] == -1

    def test_flat_labelled_zero(self):
        returns = pd.Series([0.0, 0.001, 0.0])   # < threshold → flat
        df = build_targets(returns)
        assert df["target_dir_1d"].iloc[0] == 0

    def test_exactly_at_threshold_is_flat(self):
        """Return == FLAT_THRESHOLD: NOT strictly greater → label 0."""
        returns = pd.Series([0.0, FLAT_THRESHOLD, 0.0])
        df = build_targets(returns)
        assert df["target_dir_1d"].iloc[0] == 0

    def test_labels_only_contain_valid_values(self):
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.015, 200))
        df = build_targets(returns)
        unique = set(df["target_dir_1d"].dropna().unique())
        assert unique.issubset({-1, 0, 1}), f"Unexpected labels: {unique}"

    def test_label_distribution_is_sensible(self):
        """With normal returns, all three classes should appear."""
        rng = np.random.default_rng(7)
        returns = pd.Series(rng.normal(0, 0.015, 500))
        df = build_targets(returns)
        counts = df["target_dir_1d"].dropna().value_counts()
        assert len(counts) == 3, "Expected labels -1, 0, and 1 all to appear"


# No-lookahead integrity

class TestNoLookahead:
    def test_target_1d_uses_future_return(self):
        """target_1d at row i must equal log_return at row i+1."""
        returns = pd.Series([0.01, 0.05, 0.03, 0.07, 0.02])
        df = build_targets(returns)
        for i in range(len(returns) - 1):
            assert df["target_1d"].iloc[i] == pytest.approx(returns.iloc[i + 1])

    def test_feature_differs_from_target_at_same_index(self):
        """The same-row log_return and next-day target must differ."""
        returns = pd.Series([0.01, 0.05, 0.03, 0.07, 0.02])
        df = build_targets(returns)
        # row 0: log_return=0.01, target_1d=0.05 — they should not be equal
        assert df["log_return"].iloc[0] != pytest.approx(df["target_1d"].iloc[0])

    def test_target_tail_is_nan(self):
        """Last rows of multi-step targets must be NaN (no future data available)."""
        returns = pd.Series([0.01] * 20)
        df = build_targets(returns)
        assert pd.isna(df["target_1d"].iloc[-1])
        assert pd.isna(df["target_5d"].iloc[-1])
        assert pd.isna(df["target_vol_5d"].iloc[-1])


# Leak regressions — each of these failed with the previous definitions

class TestNoLeakage:
    def test_direction_last_row_unlabelled(self):
        """The last row's next-day return is unknown: no label, not 'flat'."""
        df = build_targets(pd.Series([0.01, 0.02, 0.03]))
        assert pd.isna(df["target_dir_1d"].iloc[-1])

    def test_large_move_threshold_uses_only_past_vol(self):
        """Threshold at t = 2 × std of the 20 returns ending at t, not t+1."""
        rng = np.random.default_rng(3)
        r = pd.Series(rng.normal(0, 0.01, 80))
        df = build_targets(r)
        for t in range(19, len(r) - 1):
            expected = abs(r.iloc[t + 1]) > 2 * r.iloc[t - 19:t + 1].std()
            assert df["target_large_move"].iloc[t] == float(expected)

    def test_regime_ignores_returns_beyond_its_horizon(self):
        """Regime at t may use t+1..t+5; later returns must not change it."""
        rng = np.random.default_rng(5)
        r = pd.Series(rng.normal(0, 0.01, 200))
        base = build_targets(r.copy())["target_regime"]
        t = 120
        shocked = r.copy()
        shocked.iloc[t + 6:] *= 10          # far more volatile future
        after = build_targets(shocked)["target_regime"]
        pd.testing.assert_series_equal(base.iloc[:t + 1], after.iloc[:t + 1])

    def test_regime_needs_history(self):
        df = build_targets(pd.Series(np.random.default_rng(1).normal(0, 0.01, 100)))
        assert df["target_regime"].iloc[:REGIME_MIN_HISTORY - 1].isna().all()

    def test_regime_is_not_same_day_vol(self):
        """It forecasts the next 5 days' vol, so it must follow target_vol_5d."""
        rng = np.random.default_rng(9)
        r = pd.Series(rng.normal(0, 0.01, 300))
        df = build_targets(r)
        ok = df["target_regime"].notna()
        rho = df.loc[ok, "target_regime"].corr(df.loc[ok, "target_vol_5d"], method="spearman")
        assert rho > 0.5
