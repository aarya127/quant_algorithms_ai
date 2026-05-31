"""
tests/test_transforms.py

Unit tests for data-cleaning logic (mirrors clean.py).
Self-contained — no file I/O, no network calls.
"""
import numpy as np
import pandas as pd
import pytest


# Helpers: replicate clean.py logic as pure functions

def apply_null_drop(df: pd.DataFrame, threshold: float = 0.50) -> pd.DataFrame:
    df = df.copy()
    null_pct = df.isnull().mean()
    return df.drop(columns=null_pct[null_pct > threshold].index.tolist())


def apply_ffill_bfill(df: pd.DataFrame, ffill_limit: int = 20, bfill_limit: int = 5) -> pd.DataFrame:
    return df.ffill(limit=ffill_limit).bfill(limit=bfill_limit)


def apply_median_fill(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include="number").columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def drop_zero_variance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    zero_var = [c for c in df.select_dtypes(include="number").columns if df[c].std() == 0]
    return df.drop(columns=zero_var)


def full_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = apply_null_drop(df)
    df = apply_ffill_bfill(df)
    df = apply_median_fill(df)
    df = drop_zero_variance(df)
    return df


# Null-drop

class TestNullDrop:
    def test_drops_column_above_threshold(self):
        df = pd.DataFrame({
            "high_null": [np.nan] * 6 + [1.0] * 4,   # 60% null → drop
            "low_null":  [1.0] * 8 + [np.nan] * 2,   # 20% null → keep
        })
        result = apply_null_drop(df)
        assert "high_null" not in result.columns
        assert "low_null" in result.columns

    def test_keeps_exactly_at_threshold(self):
        """Exactly 50% nulls → kept (condition is strict >, not >=)."""
        df = pd.DataFrame({"col": [np.nan, np.nan, 1.0, 1.0]})
        result = apply_null_drop(df)
        assert "col" in result.columns

    def test_drops_all_null_column(self):
        df = pd.DataFrame({"dead": [np.nan] * 10, "live": range(10)})
        result = apply_null_drop(df)
        assert "dead" not in result.columns
        assert "live" in result.columns

    def test_no_false_drops_when_no_nulls(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result = apply_null_drop(df)
        assert set(result.columns) == {"a", "b"}

    def test_custom_threshold_respected(self):
        df = pd.DataFrame({"col": [np.nan, 1.0, 1.0, 1.0]})  # 25% null
        assert "col" not in apply_null_drop(df, threshold=0.20).columns
        assert "col" in apply_null_drop(df, threshold=0.50).columns


# Forward / back fill

class TestFfillBfill:
    def test_ffill_within_limit(self):
        s = pd.Series([1.0, np.nan, np.nan, np.nan, np.nan])
        filled = s.ffill(limit=3)
        assert filled.iloc[1] == 1.0
        assert filled.iloc[2] == 1.0
        assert filled.iloc[3] == 1.0
        assert pd.isna(filled.iloc[4])  # beyond limit

    def test_ffill_does_not_exceed_limit(self):
        s = pd.Series([5.0] + [np.nan] * 5)
        filled = s.ffill(limit=2)
        assert filled.iloc[1] == 5.0
        assert filled.iloc[2] == 5.0
        assert pd.isna(filled.iloc[3])

    def test_bfill_fills_leading_nulls(self):
        s = pd.Series([np.nan, np.nan, 3.0, 4.0])
        filled = s.bfill(limit=5)
        assert filled.iloc[0] == 3.0
        assert filled.iloc[1] == 3.0

    def test_bfill_respects_limit(self):
        s = pd.Series([np.nan, np.nan, np.nan, 7.0])
        filled = s.bfill(limit=1)
        assert pd.isna(filled.iloc[0])  # beyond limit
        assert filled.iloc[2] == 7.0   # within limit


# Median fill

class TestMedianFill:
    def test_fills_null_with_median(self):
        s = pd.Series([1.0, 3.0, np.nan, 5.0])
        med = s.median()
        df = apply_median_fill(pd.DataFrame({"col": s}))
        assert df["col"].iloc[2] == med

    def test_no_nulls_remain(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, np.nan]})
        result = apply_median_fill(df)
        assert result.isnull().sum().sum() == 0

    def test_median_computed_correctly(self):
        s = pd.Series([2.0, 4.0, 6.0, np.nan])
        assert s.median() == pytest.approx(4.0)

    def test_column_with_no_nulls_unchanged(self):
        df = pd.DataFrame({"col": [1.0, 2.0, 3.0]})
        result = apply_median_fill(df)
        pd.testing.assert_series_equal(result["col"], df["col"])


# Zero-variance drop

class TestZeroVarianceDrop:
    def test_drops_constant_column(self):
        df = pd.DataFrame({"const": [5.0] * 10, "varied": range(10)}, dtype=float)
        result = drop_zero_variance(df)
        assert "const" not in result.columns
        assert "varied" in result.columns

    def test_keeps_non_constant_column(self):
        df = pd.DataFrame({"varied": [1.0, 2.0, 3.0, 4.0]})
        result = drop_zero_variance(df)
        assert "varied" in result.columns

    def test_drops_all_if_all_constant(self):
        df = pd.DataFrame({"a": [1.0] * 5, "b": [2.0] * 5})
        result = drop_zero_variance(df)
        assert result.shape[1] == 0

    def test_single_row_not_constant(self):
        """Single-value series has std=NaN, not 0 — should be kept."""
        df = pd.DataFrame({"col": [42.0]})
        result = drop_zero_variance(df)
        assert "col" in result.columns


# Full pipeline end-to-end

class TestFullCleanPipeline:
    def test_no_nulls_in_final_output(self):
        rng = np.random.default_rng(42)
        data = rng.random((30, 6))
        data[data < 0.2] = np.nan   # ~20% nulls throughout
        df = pd.DataFrame(data, columns=[f"f{i}" for i in range(6)])
        result = full_clean(df)
        assert result.isnull().sum().sum() == 0

    def test_row_count_preserved(self):
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [5.0] * 4})
        result = full_clean(df)
        assert len(result) == len(df)

    def test_dead_column_removed_in_full_pipeline(self):
        df = pd.DataFrame({
            "dead": [np.nan] * 10,
            "live": np.arange(10, dtype=float),
        })
        result = full_clean(df)
        assert "dead" not in result.columns
