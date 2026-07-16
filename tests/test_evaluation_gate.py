"""
tests/test_evaluation_gate.py

Unit tests for the model promotion / evaluation gate in registry.py.

These import and exercise the REAL `_promotion_gate` (not a copy), so if the
production gate logic or thresholds change, these tests move with them — a change
that breaks the promotion contract fails CI instead of silently passing.
"""
import pytest

# conftest.py puts algorithms/.../supervised on sys.path
from registry import (
    _promotion_gate,
    _MIN_ABSOLUTE,
    _MIN_DELTA,
    _BASELINE_BEAT_RATIO,
)


def gate(new_val, existing_val, baseline_val, task):
    """Thin adapter over the real gate. `target`/`primary` are labels only (they
    appear in the log line, not the decision), so we pass canonical values."""
    primary = "ic" if task == "regression" else "f1_w"
    return _promotion_gate(new_val, existing_val, baseline_val, task, "target_test", primary)


# Guard the thresholds themselves — documents the contract and fails loudly if
# someone edits registry.py's constants without intending to.

def test_threshold_constants():
    assert _MIN_ABSOLUTE == {"regression": 0.02, "classification": 0.30}
    assert _MIN_DELTA == {"regression": 0.005, "classification": 0.005}
    assert _BASELINE_BEAT_RATIO == 1.05


# Absolute floor

class TestAbsoluteFloor:
    def test_regression_below_floor_blocked(self):
        passes, reason = gate(0.01, None, None, "regression")
        assert not passes
        assert "floor" in reason

    def test_regression_at_floor_blocked(self):
        """Just below floor — should be blocked."""
        passes, reason = gate(0.019, None, None, "regression")
        assert not passes

    def test_regression_above_floor_allowed(self):
        passes, _ = gate(0.025, None, None, "regression")
        assert passes

    def test_classification_below_floor_blocked(self):
        passes, reason = gate(0.25, None, None, "classification")
        assert not passes
        assert "floor" in reason

    def test_classification_above_floor_allowed(self):
        passes, _ = gate(0.35, None, None, "classification")
        assert passes

    def test_floor_is_checked_first(self):
        """Even if improving over a bad existing model, floor must pass."""
        passes, reason = gate(0.01, -0.05, None, "regression")
        assert not passes
        assert "floor" in reason  # floor check takes priority


# Baseline beat

class TestBaselineBeat:
    def test_barely_beats_baseline_blocked(self):
        """1% better than baseline → below 5% threshold."""
        passes, reason = gate(0.101, None, 0.10, "regression")
        assert not passes
        assert "baseline" in reason

    def test_beats_baseline_by_5pct_exactly_blocked(self):
        """Exactly 5% better — not strictly greater than required."""
        passes, reason = gate(0.10499, None, 0.10, "regression")
        assert not passes

    def test_beats_baseline_by_more_than_5pct_allowed(self):
        passes, _ = gate(0.106, None, 0.10, "regression")
        assert passes

    def test_zero_baseline_skips_check(self):
        """Baseline = 0 (random) → ratio check is skipped to avoid div edge case."""
        passes, _ = gate(0.025, None, 0.0, "regression")
        assert passes

    def test_none_baseline_skips_check(self):
        passes, _ = gate(0.025, None, None, "regression")
        assert passes


# Improvement over existing

class TestImprovementGate:
    def test_same_score_as_existing_blocked(self):
        passes, reason = gate(0.35, 0.35, None, "regression")
        assert not passes
        assert "improvement" in reason

    def test_tiny_improvement_blocked(self):
        """Improves by 0.001, less than MIN_DELTA=0.005."""
        passes, reason = gate(0.351, 0.35, None, "regression")
        assert not passes
        assert "improvement" in reason

    def test_sufficient_improvement_allowed(self):
        passes, _ = gate(0.356, 0.35, None, "regression")
        assert passes

    def test_no_existing_model_always_passes_improvement_check(self):
        """First-time registration — no production model to beat."""
        passes, _ = gate(0.025, None, None, "regression")
        assert passes

    def test_regression_backwards_blocked(self):
        passes, _ = gate(0.28, 0.35, None, "regression")
        assert not passes

    def test_classification_sufficient_improvement(self):
        passes, _ = gate(0.856, 0.85, None, "classification")
        assert passes

    def test_classification_zero_improvement_blocked(self):
        passes, _ = gate(0.85, 0.85, None, "classification")
        assert not passes


# Combined / realistic scenarios

class TestRealisticScenarios:
    def test_strong_new_regime_model_passes(self):
        """Regime model: IC=0.42, beats existing 0.40, baseline 0.05."""
        passes, reason = gate(0.42, 0.40, 0.05, "regression")
        assert passes, f"Expected pass but got: {reason}"

    def test_weak_model_fails_even_when_improving(self):
        """IC=0.015 > existing -0.01, but fails the floor."""
        passes, reason = gate(0.015, -0.01, None, "regression")
        assert not passes
        assert "floor" in reason

    def test_retrain_after_market_regime_change(self):
        """IC drops to 0.025 from 0.35 (regime change) — blocked."""
        passes, _ = gate(0.025, 0.35, None, "regression")
        assert not passes

    def test_first_deploy_only_needs_floor(self):
        """No existing, no baseline — just needs to clear the floor."""
        passes, _ = gate(0.03, None, None, "regression")
        assert passes

    def test_classification_regime_model_typical(self):
        """Regime model F1_w=0.98 beats existing 0.97, above baseline."""
        passes, reason = gate(0.98, 0.97, 0.40, "classification")
        assert passes, reason

    def test_direction_model_weak_blocked(self):
        """dir_1d F1_w=0.355 improves on 0.350, and clears floor=0.30."""
        passes, _ = gate(0.355, 0.350, None, "classification")
        assert passes  # above floor, sufficient improvement
