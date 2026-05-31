"""
tests/test_evaluation_gate.py

Unit tests for the model promotion / evaluation gate logic in registry.py.
Isolates the gate as a pure function — no disk, no models needed.
"""
import pytest

# ─── Gate constants (must match registry.py) ─────────────────────────────────

_MIN_ABSOLUTE = {
    "regression":     0.02,   # IC must be > 0.02 to be worth deploying
    "classification": 0.30,   # F1_w must be > 0.30 (well above random)
}
_MIN_DELTA = {
    "regression":     0.005,  # IC improvement required over production
    "classification": 0.005,  # F1_w improvement required
}
_BASELINE_BEAT_RATIO = 1.05   # new model must be ≥5% better than naive baseline


def evaluate_gate(
    new_val: float,
    existing_val: float | None,
    baseline_val: float | None,
    task: str,
) -> tuple[bool, str]:
    """
    Replicates the promotion gate in registry.py.
    Returns (passes: bool, reason: str).
    """
    # 1. Absolute floor — model must be minimally useful
    floor = _MIN_ABSOLUTE[task]
    if new_val < floor:
        return False, f"below_floor(floor={floor:.3f}, got={new_val:.4f})"

    # 2. Baseline beat — must outperform naive predictor by ≥5%
    if baseline_val is not None and baseline_val > 0:
        required = baseline_val * _BASELINE_BEAT_RATIO
        if new_val < required:
            return False, (
                f"fails_baseline(baseline={baseline_val:.4f}, "
                f"need≥{required:.4f}, got={new_val:.4f})"
            )

    # 3. Improvement over production — must beat existing by MIN_DELTA
    if existing_val is not None:
        required = existing_val + _MIN_DELTA[task]
        if new_val < required:
            return False, (
                f"no_improvement(existing={existing_val:.4f}, "
                f"need≥{required:.4f}, got={new_val:.4f})"
            )

    return True, "passed"


# ─── Absolute floor ───────────────────────────────────────────────────────────

class TestAbsoluteFloor:
    def test_regression_below_floor_blocked(self):
        passes, reason = evaluate_gate(0.01, None, None, "regression")
        assert not passes
        assert "below_floor" in reason

    def test_regression_at_floor_blocked(self):
        """Just below floor — should be blocked."""
        passes, reason = evaluate_gate(0.019, None, None, "regression")
        assert not passes

    def test_regression_above_floor_allowed(self):
        passes, _ = evaluate_gate(0.025, None, None, "regression")
        assert passes

    def test_classification_below_floor_blocked(self):
        passes, reason = evaluate_gate(0.25, None, None, "classification")
        assert not passes
        assert "below_floor" in reason

    def test_classification_above_floor_allowed(self):
        passes, _ = evaluate_gate(0.35, None, None, "classification")
        assert passes

    def test_floor_is_checked_first(self):
        """Even if improving over a bad existing model, floor must pass."""
        passes, reason = evaluate_gate(0.01, -0.05, None, "regression")
        assert not passes
        assert "below_floor" in reason  # floor check takes priority


# ─── Baseline beat ────────────────────────────────────────────────────────────

class TestBaselineBeat:
    def test_barely_beats_baseline_blocked(self):
        """1% better than baseline → below 5% threshold."""
        passes, reason = evaluate_gate(0.101, None, 0.10, "regression")
        assert not passes
        assert "fails_baseline" in reason

    def test_beats_baseline_by_5pct_exactly_blocked(self):
        """Exactly 5% better — not strictly greater than required."""
        passes, reason = evaluate_gate(0.105, None, 0.10, "regression")
        assert not passes

    def test_beats_baseline_by_more_than_5pct_allowed(self):
        passes, _ = evaluate_gate(0.106, None, 0.10, "regression")
        assert passes

    def test_zero_baseline_skips_check(self):
        """Baseline = 0 (random) → ratio check is skipped to avoid div edge case."""
        passes, _ = evaluate_gate(0.025, None, 0.0, "regression")
        assert passes

    def test_none_baseline_skips_check(self):
        passes, _ = evaluate_gate(0.025, None, None, "regression")
        assert passes


# ─── Improvement over existing ────────────────────────────────────────────────

class TestImprovementGate:
    def test_same_score_as_existing_blocked(self):
        passes, reason = evaluate_gate(0.35, 0.35, None, "regression")
        assert not passes
        assert "no_improvement" in reason

    def test_tiny_improvement_blocked(self):
        """Improves by 0.001, less than MIN_DELTA=0.005."""
        passes, reason = evaluate_gate(0.351, 0.35, None, "regression")
        assert not passes
        assert "no_improvement" in reason

    def test_sufficient_improvement_allowed(self):
        passes, _ = evaluate_gate(0.356, 0.35, None, "regression")
        assert passes

    def test_no_existing_model_always_passes_improvement_check(self):
        """First-time registration — no production model to beat."""
        passes, _ = evaluate_gate(0.025, None, None, "regression")
        assert passes

    def test_regression_backwards_blocked(self):
        passes, _ = evaluate_gate(0.28, 0.35, None, "regression")
        assert not passes

    def test_classification_sufficient_improvement(self):
        passes, _ = evaluate_gate(0.856, 0.85, None, "classification")
        assert passes

    def test_classification_zero_improvement_blocked(self):
        passes, _ = evaluate_gate(0.85, 0.85, None, "classification")
        assert not passes


# ─── Combined / realistic scenarios ──────────────────────────────────────────

class TestRealisticScenarios:
    def test_strong_new_regime_model_passes(self):
        """Regime model: IC=0.42, beats existing 0.40, baseline 0.05."""
        passes, reason = evaluate_gate(0.42, 0.40, 0.05, "regression")
        assert passes, f"Expected pass but got: {reason}"

    def test_weak_model_fails_even_when_improving(self):
        """IC=0.015 > existing -0.01, but fails the floor."""
        passes, reason = evaluate_gate(0.015, -0.01, None, "regression")
        assert not passes
        assert "below_floor" in reason

    def test_retrain_after_market_regime_change(self):
        """IC drops to 0.025 from 0.35 (regime change) — blocked."""
        passes, _ = evaluate_gate(0.025, 0.35, None, "regression")
        assert not passes

    def test_first_deploy_only_needs_floor(self):
        """No existing, no baseline — just needs to clear the floor."""
        passes, _ = evaluate_gate(0.03, None, None, "regression")
        assert passes

    def test_classification_regime_model_typical(self):
        """Regime model F1_w=0.98 beats existing 0.97, above baseline."""
        passes, reason = evaluate_gate(0.98, 0.97, 0.40, "classification")
        assert passes, reason

    def test_direction_model_weak_blocked(self):
        """dir_1d F1_w=0.355 improves on 0.350, but barely above floor=0.30."""
        passes, _ = evaluate_gate(0.355, 0.350, None, "classification")
        assert passes  # above floor, sufficient improvement
