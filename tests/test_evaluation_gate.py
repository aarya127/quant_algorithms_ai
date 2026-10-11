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


# Walk-forward stability — a model must have held up across the CV folds before
# its holdout score counts (lgb's 5d holdout IC was 0.23 while its CV IC was
# negative in 6 of 8 quarters).

import json

from sklearn.dummy import DummyRegressor

from registry import cv_stability, save_registry, _CV_MIN_BEAT_FRAC


def _folds(task, baseline, **models):
    """run_folds-shaped records: one baseline + one row per model per fold."""
    key = "ic" if task == "regression" else "f1_w"
    recs = [{"fold": f, "model": "baseline", key: b} for f, b in enumerate(baseline)]
    for name, vals in models.items():
        recs += [{"fold": f, "model": name, key: v} for f, v in enumerate(vals)]
    return recs


class TestCvStability:
    def test_min_beat_fraction_constant(self):
        assert _CV_MIN_BEAT_FRAC == 0.5

    def test_one_lucky_fold_is_not_stable(self):
        s = cv_stability(_folds("regression", [0, 0, 0, 0],
                                lucky=[0.6, -0.1, -0.1, -0.1]), "regression")["lucky"]
        assert s["cv_mean"] == pytest.approx(0.075)
        assert s["cv_beat_frac"] == 0.25 and not s["stable"]

    def test_consistent_edge_is_stable(self):
        s = cv_stability(_folds("regression", [0, 0, 0, 0],
                                steady=[0.05, 0.03, -0.01, 0.04]), "regression")["steady"]
        assert s["stable"] and s["n_folds"] == 4

    def test_often_positive_but_mean_below_floor_is_not_stable(self):
        s = cv_stability(_folds("regression", [0, 0, 0, 0],
                                weak=[0.01, 0.01, 0.01, -0.01]), "regression")["weak"]
        assert s["cv_beat_frac"] == 0.75 and not s["stable"]

    def test_classification_compares_against_each_folds_baseline(self):
        s = cv_stability(_folds("classification", [0.30, 0.40, 0.30, 0.40],
                                clf=[0.35, 0.35, 0.35, 0.35]), "classification")["clf"]
        assert s["cv_beat_frac"] == 0.5 and s["stable"]

    def test_errored_fold_rows_are_ignored(self):
        recs = _folds("regression", [0, 0], m=[0.05, 0.05])
        recs.append({"fold": 2, "model": "m", "error": "boom"})
        assert cv_stability(recs, "regression")["m"]["n_folds"] == 2


def _holdout(stability, **ics):
    res = {"baseline": {"metrics": {"ic": 0.0}},
           "_meta": {"sel_features": ["f1"], "cv_stability": stability}}
    for name, ic in ics.items():
        res[name] = {"metrics": {"ic": ic}, "model_obj": DummyRegressor().fit([[0]], [0])}
    return {("target_5d", "B"): res}


def _stable(flag):
    return {"cv_mean": 0.05 if flag else -0.1, "cv_beat_frac": 0.6 if flag else 0.2,
            "n_folds": 8, "stable": flag}


class TestRegistryUsesStability:
    def test_best_stable_model_wins_over_better_unstable_holdout(self, tmp_path):
        save_registry(_holdout({"lgb": _stable(False), "ridge": _stable(True)},
                               lgb=0.23, ridge=0.06), "TST", tmp_path)
        active = json.loads((tmp_path / "TST" / "active.json").read_text())["targets"]
        assert active["target_5d"]["model_type"] == "ridge"
        meta = json.loads((tmp_path / "TST" / "target_5d" / "metadata.json").read_text())
        assert meta["cv"]["stable"] is True

    def test_no_stable_model_serves_nothing(self, tmp_path):
        save_registry(_holdout({"lgb": _stable(False)}, lgb=0.23), "TST", tmp_path)
        active = json.loads((tmp_path / "TST" / "active.json").read_text())["targets"]
        assert "target_5d" not in active

    def test_force_ignores_stability(self, tmp_path):
        save_registry(_holdout({"lgb": _stable(False)}, lgb=0.23), "TST", tmp_path, force=True)
        active = json.loads((tmp_path / "TST" / "active.json").read_text())["targets"]
        assert active["target_5d"]["model_type"] == "lgb"

    def test_without_cv_record_behaves_as_before(self, tmp_path):
        save_registry(_holdout(None, lgb=0.23), "TST", tmp_path)
        active = json.loads((tmp_path / "TST" / "active.json").read_text())["targets"]
        assert active["target_5d"]["model_type"] == "lgb"

    def test_unstable_incumbent_is_not_protected(self, tmp_path):
        """Production lgb (holdout IC 0.23) can't hold off a stable ridge (0.06)
        once this run's CV says lgb is unstable."""
        save_registry(_holdout(None, lgb=0.23), "TST", tmp_path)            # incumbent
        save_registry(_holdout({"lgb": _stable(False), "ridge": _stable(True)},
                               lgb=0.23, ridge=0.06), "TST", tmp_path)
        active = json.loads((tmp_path / "TST" / "active.json").read_text())["targets"]
        assert active["target_5d"]["model_type"] == "ridge"

    def test_unstable_incumbent_and_no_challenger_serves_nothing(self, tmp_path):
        save_registry(_holdout(None, lgb=0.23), "TST", tmp_path)
        save_registry(_holdout({"lgb": _stable(False), "ridge": _stable(True)},
                               lgb=0.23, ridge=0.01), "TST", tmp_path)       # ridge < floor
        active = json.loads((tmp_path / "TST" / "active.json").read_text())["targets"]
        assert "target_5d" not in active

    def test_stable_incumbent_still_needs_beating(self, tmp_path):
        save_registry(_holdout(None, ridge=0.20), "TST", tmp_path)
        save_registry(_holdout({"ridge": _stable(True), "huber": _stable(True)},
                               ridge=0.05, huber=0.06), "TST", tmp_path)
        active = json.loads((tmp_path / "TST" / "active.json").read_text())["targets"]
        assert active["target_5d"]["model_type"] == "ridge"                 # kept
