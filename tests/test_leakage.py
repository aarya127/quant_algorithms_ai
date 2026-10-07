"""
tests/test_leakage.py

Look-ahead and evaluation-leak regressions, against the real pipeline code:
news session dating, the regression baseline signal, and the promotion gate's
re-scoring of the production model.
"""
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from extractor import _session_date          # data_pipelines/ (conftest)
from baselines import naive_signal           # supervised/ (conftest)
from registry import FEATURE_SPACE, _rescore_existing


# News → trading session

class TestSessionDate:
    def test_before_close_is_same_session(self):
        # 15:59 ET (EDT, UTC-4) on a Tuesday
        assert _session_date("2026-10-06T19:59:00Z") == pd.Timestamp("2026-10-06")

    def test_after_close_moves_to_next_day(self):
        # 16:05 ET — after-hours news (earnings) can't inform that day's row
        assert _session_date("2026-10-06T20:05:00Z") == pd.Timestamp("2026-10-07")

    def test_uses_new_york_date_not_utc_date(self):
        # 22:00 ET Monday is already Tuesday in UTC; it belongs to Tuesday's
        # session because it's after Monday's close — not because of the UTC date
        assert _session_date("2026-10-06T02:00:00Z") == pd.Timestamp("2026-10-06")

    def test_naive_timestamps_are_utc(self):
        assert _session_date("2026-10-06 19:59:00") == pd.Timestamp("2026-10-06")

    def test_winter_offset(self):
        # 16:30 EST (UTC-5) in January → next day
        assert _session_date("2026-01-13T21:30:00Z") == pd.Timestamp("2026-01-14")


# Regression baseline

class TestNaiveSignal:
    def test_vol_target_uses_trailing_vol(self):
        df = pd.DataFrame({"realized_vol_20d": [0.1, 0.2], "log_return": [0.0, 0.0]})
        assert list(naive_signal(df, "target_vol_5d")) == [0.1, 0.2]

    def test_return_targets_use_momentum(self):
        df = pd.DataFrame({"log_return": [0.01, -0.02]})
        assert list(naive_signal(df, "target_1d")) == [0.01, -0.02]

    def test_signal_is_not_constant(self):
        """A constant baseline has no rank correlation, so its IC check never fired."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"log_return": rng.normal(0, 0.01, 50)})
        assert np.std(naive_signal(df, "target_5d")) > 0


# Promotion gate re-scoring

def _register(tmp_path, model, features, metric_value, feature_space=FEATURE_SPACE):
    tgt_dir = tmp_path / "TEST" / "target_1d"
    tgt_dir.mkdir(parents=True)
    joblib.dump(model, tgt_dir / "model.pkl")
    (tgt_dir / "features.json").write_text(json.dumps(features))
    (tgt_dir / "metadata.json").write_text(json.dumps(
        {"metric_value": metric_value, "feature_space": feature_space}))


class TestRescoreExisting:
    def test_scores_on_current_holdout_not_stored_metric(self, tmp_path):
        rng = np.random.default_rng(1)
        x = rng.normal(size=60)
        train = pd.DataFrame({"f": x, "target_1d": x})
        model = LinearRegression().fit(train[["f"]], train["target_1d"])
        _register(tmp_path, model, ["f"], metric_value=0.99)   # stale, inflated

        # On the new holdout the relationship is gone → IC near zero
        holdout = pd.DataFrame({"f": rng.normal(size=51), "target_1d": rng.normal(size=51)})
        ic = _rescore_existing("TEST", tmp_path, "target_1d", "regression", "ic", holdout)
        assert ic is not None and abs(ic) < 0.5

    def test_unscorable_model_returns_none(self, tmp_path):
        _register(tmp_path, "not a model", ["f"], metric_value=0.99)
        holdout = pd.DataFrame({"f": [1.0, 2.0], "target_1d": [0.1, 0.2]})
        assert _rescore_existing("TEST", tmp_path, "target_1d",
                                 "regression", "ic", holdout) is None

    def test_missing_feature_imputed_not_dropped(self, tmp_path):
        rng = np.random.default_rng(2)
        X = pd.DataFrame({"a": rng.normal(size=60), "b": rng.normal(size=60)})
        model = LinearRegression().fit(X, X["a"])
        _register(tmp_path, model, ["a", "b"], metric_value=0.5)
        holdout = pd.DataFrame({"a": rng.normal(size=51)})
        holdout["target_1d"] = holdout["a"]
        ic = _rescore_existing("TEST", tmp_path, "target_1d", "regression", "ic", holdout)
        assert ic == pytest.approx(1.0)

    def test_model_from_older_feature_space_is_unscorable(self, tmp_path):
        """IC is rank-based, so a model fed differently-scaled inputs can still
        score well while predicting nonsense — it must not be compared or kept."""
        rng = np.random.default_rng(4)
        x = rng.normal(size=60)
        model = LinearRegression().fit(x.reshape(-1, 1), x)
        _register(tmp_path, model, ["f"], metric_value=0.9, feature_space=None)
        holdout = pd.DataFrame({"f": x[:51], "target_1d": x[:51]})
        assert _rescore_existing("TEST", tmp_path, "target_1d",
                                 "regression", "ic", holdout) is None
