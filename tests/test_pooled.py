"""
tests/test_pooled.py

The pooled cross-sectional model (backtesting/pooled_backtest.py) on synthetic
panels: ranking within dates, per-ticker targets, no look-ahead in the walk
forward, and that a genuinely predictive feature is found.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

import pooled_backtest as pb
from cross_section import rank_ic, forward_returns, wide


def _panel(n_tickers=20, n_dates=700, seed=0, leak=False):
    """Random-walk prices; every feature is noise unless `leak`, which makes
    momentum_1m the ticker's actual next-21-session return."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    parts = []
    for k in range(n_tickers):
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n_dates)))
        df = pd.DataFrame({"Date": dates, "symbol": f"T{k:02d}", "Open": c, "Close": c})
        for f in pb.FEATURES:
            df[f] = rng.normal(size=n_dates)
        if leak:
            df["momentum_1m"] = pd.Series(c).shift(-pb.HORIZON) / c - 1
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def test_cs_rank_is_within_date_and_keeps_nan():
    p = pd.DataFrame({"Date": ["d1"] * 3 + ["d2"] * 2, "x": [1, 2, np.nan, 10, 5]})
    r = pb.cs_rank(p, ["x"])["x"]
    assert r.iloc[0] == pytest.approx(0.0) and r.iloc[1] == pytest.approx(0.5) and np.isnan(r.iloc[2])
    assert r.iloc[3] == pytest.approx(0.5) and r.iloc[4] == pytest.approx(0.0)


def test_target_is_each_tickers_own_future_ranked_within_date():
    dates = pd.bdate_range("2026-01-01", periods=3)
    p = pd.DataFrame({"Date": list(dates) * 2, "symbol": ["A"] * 3 + ["B"] * 3,
                      "Close": [10, 11, 12, 10, 20, 40]})
    t = pb.add_target(p, h=1)
    a0 = t[(t.symbol == "A") & (t.Date == dates[0])].iloc[0]
    b0 = t[(t.symbol == "B") & (t.Date == dates[0])].iloc[0]
    assert a0["fwd"] == pytest.approx(np.log(1.1)) and b0["fwd"] == pytest.approx(np.log(2.0))
    assert a0["y"] < b0["y"]
    assert t[t.Date == dates[-1]]["fwd"].isna().all()       # no future past the data


def test_walk_forward_trains_only_on_labels_known_before_each_refit(monkeypatch):
    monkeypatch.setattr(pb, "MIN_TRAIN_DATES", 100)
    rows, X, dates = pb.prepare(_panel(n_dates=300))
    seen = []

    class Spy:
        def fit(self, Xt, yt):
            seen.append(len(yt))
            return self

        def predict(self, Xs):
            return np.zeros(len(Xs))

    start = pb.first_refit(rows)
    dpos = rows["date_pos"].to_numpy()
    scores = pb.walk_forward_scores(rows, X, dates, Spy, start)
    scored_dates = scores.dropna(how="all").index
    for k, d in enumerate(scored_dates):
        s = int(np.searchsorted(dates, np.datetime64(d)))
        assert (s - start) % pb.HORIZON == 0
        allowed = ((dpos < s - pb.HORIZON) & rows["usable"].to_numpy()
                   & np.isfinite(rows["y"].to_numpy())).sum()
        assert seen[k] == allowed


def test_a_predictive_feature_is_found_out_of_sample():
    panel = _panel(leak=True)
    rows, X, dates = pb.prepare(panel)
    start = pb.first_refit(rows)
    scores = pb.walk_forward_scores(rows, X, dates, lambda: Ridge(alpha=1.0), start)
    oos = pd.DatetimeIndex(dates[start:])
    ic = rank_ic(scores.loc[oos], forward_returns(wide(panel, "Close").loc[oos], pb.HORIZON), pb.HORIZON)
    assert ic.mean() > 0.8


def test_noise_features_find_nothing():
    panel = _panel(seed=3)
    rows, X, dates = pb.prepare(panel)
    start = pb.first_refit(rows)
    scores = pb.walk_forward_scores(rows, X, dates, lambda: Ridge(alpha=1.0), start)
    oos = pd.DatetimeIndex(dates[start:])
    ic = rank_ic(scores.loc[oos], forward_returns(wide(panel, "Close").loc[oos], pb.HORIZON), pb.HORIZON)
    assert abs(ic.mean()) < 0.15
