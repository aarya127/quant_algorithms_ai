"""
tests/test_analysis_tabs.py

Scenarios (backend/scenario_engine.py) and Metrics & Grades (backend/grading.py)
with synthetic prices / metrics — no network.
"""
import math
import sys
import types

import numpy as np
import pytest

import scenario_engine as se
from grading import MIN_PEERS, grade_fundamentals, peer_percentile


@pytest.fixture()
def prices(monkeypatch):
    """5 years of synthetic daily bars with an upward drift, no trained model."""
    import pandas as pd
    rng = np.random.default_rng(0)
    closes = 100 * np.exp(np.cumsum(rng.normal(0.0008, 0.015, 1260)))
    bars = pd.DataFrame({"Date": pd.bdate_range("2021-10-01", periods=1260),
                         "Open": closes * (1 + rng.normal(0, 0.002, 1260)), "Close": closes})
    monkeypatch.setattr(se, "prices", lambda s: (bars, "USD"))
    monkeypatch.setattr(se, "has_model", lambda s: False)
    se._cache.clear()
    return bars


@pytest.mark.parametrize("tf", list(se.HORIZON_DAYS))
def test_targets_are_ordered_percentiles(prices, tf):
    r = se.compute_scenarios("TEST", tf)
    s = r["scenarios"]
    assert s["bear_case"]["price_target"] < s["base_case"]["price_target"] < s["bull_case"]["price_target"]
    assert r["range_90"]["low"] < s["bear_case"]["price_target"]
    assert r["range_90"]["high"] > s["bull_case"]["price_target"]
    # the probabilities are true by construction: 20 / 60 / 20
    assert [s[k]["probability"] for k in ("bull_case", "base_case", "bear_case")] == [20.0, 60.0, 20.0]


def test_long_horizons_never_certain(prices):
    """Overlapping windows used to give 0% bear / 100% up at 1Y."""
    r = se.compute_scenarios("TEST", "1Y")
    assert 0.05 < r["p_up"] < 0.95
    assert r["calibration"] is None              # 5 independent years: too few to check


def test_bands_are_calibrated_on_well_behaved_data(prices):
    """On i.i.d. returns the 20–80 band should hold ~60% of past moves."""
    cal = se.compute_scenarios("TEST", "1M")["calibration"]
    assert 0.5 < cal["inside_band"] < 0.7 and cal["ideal"] == pytest.approx(0.6)


def test_vol_reverts_toward_the_long_run():
    path = se.var_path(var_now=4e-4, var_lr=1e-4, h=252)
    assert path[0] < 4e-4 and abs(path[-1] - 1e-4) < 1e-5 and np.all(np.diff(path) < 0)


def test_ewma_uses_only_past_returns():
    r = np.r_[np.full(30, 0.01), 0.5, np.full(5, 0.01)]
    var = se.ewma_var(r)
    assert var[30] < 0.01 and var[31] > var[30]  # the shock moves only the next day's forecast


def test_simulation_is_deterministic(prices):
    a = se.compute_scenarios("TEST", "3M")
    se._cache.clear()
    assert se.compute_scenarios("TEST", "3M") == a


def test_model_vol_is_already_daily(monkeypatch):
    """predicted_vol_5d is a std of daily returns: it must not be divided by √5."""
    fake = types.SimpleNamespace(predict_latest=lambda s: {
        "date": "2026-10-07", "signal": "long",
        "predictions": {"predicted_5d_return": 0.03, "predicted_vol_5d": 0.02}})
    monkeypatch.setitem(sys.modules, "predictor", fake)
    (drift, vol, meta), problem = se._model_overlay("NVDA", "2026-10-08", 0.02)
    assert problem is None
    assert vol == pytest.approx(0.02)
    assert drift == pytest.approx(0.03 / 5 * se.DRIFT_SHRINK)
    assert meta["prediction_date"] == "2026-10-07"


def test_canadian_symbol_uses_tsx_listing(monkeypatch):
    import pandas as pd
    seen = {}

    class Ticker:
        def __init__(self, t):
            seen["t"] = t
            self.history_metadata = {"currency": "CAD"}

        def history(self, **kw):
            idx = pd.bdate_range("2024-01-01", periods=300, name="Date")
            return pd.DataFrame({"Open": np.linspace(100, 110, 300), "Close": np.linspace(100, 110, 300)}, index=idx)

    monkeypatch.setattr(se, "yf", types.SimpleNamespace(Ticker=Ticker))
    se._prices_cache.clear()
    bars, currency = se.prices("TD")
    assert seen["t"] == "TD.TO" and currency == "CAD" and list(bars.columns) == ["Date", "Open", "Close"]


# Recommendations

import recommendation_engine as rec


def test_price_signals_match_their_definitions():
    closes = np.r_[np.linspace(100, 50, 300), np.linspace(50, 120, 300)]
    sig = rec.price_signals(closes)
    assert np.isnan(sig["trend"][198]) and np.isnan(sig["momentum"][rec.WARMUP - 1])
    t = 599
    assert sig["trend"][t] == (1.0 if closes[t] > closes[t - 199:t + 1].mean() else -1.0)
    assert sig["momentum"][t] == np.sign(closes[t - 5] / closes[t - 257] - 1)


def test_recommend_votes_and_evidence(prices, monkeypatch):
    monkeypatch.setattr(rec, "_quality_vote", lambda s: (-1.0, "overall grade D"))
    r = rec.recommend("TEST")
    assert set(r["horizons"]) == set(rec.WEIGHTS)
    one_y = r["horizons"]["1Y"]
    assert {v["signal"] for v in one_y["votes"]} == {"price", "quality"}
    assert sum(v["weight"] for v in one_y["votes"]) == pytest.approx(1.0, abs=0.01)
    # no model served → 1W uses only the price vote, reweighted
    assert {v["signal"] for v in r["horizons"]["1W"]["votes"]} == {"price"}
    expected = sum(v["vote"] * v["weight"] for v in one_y["votes"])
    assert one_y["score"] == pytest.approx(expected, abs=0.01)
    assert one_y["action"] == ("Buy" if expected > 0.3 else "Sell" if expected < -0.3 else "Hold")
    rules = [row["rule"] for row in r["evidence"]["rows"]]
    assert rules[0] == "Buy and hold" and "Trend (200-day average)" in rules


def test_long_or_cash_never_shorts():
    out = rec._long_or_cash(np.array([1.0, -1.0, 0.0, np.nan]))
    assert out[:3].tolist() == [1.0, 0.0, 0.0] and np.isnan(out[3])


# Metrics & Grades

def test_no_data_grades_nothing():
    """ETFs / unknown tickers used to get a C from made-up defaults."""
    r = grade_fundamentals({})
    assert r["graded_categories"] == 0 and r["overall_grade"] is None


def test_negative_pe_is_not_meaningful():
    r = grade_fundamentals({"peBasicExclExtraTTM": -12.5, "roeTTM": -30})
    v = r["metrics"]["valuation"]
    assert v["grade"] is None and "negative earnings" in v["description"]
    assert v["inputs"]["P/E (TTM)"] == {"value": -12.5, "percentile": None, "grade": None,
                                        "note": "not meaningful: negative earnings"}


def test_market_scale_outlier_cant_dominate():
    """Without peers an ROE of 193% is one A, not a number that swamps the average."""
    r = grade_fundamentals({"roeTTM": 193, "netProfitMarginTTM": -20})
    assert r["metrics"]["profitability"]["grade"] == "C"     # A (90) and F (10) → 50


def test_real_zero_is_kept():
    r = grade_fundamentals({"epsGrowthTTMYoy": 0, "revenueGrowthTTMYoy": 0})
    assert r["metrics"]["growth"]["grade"] == "F"            # not a default 10%


def test_overall_uses_only_graded_categories():
    r = grade_fundamentals({"peBasicExclExtraTTM": 12, "totalDebt/totalEquityQuarterly": 0.1})
    assert r["graded_categories"] == 2
    assert r["overall_grade"] == "A" and r["average_score"] == 90


@pytest.mark.parametrize("pe, grade", [(10, "A"), (15, "B"), (19.9, "B"), (24, "C"), (30, "D"), (50, "F")])
def test_pe_bands(pe, grade):
    assert grade_fundamentals({"peBasicExclExtraTTM": pe})["metrics"]["valuation"]["grade"] == grade


def _peers(key, *values):
    return [{key: v} for v in values]


def test_percentile_direction():
    assert peer_percentile(30, [10, 20, 40, 50], higher_is_better=True) == 50
    assert peer_percentile(30, [10, 20, 40, 50], higher_is_better=False) == 50
    assert peer_percentile(60, [10, 20, 40, 50], higher_is_better=True) == 100
    assert peer_percentile(60, [10, 20, 40, 50], higher_is_better=False) == 0


def test_ties_count_half():
    assert peer_percentile(20, [20, 20, 20, 20], higher_is_better=True) == 50
    assert peer_percentile(20, [10, 20, 30, 40], higher_is_better=True) == 37.5


def test_peer_pe_beats_market_scale():
    """NVDA's P/E of 30 was a market-wide D; among pricier chip peers it ranks well."""
    r = grade_fundamentals({"peBasicExclExtraTTM": 30},
                           _peers("peBasicExclExtraTTM", 90, 110, 60, 35, 18))
    rec = r["metrics"]["valuation"]["inputs"]["P/E (TTM)"]
    assert rec["percentile"] == 80 and rec["grade"] == "A" and rec["note"] is None


def test_outlier_peer_cant_dominate():
    """A peer's 1000% growth moves the company's rank one place, like a 31% peer would."""
    base = [5, 10, 15, 25]
    calm = grade_fundamentals({"revenueGrowthTTMYoy": 30}, _peers("revenueGrowthTTMYoy", *base, 31))
    wild = grade_fundamentals({"revenueGrowthTTMYoy": 30}, _peers("revenueGrowthTTMYoy", *base, 1000))
    assert calm["metrics"]["growth"] == wild["metrics"]["growth"]
    assert wild["metrics"]["growth"]["score"] == 80


def test_too_few_peers_falls_back_to_market_scale():
    peers = _peers("peBasicExclExtraTTM", *range(10, 10 + MIN_PEERS - 1))
    v = grade_fundamentals({"peBasicExclExtraTTM": 12, "psTTM": 3}, peers)["metrics"]["valuation"]
    assert v["inputs"]["P/E (TTM)"] == {"value": 12, "percentile": None, "grade": "A",
                                        "note": "market-wide scale"}
    # P/S never had a market-wide scale: shown, not graded
    assert v["inputs"]["P/S (TTM)"]["grade"] is None
    assert v["grade"] == "A" and "market-wide scale" in v["description"]


def test_loss_making_peers_are_not_cheap():
    """Negative peer P/Es leave the pool; the rest are too few, so the market scale applies."""
    peers = _peers("peBasicExclExtraTTM", -5, -8, 40, 50, 60)
    rec = grade_fundamentals({"peBasicExclExtraTTM": 30}, peers)["metrics"]["valuation"]["inputs"]["P/E (TTM)"]
    assert rec["percentile"] is None and rec["note"] == "market-wide scale"


def test_bank_debt_is_not_graded():
    metric = {"totalDebt/totalEquityQuarterly": 2.26, "currentRatioQuarterly": 0.9, "roeTTM": 13}
    peers = _peers("totalDebt/totalEquityQuarterly", 1.0, 1.2, 1.5, 3.0)
    bank = grade_fundamentals(metric, peers, industry="Banking")
    health = bank["metrics"]["financial_health"]
    assert health["grade"] is None and "banks" in health["description"]
    assert bank["graded_categories"] == 1                    # profitability only
    other = grade_fundamentals(metric, peers, industry="Technology")
    assert other["metrics"]["financial_health"]["inputs"]["Debt/Equity"]["percentile"] == 25


# Robustness (bad price rows, stale/implausible ML overlay, past-only calibration)

def test_clean_bars_drops_missing_and_non_positive_prices():
    import pandas as pd
    df = pd.DataFrame({"Date": pd.bdate_range("2026-01-05", periods=5),
                       "Open": [10, np.nan, 0, 11, 12], "Close": [10, 10.5, 11, np.nan, 12]})
    assert se.clean_bars(df)["Open"].tolist() == [10, 12]


def test_a_bad_yfinance_row_cant_poison_the_scenarios(monkeypatch):
    import pandas as pd
    rng = np.random.default_rng(1)
    closes = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, 800)))
    hist = pd.DataFrame({"Open": closes, "Close": closes},
                        index=pd.bdate_range("2023-01-02", periods=800, name="Date"))
    hist.iloc[400, 1] = np.nan
    hist.iloc[401, 0] = 0.0

    class Ticker:
        history_metadata = {"currency": "USD"}

        def __init__(self, t):
            pass

        def history(self, **kw):
            return hist

    monkeypatch.setattr(se, "yf", types.SimpleNamespace(Ticker=Ticker))
    monkeypatch.setattr(se, "has_model", lambda s: False)
    se._cache.clear(); se._prices_cache.clear()
    r = se.compute_scenarios("BADROW", "1M")
    assert all(math.isfinite(v["price_target"]) for v in r["scenarios"].values())
    assert math.isfinite(r["annualized_vol"])


@pytest.mark.parametrize("args, phrase", [
    ((0.02, 0.02, "2026-10-01", "2026-10-09", 0.02), "sessions before"),
    ((0.02, 0.0, "2026-10-08", "2026-10-09", 0.02), "implausible"),
    ((0.02, -0.01, "2026-10-08", "2026-10-09", 0.02), "implausible"),
    ((0.40, 0.02, "2026-10-08", "2026-10-09", 0.02), "four standard deviations"),
    ((None, 0.02, "2026-10-08", "2026-10-09", 0.02), "no 5-day"),
    ((0.02, 0.02, None, "2026-10-09", 0.02), "no usable date"),
])
def test_overlay_problems_are_named(args, phrase):
    assert phrase in se.overlay_problem(*args)


def test_overlay_within_three_sessions_is_used():
    assert se.overlay_problem(0.02, 0.02, "2026-10-06", "2026-10-09", 0.015) is None


def test_stale_model_forecast_is_skipped_and_said_so(prices, monkeypatch):
    fake = types.SimpleNamespace(predict_latest=lambda s: {
        "date": "2020-01-02", "predictions": {"predicted_5d_return": 0.03, "predicted_vol_5d": 0.015}})
    monkeypatch.setitem(sys.modules, "predictor", fake)
    monkeypatch.setattr(se, "has_model", lambda s: True)
    r = se.compute_scenarios("TEST", "1W")
    assert r["model_backed"] is False and "not used" in r["model_note"]


def test_calibration_uses_non_overlapping_windows(prices):
    cal = se.compute_scenarios("TEST", "3M")["calibration"]
    assert cal["independent_windows"] == len(range(252, 1259 - 63 + 1, 63))
    assert "data before" in cal["method"]


def test_calibration_ignores_the_future():
    """Each window's verdict depends only on returns before it: appending wild
    history afterwards leaves every earlier window's verdict unchanged."""
    rng = np.random.default_rng(3)
    r = rng.normal(0, 0.01, 252 + 21 * 12)
    base = se.calibration_flags(r, se.ewma_var(r), 21, seed=7)
    longer = np.r_[r, rng.normal(0, 0.2, 21 * 12)]
    ext = se.calibration_flags(longer, se.ewma_var(longer), 21, seed=7)
    assert len(ext) > len(base) and (ext[:len(base)] == base).all()


def test_unknown_timeframe_falls_back_to_one_month(prices):
    assert se.compute_scenarios("TEST", "5Y")["timeframe"] == "1M"


# Recommendations: one price vote, quality that can matter

import recommendation_engine as rec_eng   # noqa: E402


@pytest.mark.parametrize("votes, tf, action", [
    ({"price": 1, "quality": 1}, "3M", "Buy"),
    ({"price": 1, "quality": -1}, "3M", "Hold"),      # quality can stop a Buy
    ({"price": 0, "quality": 1}, "6M", "Buy"),        # mixed price signals, quality decides
    ({"price": 0, "quality": -1}, "1Y", "Sell"),
    ({"price": -1, "quality": 1}, "1Y", "Hold"),
    ({"price": 1}, "1Y", "Buy"),                      # no grade (ETF): price alone
    ({"price": 1, "model": -1}, "1W", "Hold"),
])
def test_horizon_actions(votes, tf, action):
    assert rec_eng.horizon_action(votes, rec_eng.WEIGHTS[tf])[0] == action


def test_weights_sum_to_one_per_horizon():
    for w in rec_eng.WEIGHTS.values():
        assert sum(w.values()) == pytest.approx(1.0)


# Metrics & Grades: negative equity, share classes

def test_roe_over_negative_equity_is_not_graded():
    peers = [{"roeTTM": v} for v in (5, 10, 15, 20)]
    r = grade_fundamentals({"roeTTM": 80, "bookValuePerShareQuarterly": -4.0}, peers)
    rec = r["metrics"]["profitability"]["inputs"]["ROE % (TTM)"]
    assert rec["grade"] is None and "negative equity" in rec["note"]


def test_peers_with_negative_equity_dont_set_the_roe_bar():
    peers = [{"roeTTM": v} for v in (5, 10, 15, 20)] + [{"roeTTM": 400, "pbQuarterly": -2}]
    rec = grade_fundamentals({"roeTTM": 25}, peers)["metrics"]["profitability"]["inputs"]["ROE % (TTM)"]
    assert rec["percentile"] == 100


def test_share_classes_are_the_same_company():
    from services import same_company
    googl = {"revenueGrowthTTMYoy": 12.1, "grossMarginTTM": 58.2, "netProfitMarginTTM": 28.6, "roeTTM": 31.0}
    assert same_company(googl, dict(googl))
    assert not same_company(googl, {**googl, "roeTTM": 30.9})
    assert not same_company(googl, {"roeTTM": 31.0})          # too little to tell


# Scenarios route: the AI rationale is written for 1M only

@pytest.fixture()
def client():
    import app as app_module
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


@pytest.mark.parametrize("tf, uses_brief", [("1M", True), ("1Y", False), ("bogus", True)])
def test_brief_rationale_only_on_one_month(prices, client, monkeypatch, tf, uses_brief):
    import llm_analyst
    monkeypatch.setitem(llm_analyst._brief_cache, "TEST",
                        {"scenario_rationale": {"bull": "BRIEF-BULL", "base": "BRIEF-BASE", "bear": "BRIEF-BEAR"}})
    d = client.get(f"/api/scenarios/TEST?timeframe={tf}").get_json()
    assert d["success"] and d["timeframe"] in se.HORIZON_DAYS
    assert (d["bull_case"]["rationale"] == "BRIEF-BULL") is uses_brief


def test_stale_forecast_casts_no_model_vote(monkeypatch):
    import pandas as pd
    fake = types.SimpleNamespace(predict_latest=lambda s: {
        "date": "2026-05-06", "predictions": {"predicted_5d_return": 0.021}})
    monkeypatch.setitem(sys.modules, "predictor", fake)
    monkeypatch.setattr(se, "has_model", lambda s: True)
    assert rec_eng._model_vote("NVDA", pd.Timestamp("2026-10-30")) is None
    vote = rec_eng._model_vote("NVDA", pd.Timestamp("2026-05-08"))
    assert vote[0] == 1.0
