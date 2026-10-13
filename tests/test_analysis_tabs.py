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
    """2 years of synthetic closes with an upward drift, no trained model."""
    rng = np.random.default_rng(0)
    closes = 100 * np.exp(np.cumsum(rng.normal(0.0008, 0.015, 504)))
    monkeypatch.setattr(se, "_history", lambda s: (closes, "USD"))
    monkeypatch.setattr(se, "has_model", lambda s: False)
    se._cache.clear()
    return closes


@pytest.mark.parametrize("tf", list(se.HORIZON_DAYS))
def test_targets_ordered_and_probabilities_sum(prices, tf):
    r = se.compute_scenarios("TEST", tf)
    s = r["scenarios"]
    assert s["bear_case"]["price_target"] < s["base_case"]["price_target"] < s["bull_case"]["price_target"]
    probs = [s[k]["probability"] for k in ("bull_case", "base_case", "bear_case")]
    assert all(0 <= p <= 100 for p in probs)
    assert sum(probs) == pytest.approx(100, abs=0.2)


def test_long_horizons_never_certain(prices):
    """Overlapping windows gave 0% bear / 100% up at 1Y; the normal approximation doesn't."""
    r = se.compute_scenarios("TEST", "1Y")
    assert r["probability_method"].startswith("normal")
    assert 0.05 < r["p_up"] < 0.95
    assert r["scenarios"]["bear_case"]["probability"] == pytest.approx(15.9, abs=0.1)


def test_short_horizons_use_the_tickers_own_distribution(prices):
    r = se.compute_scenarios("TEST", "1W")
    assert r["probability_method"].startswith("empirical")
    # roughly normal synthetic returns → about 16% beyond ±1σ each side
    assert 10 < r["scenarios"]["bull_case"]["probability"] < 22


def test_model_vol_is_already_daily(monkeypatch):
    """predicted_vol_5d is a std of daily returns: it must not be divided by √5."""
    fake = types.SimpleNamespace(predict_latest=lambda s: {
        "date": "2026-10-07", "signal": "long",
        "predictions": {"predicted_5d_return": 0.03, "predicted_vol_5d": 0.02}})
    monkeypatch.setitem(sys.modules, "predictor", fake)
    drift, vol, meta = se._model_overlay("NVDA")
    assert vol == pytest.approx(0.02)
    assert drift == pytest.approx(0.03 / 5 * se.DRIFT_SHRINK)
    assert meta["prediction_date"] == "2026-10-07"


def test_canadian_symbol_uses_tsx_listing(monkeypatch):
    seen = {}

    class Ticker:
        def __init__(self, t):
            seen["t"] = t
            self.history_metadata = {"currency": "CAD"}

        def history(self, **kw):
            import pandas as pd
            return pd.DataFrame({"Close": np.linspace(100, 110, 300)})

    monkeypatch.setattr(se, "yf", types.SimpleNamespace(Ticker=Ticker))
    closes, currency = se._history("TD")
    assert seen["t"] == "TD.TO" and currency == "CAD"


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
