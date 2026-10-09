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
from grading import grade_fundamentals


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
    assert v["inputs"] == {"P/E (TTM)": -12.5}


def test_outliers_are_clamped():
    """An ROE of 193% shouldn't single-handedly decide profitability."""
    r = grade_fundamentals({"roeTTM": 193, "netProfitMarginTTM": -20})
    assert r["metrics"]["profitability"]["grade"] == "C"     # (50 − 20) / 2 = 15 → C


def test_real_zero_is_kept():
    r = grade_fundamentals({"epsGrowthTTMYoy": 0, "revenueGrowthTTMYoy": 0})
    assert r["metrics"]["growth"]["grade"] == "F"            # not a default 10%


def test_overall_uses_only_graded_categories():
    r = grade_fundamentals({"peBasicExclExtraTTM": 12, "totalDebt/totalEquityQuarterly": 0.1})
    assert r["graded_categories"] == 2
    assert r["overall_grade"] == "A" and r["average_score"] == 95


@pytest.mark.parametrize("pe, grade", [(10, "A"), (15, "B"), (19.9, "B"), (24, "C"), (30, "D"), (50, "F")])
def test_pe_bands(pe, grade):
    assert grade_fundamentals({"peBasicExclExtraTTM": pe})["metrics"]["valuation"]["grade"] == grade
