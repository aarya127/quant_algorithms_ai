"""
tests/test_cross_section.py

Multi-asset portfolio simulation and factor diagnostics (backtesting/cross_section.py)
and the universe panel builder (data_pipelines/universe.py), on synthetic prices.
"""
import numpy as np
import pandas as pd
import pytest

from cross_section import (top_weights, equal_weights, simulate_portfolio, portfolio_stats,
                           forward_returns, rank_ic, ic_summary, wide)


def _frame(rows, cols=("A", "B")):
    return pd.DataFrame(rows, columns=list(cols), index=pd.bdate_range("2026-01-05", periods=len(rows)),
                        dtype=float)


class TestSelection:
    def test_top_fraction_equal_weighted(self):
        scores = _frame([[5, 4, 3, 2, 1, 0, -1, -2, -3, -4]], cols="ABCDEFGHIJ")
        w = top_weights(scores, top_frac=0.2, every=1)
        assert w.iloc[0].to_dict() == {**{k: 0.0 for k in "CDEFGHIJ"}, "A": 0.5, "B": 0.5}

    def test_unscored_names_cant_be_picked_and_thin_dates_skip(self):
        scores = _frame([[np.nan, 1, 2, 3, 4, 5], [np.nan] * 5 + [9]], cols="ABCDEF")
        w = top_weights(scores, top_frac=0.2, every=1, min_names=5)
        assert w.iloc[0]["F"] == 1.0 and w.iloc[0]["A"] == 0.0
        assert w.iloc[1].isna().all()

    def test_decisions_only_on_schedule(self):
        scores = _frame([[1, 2]] * 7)
        w = top_weights(scores, top_frac=0.5, every=3, min_names=1)
        assert w.notna().all(axis=1).tolist() == [True, False, False, True, False, False, True]

    def test_equal_weights_skip_missing_prices(self):
        w = equal_weights(_frame([[10, np.nan]]), every=1)
        assert w.iloc[0].tolist() == [1.0, 0.0]


class TestSimulation:
    O = _frame([[10, 20], [11, 20], [12, 20], [12, 22]])
    C = _frame([[10, 20], [11, 20], [12, 21], [12, 22]])

    def test_decision_trades_at_next_open(self):
        w = _frame([[1, 0], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]])
        daily, reb = simulate_portfolio(self.O, self.C, w, 1000, cost=0.0)
        assert reb[0]["date"] == "2026-01-06" and daily["equity"][0] == 1000
        # bought A at 11 on day 1; A closes 12 on day 3
        assert daily["equity"].iloc[-1] == pytest.approx(1000 * 12 / 11)

    def test_costs_are_charged_on_traded_value(self):
        w = _frame([[0.5, 0.5]] + [[np.nan, np.nan]] * 3)
        daily, reb = simulate_portfolio(self.O, self.C, w, 1000, cost=0.001)
        assert reb[0]["cost"] == pytest.approx(1.0) and reb[0]["turnover"] == pytest.approx(1.0)

    def test_holdings_drift_between_rebalances(self):
        o = _frame([[10, 10], [10, 10], [20, 10], [20, 10]])
        w = _frame([[0.5, 0.5]] + [[np.nan, np.nan]] * 3)
        daily, reb = simulate_portfolio(o, o, w, 1000, cost=0.0)
        assert len(reb) == 1 and daily["equity"].iloc[-1] == pytest.approx(1500)

    def test_later_prices_cant_change_earlier_equity(self):
        w = _frame([[1, 0], [0, 1], [np.nan, np.nan], [np.nan, np.nan]])
        a, _ = simulate_portfolio(self.O, self.C, w, 1000, cost=0.0005)
        o2, c2 = self.O.copy(), self.C.copy()
        o2.iloc[3], c2.iloc[3] = [500, 1], [1, 500]
        b, _ = simulate_portfolio(o2, c2, w, 1000, cost=0.0005)
        assert a["equity"][:3].tolist() == b["equity"][:3].tolist()

    def test_a_name_without_a_price_stays_put(self):
        o = _frame([[10, 10], [10, np.nan], [10, 10]])
        c = _frame([[10, 10], [10, np.nan], [10, 12]])
        w = _frame([[0.5, 0.5], [np.nan, np.nan], [np.nan, np.nan]])
        daily, _ = simulate_portfolio(o, c, w, 1000, cost=0.0)
        # B couldn't be bought on day 1, so half the money sits in cash
        assert daily["equity"].iloc[-1] == pytest.approx(1000) and daily["invested"].iloc[1] == pytest.approx(0.5)

    def test_turnover_ignores_the_initial_buy(self):
        daily, reb = simulate_portfolio(self.O, self.C, _frame([[1, 0], [0, 1], [np.nan] * 2, [np.nan] * 2]),
                                        1000, cost=0.0)
        s = portfolio_stats(daily, reb, 1000)
        assert s["rebalances"] == 2 and s["turnover_per_year"] == pytest.approx(2.0 * 252 / 4, rel=0.01)


class TestIC:
    def test_perfect_and_reversed_rankings(self):
        scores = _frame([list(range(10))], cols="ABCDEFGHIJ")
        fwd = _frame([list(range(10))], cols="ABCDEFGHIJ")
        assert rank_ic(scores, fwd, every=1).iloc[0] == pytest.approx(1.0)
        assert rank_ic(-scores, fwd, every=1).iloc[0] == pytest.approx(-1.0)

    def test_too_few_names_give_no_sample(self):
        assert rank_ic(_frame([[1, 2]]), _frame([[1, 2]]), every=1).empty

    def test_forward_returns_look_ahead_by_h(self):
        fwd = forward_returns(_frame([[10, 1], [11, 1], [12, 2]]), 2)
        assert fwd.iloc[0].tolist() == pytest.approx([0.2, 1.0]) and fwd.iloc[1:].isna().all().all()

    def test_ic_t_statistic(self):
        s = ic_summary(pd.Series([0.1, 0.2, 0.3]))
        assert s["ic_mean"] == pytest.approx(0.2) and s["ic_t"] == pytest.approx(0.2 / (0.1 / np.sqrt(3)))
        assert ic_summary(pd.Series([0.1]))["ic_t"] is None


class TestUniversePanel:
    @staticmethod
    def _bars(seed, n=320):
        rng = np.random.default_rng(seed)
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        return pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c,
                             "Volume": 1e6}, index=pd.bdate_range("2025-01-01", periods=n, name="Date"))

    def test_each_tickers_features_use_only_its_own_rows(self):
        from universe import ticker_features, to_panel
        a, b = self._bars(1), self._bars(2)
        both = to_panel({"A": ticker_features(a), "B": ticker_features(b)})
        alone = to_panel({"B": ticker_features(b)})
        pd.testing.assert_frame_equal(both[both.symbol == "B"].reset_index(drop=True), alone,
                                      check_dtype=False)

    def test_panel_has_features_and_targets(self):
        from universe import ticker_features, to_panel
        p = to_panel({"A": ticker_features(self._bars(3))})
        for col in ("momentum_12m", "realized_vol_60d", "reversal_1w", "price_to_sma200",
                    "target_5d", "target_dir_1d"):
            assert col in p.columns
        assert p["momentum_12m"].notna().sum() == 320 - 257
        assert wide(p, "Close").shape == (320, 1)
