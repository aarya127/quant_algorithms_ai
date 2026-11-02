"""
tests/test_membership.py

Point-in-time index membership (data_pipelines/membership.py) and the
survivorship-aware pieces of the portfolio engine — synthetic data, no network.
"""
import numpy as np
import pandas as pd
import pytest

import membership as mb
from cross_section import equal_weights, simulate_portfolio, eligible_frame


def _spells():
    return pd.DataFrame({
        "ticker": ["AAA", "BBB", "OLD", "GONE", "BRK.B"],
        "start": pd.to_datetime(["2015-01-01", "2022-03-01", "2010-01-01", "2010-01-01", "2010-01-01"]),
        "end": pd.to_datetime([None, None, "2023-06-30", "2021-06-30", None]),
    })


class TestSpells:
    def test_window_clips_and_keeps_leavers(self):
        s = mb.spells_in_window(_spells(), "2021-01-01", "2026-01-01")
        assert set(s["ticker"]) == {"AAA", "BBB", "OLD", "GONE", "BRK-B"}     # yfinance-style ticker
        gone = s[s.ticker == "GONE"].iloc[0]
        assert gone["start"] == pd.Timestamp("2021-01-01") and gone["end"] == pd.Timestamp("2021-06-30")
        assert s[s.ticker == "AAA"]["end"].item() == pd.Timestamp("2026-01-01")

    def test_spell_ended_before_the_window_is_dropped(self):
        assert "GONE" not in set(mb.spells_in_window(_spells(), "2022-01-01", "2026-01-01")["ticker"])

    def test_rename_used_only_when_its_history_covers_the_spell(self, monkeypatch):
        monkeypatch.setattr(mb, "RENAMES", {"OLD": "NEW", "GONE": "LATE"})
        s = mb.spells_in_window(_spells(), "2021-01-01", "2026-01-01")
        avail = {"AAA": (pd.Timestamp("2020-11-02"), pd.Timestamp("2026-01-01")),
                 "NEW": (pd.Timestamp("2020-11-02"), pd.Timestamp("2026-01-01")),
                 "LATE": (pd.Timestamp("2024-01-02"), pd.Timestamp("2026-01-01"))}
        p = mb.price_symbols(s, avail).set_index("ticker")["price_symbol"]
        assert p["AAA"] == "AAA" and p["OLD"] == "NEW"
        assert pd.isna(p["GONE"])                      # LATE starts long after GONE's spell
        assert pd.isna(p["BBB"])                       # no prices at all


class TestFlagsAndCoverage:
    DATES = pd.bdate_range("2021-01-01", "2021-12-31")

    def _priced(self):
        s = mb.spells_in_window(_spells(), "2021-01-01", "2021-12-31")
        s["price_symbol"] = s["ticker"].where(s["ticker"] != "GONE")
        return s

    def test_in_index_marks_only_member_dates(self):
        f = mb.in_index(self.DATES, self._priced())
        old = f[f.symbol == "OLD"]["Date"]
        assert old.min() == self.DATES[0] and old.max() == self.DATES[-1]
        assert "GONE" not in set(f["symbol"])
        assert len(f[f.symbol == "BBB"]) == (self.DATES >= "2022-03-01").sum()   # 0: joins later

    def test_coverage_counts_missing_member_days(self):
        r = mb.coverage_report(self._priced(), self.DATES)
        gone_days = (self.DATES <= "2021-06-30").sum()
        total = 3 * len(self.DATES) + gone_days                 # AAA, OLD, BRK-B all year; GONE half
        assert r["missing"] == ["GONE"]
        assert r["missing_member_day_share"] == pytest.approx(gone_days / total)


def _frame(rows, cols=("A", "B")):
    return pd.DataFrame(rows, columns=list(cols), index=pd.bdate_range("2026-01-05", periods=len(rows)),
                        dtype=float)


class TestSurvivorshipAwarePortfolio:
    def test_equal_weight_only_over_members(self):
        closes = _frame([[10, 10]])
        w = equal_weights(closes, every=1, eligible=_frame([[1, 0]]).astype(bool))
        assert w.iloc[0].tolist() == [1.0, 0.0]

    def test_eligible_frame_defaults_to_everyone(self):
        panel = pd.DataFrame({"Date": pd.bdate_range("2026-01-05", periods=1).repeat(2),
                              "symbol": ["A", "B"], "Close": [1.0, 2.0]})
        assert eligible_frame(panel, _frame([[1, 2]])).all().all()

    def test_holding_whose_prices_end_is_cashed_out_at_its_last_price(self):
        # B is acquired: last trade at 15, then no data
        o = _frame([[10, 10], [10, 10], [10, np.nan], [10, np.nan]])
        c = _frame([[10, 10], [10, 15], [10, np.nan], [10, np.nan]])
        w = _frame([[0.5, 0.5]] + [[np.nan, np.nan]] * 3)
        daily, _ = simulate_portfolio(o, c, w, 1000, cost=0.0)
        assert daily["equity"].iloc[-1] == pytest.approx(500 + 750)
        assert daily["names"].iloc[-1] == 1          # only A still held

    def test_a_cashed_out_name_isnt_bought_back(self):
        o = _frame([[10, 10], [10, 10], [10, np.nan], [10, np.nan]])
        c = _frame([[10, 10], [10, 15], [10, np.nan], [10, np.nan]])
        w = _frame([[0.5, 0.5], [np.nan] * 2, [0.5, 0.5], [np.nan] * 2])
        daily, reb = simulate_portfolio(o, c, w, 1000, cost=0.0)
        # day-3 rebalance can't buy B (no price): half the money stays in cash
        assert daily["invested"].iloc[-1] == pytest.approx(0.5, abs=0.01)
