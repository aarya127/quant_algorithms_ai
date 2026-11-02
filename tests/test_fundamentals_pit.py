"""
tests/test_fundamentals_pit.py

Point-in-time fundamentals and earnings surprises
(data_pipelines/fundamentals_pit.py) on synthetic Finnhub-style filings — no
network. The rule under test everywhere: a number is usable only from the
business day after it was filed, as it was first reported.
"""
import numpy as np
import pandas as pd
import pytest

import fundamentals_pit as fp


def _rep(year, quarter, end, filed, start="2020-01-01", ic=None, bs=None, cf=None):
    to_items = lambda d: [{"concept": f"us-gaap_{k}", "value": v} for k, v in (d or {}).items()]
    return {"year": year, "quarter": quarter, "startDate": start, "endDate": end, "filedDate": filed,
            "report": {"ic": to_items(ic), "bs": to_items(bs), "cf": to_items(cf)}}


def _company():
    """FY = calendar year. YTD income: Q1 10, Q2 22, Q3 36, FY 50 (2021);
    Q1 12, Q2 26 (2022). Assets 100 → 110 a year later."""
    quarterly = [
        _rep(2021, 1, "2021-03-31", "2021-05-01", ic={"Revenues": 10, "NetIncomeLoss": 2, "EarningsPerShareDiluted": 0.2},
             bs={"Assets": 100, "StockholdersEquity": 40, "Liabilities": 60}, cf={"NetCashProvidedByUsedInOperatingActivities": 3}),
        _rep(2021, 2, "2021-06-30", "2021-08-01", ic={"Revenues": 22, "NetIncomeLoss": 4, "EarningsPerShareDiluted": 0.4},
             bs={"Assets": 102, "StockholdersEquity": 41, "Liabilities": 61}, cf={"NetCashProvidedByUsedInOperatingActivities": 6}),
        _rep(2021, 3, "2021-09-30", "2021-11-01", ic={"Revenues": 36, "NetIncomeLoss": 6, "EarningsPerShareDiluted": 0.6},
             bs={"Assets": 104, "StockholdersEquity": 42, "Liabilities": 62}, cf={"NetCashProvidedByUsedInOperatingActivities": 9}),
        _rep(2022, 1, "2022-03-31", "2022-05-02", ic={"Revenues": 12, "NetIncomeLoss": 3, "EarningsPerShareDiluted": 0.3},
             bs={"Assets": 110, "StockholdersEquity": 45, "Liabilities": 65}, cf={"NetCashProvidedByUsedInOperatingActivities": 4}),
        _rep(2022, 2, "2022-06-30", "2022-08-01", ic={"Revenues": 26, "NetIncomeLoss": 6, "EarningsPerShareDiluted": 0.6},
             cf={"NetCashProvidedByUsedInOperatingActivities": 8}),           # no balance sheet parsed
    ]
    annual = [_rep(2021, 0, "2021-12-31", "2022-02-15", ic={"Revenues": 50, "NetIncomeLoss": 8, "EarningsPerShareDiluted": 0.8},
                   bs={"Assets": 106, "StockholdersEquity": 43, "Liabilities": 63},
                   cf={"NetCashProvidedByUsedInOperatingActivities": 12})]
    return quarterly, annual


class TestFilings:
    def test_one_row_per_filing_with_shares_from_eps(self):
        f = fp.filings_frame(*_company())
        assert list(zip(f.fy, f.fq)) == [(2021, 1), (2021, 2), (2021, 3), (2021, 4), (2022, 1), (2022, 2)]
        assert f.loc[0, "shares"] == pytest.approx(10)            # 2 / 0.2

    def test_amendment_keeps_the_first_filing(self):
        q, a = _company()
        q.append(_rep(2021, 1, "2021-03-31", "2021-07-01", ic={"Revenues": 999, "NetIncomeLoss": 2}))
        f = fp.filings_frame(q, a)
        assert f[(f.fy == 2021) & (f.fq == 1)]["revenue"].item() == 10

    def test_gross_profit_from_cost_of_revenue(self):
        f = fp.filings_frame([_rep(2021, 1, "2021-03-31", "2021-05-01",
                                   ic={"RevenueFromContractWithCustomerExcludingAssessedTax": 10, "CostOfRevenue": 4})], [])
        assert f.loc[0, "revenue"] == 10 and f.loc[0, "gross_profit"] == 6


class TestTTM:
    def test_ttm_from_ytd_and_the_prior_year(self):
        t = fp.ttm_frame(fp.filings_frame(*_company()))
        by = dict(zip(t["filed"].dt.date.astype(str), t["revenue"]))
        assert by["2022-02-15"] == 50                              # 10-K: the year
        assert by["2022-05-02"] == 50 + 12 - 10                    # Q1 2022
        assert by["2022-08-01"] == 50 + 26 - 22                    # Q2 2022

    def test_missing_prior_year_piece_carries_the_last_ttm(self):
        t = fp.ttm_frame(fp.filings_frame(*_company()))
        # 2021 Q1-Q3 have no 2020 filings to build from; nothing earlier to carry either
        assert np.isnan(t["revenue"].iloc[0])
        q, a = _company()
        q = [r for r in q if not (r["year"] == 2021 and r["quarter"] == 2)]
        t = fp.ttm_frame(fp.filings_frame(q, a))
        q2 = t[t["filed"] == pd.Timestamp("2022-08-01")]["revenue"].item()
        assert q2 == 50 + 12 - 10                                 # Q1 2022's TTM, carried

    def test_missing_balance_sheet_carries_from_an_earlier_filing(self):
        t = fp.ttm_frame(fp.filings_frame(*_company()))
        assert t["assets"].iloc[-1] == 110 and t["assets_year_ago"].iloc[-1] == 102

    def test_available_the_business_day_after_filing(self):
        t = fp.ttm_frame(fp.filings_frame(*_company()))
        assert t.loc[t["filed"] == pd.Timestamp("2021-05-01"), "available"].item() == pd.Timestamp("2021-05-03")


class TestDailyRatios:
    @staticmethod
    def _prices(start="2022-04-25", n=20, close=5.0):
        return pd.DataFrame({"Date": pd.bdate_range(start, periods=n), "Close_raw": close})

    def test_values_appear_only_once_available(self):
        t = fp.ttm_frame(fp.filings_frame(*_company()))
        d = fp.fundamentals_for(self._prices(), t)
        before = d[d.Date < "2022-05-03"]["roa"]
        after = d[d.Date >= "2022-05-03"]["roa"]
        assert (before.round(6) == round(8 / 106, 6)).all()       # the 10-K's TTM
        assert (after.round(6) == round(9 / 110, 6)).all()        # Q1 2022: NI TTM 8+3-2

    def test_market_cap_uses_todays_share_basis(self):
        t = fp.ttm_frame(fp.filings_frame(*_company()))
        splits = pd.Series([2.0], index=pd.to_datetime(["2022-09-01"]))
        d = fp.fundamentals_for(self._prices(), t, splits)
        row = d[d.Date == "2022-05-04"].iloc[0]
        # shares 10 (3 / 0.3) → 20 in today's basis; price 5 → cap 100; NI TTM 9
        assert row["earnings_yield"] == pytest.approx(9 / 100)

    def test_negative_equity_leaves_roe_and_book_to_market_blank(self):
        t = fp.ttm_frame(fp.filings_frame(*_company()))
        t["equity"] = -5.0
        d = fp.fundamentals_for(self._prices(), t)
        assert d["roe"].isna().all() and d["book_to_market"].isna().all() and d["roa"].notna().all()

    def test_split_factor_counts_only_later_splits(self):
        s = pd.Series([4.0, 10.0], index=pd.to_datetime(["2021-07-20", "2024-06-10"]))
        assert fp.split_factor_after(s, pd.Timestamp("2021-01-01")) == 40
        assert fp.split_factor_after(s, pd.Timestamp("2022-01-01")) == 10
        assert fp.split_factor_after(s, pd.Timestamp("2025-01-01")) == 1

    def test_cached_splits_keep_their_ratios(self, monkeypatch):
        monkeypatch.setattr(fp, "_cached_json", lambda name, fetch: {"2024-06-10": 10.0})
        s = fp.fetch_splits("NVDA")
        assert s.iloc[0] == 10.0 and s.index[0] == pd.Timestamp("2024-06-10")


class TestEarnings:
    DATES = pd.bdate_range("2026-05-18", periods=6)               # Mon 18 → Mon 25

    def test_after_close_report_counts_from_the_next_session(self):
        ev = pd.DataFrame({"announced": [pd.Timestamp("2026-05-20 16:05")], "surprise": [0.06]})
        e = fp.earnings_for(self.DATES, ev).set_index("Date")
        assert np.isnan(e.loc["2026-05-20", "eps_surprise"]) and e.loc["2026-05-21", "eps_surprise"] == 0.06
        assert e.loc["2026-05-21", "days_since_earnings"] == 0 and e.loc["2026-05-25", "days_since_earnings"] == 2

    def test_pre_market_report_counts_the_same_day(self):
        ev = pd.DataFrame({"announced": [pd.Timestamp("2026-05-20 07:00")], "surprise": [-0.02]})
        e = fp.earnings_for(self.DATES, ev).set_index("Date")
        assert e.loc["2026-05-20", "eps_surprise"] == -0.02

    def test_surprise_is_clipped(self):
        ev = pd.DataFrame({"announced": [pd.Timestamp("2026-05-18 07:00")], "surprise": [7.5]})
        assert fp.earnings_for(self.DATES, ev)["eps_surprise"].max() == 1.0
