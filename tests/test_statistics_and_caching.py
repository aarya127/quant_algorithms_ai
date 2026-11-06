"""
Network-free tests for the Statistics tab units, the daily-range chart slicing,
and the shared TTL cache (no Yahoo/Finnhub calls — inputs are passed in).
"""
import sys
import threading
import time
from pathlib import Path

import pandas as pd
from cachetools import TTLCache

# conftest puts supervised/ on sys.path, and its data.py beats the top-level
# data/ namespace package — hide it while importing ours, then restore it.
_saved_path, _shadowed = sys.path[:], sys.modules.pop('data', None)
sys.path[:] = [str(Path(__file__).parent.parent)] + [
    p for p in sys.path if not p.endswith('supervised')]
from data.company_statistics import (
    get_company_statistics, format_statistics_for_display, has_any_statistic,
)
from data.financial_statements import latest_ttm
from data.charts import _slice_start
from services import _ttl_cached, _is_real_yf_info
sys.path[:] = _saved_path
if _shadowed is not None:
    sys.modules['data'] = _shadowed


class TestStatisticUnits:
    def test_fraction_above_one_is_scaled(self):
        # NVDA-style ROE of 117% arrives from yfinance as 1.17
        stats = get_company_statistics('X', info={'quoteType': 'EQUITY', 'returnOnEquity': 1.17},
                                       financials={})
        assert format_statistics_for_display(stats)['returns']['ROE'] == '117.0%'

    def test_finnhub_percent_is_converted_to_fraction(self):
        stats = get_company_statistics('X', info={}, financials={'metric': {'roiTTM': 0.6,
                                                                             'roeTTM': 110.11}})
        assert stats['returns']['roic'] == 0.006
        shown = format_statistics_for_display(stats)['returns']
        assert shown['ROIC'] == '0.6%' and shown['ROE'] == '110.1%'

    def test_yfinance_wins_over_finnhub(self):
        stats = get_company_statistics('X', info={'returnOnEquity': 0.2},
                                       financials={'metric': {'roeTTM': 50.0}})
        assert stats['returns']['roe'] == 0.2

    def test_dividend_yield_percent_from_yfinance(self):
        stats = get_company_statistics('X', info={'dividendYield': 0.49}, financials={})
        assert format_statistics_for_display(stats)['dividends']['Yield'] == '0.5%'

    def test_all_upstreams_empty_is_detectable(self):
        assert not has_any_statistic(get_company_statistics('X', info={}, financials={}))
        assert has_any_statistic(get_company_statistics('X', info={'marketCap': 1}, financials={}))


    def test_yfinance_debt_to_equity_percent_becomes_ratio(self):
        stats = get_company_statistics('X', info={'debtToEquity': 150.0}, financials={})
        assert stats['financial_health']['debt_to_equity'] == 1.5
        assert format_statistics_for_display(stats)['financial_health']['Debt/Equity'] == '1.50'

    def test_negative_currency_puts_sign_before_dollar(self):
        stats = get_company_statistics('X', info={'totalDebt': 1e9, 'totalCash': 3e9}, financials={})
        assert format_statistics_for_display(stats)['financial_health']['Net Debt'] == '-$2.00B'


class TestFinnhubFills:
    """Yahoo down: the Statistics tab fills from Finnhub's metric payload."""
    METRIC = {'marketCapitalization': 684007.3, 'enterpriseValue': 683682.3,
              'revenuePerShareTTM': 27.0, 'ebitdPerShareTTM': 5.4,
              'cashPerSharePerShareQuarterly': 9.0, 'bookValuePerShareQuarterly': 40.0,
              'totalDebt/totalEquityQuarterly': 0.1, 'revenueEmployeeTTM': 1.5,
              'psTTM': 15.0, 'pfcfShareTTM': 100.0, 'forwardPE': 39.8, 'pegTTM': 2.0,
              'evRevenueTTM': 15.0, 'grossMarginTTM': 50.0}

    def stats(self, **kw):
        return get_company_statistics('X', info={}, financials={'metric': self.METRIC},
                                      profile={'shareOutstanding': 1600.0}, **kw)

    def test_millions_are_scaled(self):
        p = self.stats()['profile']
        assert p['market_cap'] == 684007.3e6 and p['shares_outstanding'] == 1.6e9

    def test_per_share_metrics_become_totals(self):
        s = self.stats()
        assert s['profile']['revenue_ttm'] == 27.0 * 1.6e9
        assert s['profile']['employees'] == round(27.0 * 1.6e9 / 1.5e6)
        h = s['financial_health']
        assert h['cash'] == 9.0 * 1.6e9
        assert abs(h['total_debt'] - 0.1 * 40.0 * 1.6e9) < 1
        assert abs(h['net_debt'] - (h['total_debt'] - h['cash'])) < 1

    def test_derived_ratios(self):
        s = self.stats()
        assert abs(s['margins']['ebitda_margin'] - 0.2) < 1e-12
        assert abs(s['margins']['fcf_margin'] - 0.15) < 1e-12      # P/S / P/FCF
        assert abs(s['valuation_ttm']['ev_to_gross_profit'] - 30.0) < 1e-9
        assert s['valuation_forward']['forward_pe'] == 39.8

    def test_no_share_count_leaves_totals_empty(self):
        s = get_company_statistics('X', info={}, financials={'metric': self.METRIC})
        assert s['profile']['revenue_ttm'] is None and s['financial_health']['cash'] is None

    def test_filings_fill_cash_flow_and_returns(self):
        cf = {'revenue': 40e9, 'operating_income': 8e9, 'operating_cash_flow': 9e9, 'capex': 2e9,
              'free_cash_flow': 7e9, 'stock_comp': 2e9, 'total_assets': 80e9,
              'current_liabilities': 10e9, 'basis': 'ttm', 'period_end': '2026-09-26'}
        s = self.stats(cash_flow=cf)
        assert s['cash_flow']['capex_to_revenue'] == 0.05
        assert s['cash_flow']['fcf_after_sbc'] == 5e9
        assert s['returns']['rota'] == 0.1 and s['returns']['roce'] == 8 / 70
        shown = format_statistics_for_display(s)['cash_flow']
        assert shown['CapEx'] == '$2.00B' and shown['Period'] == 'TTM to 2026-09-26'

    def test_no_filings_hides_cash_flow(self):
        assert format_statistics_for_display(self.stats())['cash_flow']['Period'] is None


def _filing(year, quarter, end, cf=None, bs=None):
    return {'year': year, 'quarter': quarter, 'endDate': end + ' 00:00:00',
            'report': {'cf': [{'concept': k, 'value': v} for k, v in (cf or {}).items()],
                       'bs': [{'concept': k, 'value': v} for k, v in (bs or {}).items()],
                       'ic': []}}


OCF, CAPEX = 'us-gaap_NetCashProvidedByUsedInOperatingActivities', 'us-gaap_PaymentsToAcquirePropertyPlantAndEquipment'


class TestLatestTtm:
    """10-Q cash flows are year-to-date (AMD's real Q3 2026 numbers below)."""
    ANNUAL = [_filing(2025, 0, '2025-12-27', {OCF: 7709, CAPEX: 974}, {'us-gaap_Assets': 70})]
    YEAR_AGO = _filing(2025, 3, '2025-09-27', {OCF: 5109, CAPEX: 752})

    def test_ytd_plus_last_10k_minus_year_ago_ytd(self):
        latest = _filing(2026, 3, '2026-09-26', {OCF: 6523, CAPEX: 1845})
        out = latest_ttm([latest, self.YEAR_AGO], self.ANNUAL)
        assert out['basis'] == 'ttm' and out['period_end'] == '2026-09-26'
        assert out['operating_cash_flow'] == 9123 and out['capex'] == 2067
        assert out['free_cash_flow'] == 9123 - 2067
        assert out['total_assets'] == 70      # carried from the newest filing that has one

    def test_missing_year_ago_falls_back_to_10k(self):
        latest = _filing(2026, 3, '2026-09-26', {OCF: 6523, CAPEX: 1845})
        out = latest_ttm([latest], self.ANNUAL)
        assert out['basis'] == 'annual' and out['period_end'] == '2025-12-27'
        assert out['operating_cash_flow'] == 7709

    def test_latest_10k_is_used_as_is(self):
        newer = [_filing(2026, 0, '2026-12-26', {OCF: 10000, CAPEX: 2500})]
        out = latest_ttm([self.YEAR_AGO], self.ANNUAL + newer)
        assert out['basis'] == 'annual' and out['capex'] == 2500

    def test_da_parts_are_summed_and_outflows_positive(self):
        parts = {'us-gaap_OtherDepreciationAndAmortization': 600,
                 'us-gaap_AmortizationOfIntangibleAssets': 190, CAPEX: -974}
        out = latest_ttm([], [_filing(2025, 0, '2025-12-27', parts)])
        assert out['depreciation_amortization'] == 790 and out['capex'] == 974

    def test_no_filings(self):
        assert latest_ttm([], []) == {}


class TestDailySlice:
    def test_one_month_window(self):
        idx = pd.bdate_range('2025-01-01', '2026-11-03')
        hist = pd.DataFrame({'Close': range(len(idx))}, index=idx)
        start = _slice_start(hist, '1mo')
        assert hist.index[start] > pd.Timestamp('2026-10-03')
        assert hist.index[start - 1] <= pd.Timestamp('2026-10-03')

    def test_empty(self):
        assert _slice_start(pd.DataFrame(), '1y') == 0


class TestTtlCache:
    def test_invalid_results_are_not_cached(self):
        calls = []

        @_ttl_cached(TTLCache(maxsize=8, ttl=60))
        def fetch(key):
            calls.append(key)
            return None

        fetch('a'); fetch('a')
        assert len(calls) == 2

    def test_concurrent_misses_share_one_call(self):
        calls = []

        @_ttl_cached(TTLCache(maxsize=8, ttl=60))
        def fetch(key):
            calls.append(key)
            time.sleep(0.05)
            return {'ok': key}

        threads = [threading.Thread(target=fetch, args=('a',)) for _ in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert len(calls) == 1

    def test_yf_info_stub_detection(self):
        assert not _is_real_yf_info({'trailingPegRatio': None})
        assert not _is_real_yf_info(None)
        assert _is_real_yf_info({'quoteType': 'EQUITY'})
