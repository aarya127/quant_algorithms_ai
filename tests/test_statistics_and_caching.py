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
