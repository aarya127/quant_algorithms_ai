"""
Network-free tests for the /api/calendar helpers (backend/market_calendar.py)
and the economic-events file.
"""
import datetime
import json
from pathlib import Path

from common import TOP_COMPANIES
from market_calendar import chunk_ranges, earnings_event, nyse_holidays, parse_symbols

D = datetime.date


def _closures(year):
    return [e['date'] for e in nyse_holidays(year) if 'Closed' in e['description']]


class TestNyseHolidays:
    def test_2026_matches_nyse_schedule(self):
        assert _closures(2026) == [
            D(2026, 1, 1), D(2026, 1, 19), D(2026, 2, 16), D(2026, 4, 3), D(2026, 5, 25),
            D(2026, 6, 19), D(2026, 7, 3), D(2026, 9, 7), D(2026, 11, 26), D(2026, 12, 25)]

    def test_2027_weekend_observances(self):
        assert _closures(2027) == [
            D(2027, 1, 1), D(2027, 1, 18), D(2027, 2, 15), D(2027, 3, 26), D(2027, 5, 31),
            D(2027, 6, 18), D(2027, 7, 5), D(2027, 9, 6), D(2027, 11, 25), D(2027, 12, 24)]

    def test_saturday_new_year_not_observed(self):
        # 2022-01-01 was a Saturday: NYSE stayed open on Friday 2021-12-31
        assert D(2021, 12, 31) not in _closures(2022)
        assert D(2022, 1, 1) not in _closures(2022)

    def test_black_friday_early_close(self):
        early = [e for e in nyse_holidays(2027) if e['importance'] == 'low']
        assert [e['date'] for e in early] == [D(2027, 11, 26)]


class TestChunkRanges:
    def test_covers_range_without_gaps(self):
        chunks = chunk_ranges(D(2026, 11, 2), D(2026, 11, 20), 7)
        assert chunks == [(D(2026, 11, 2), D(2026, 11, 8)), (D(2026, 11, 9), D(2026, 11, 15)),
                          (D(2026, 11, 16), D(2026, 11, 20))]


class TestEarningsEvent:
    def test_description_and_group(self):
        row = {'symbol': 'AMD', 'date': '2026-11-03', 'hour': 'amc', 'quarter': 3,
               'year': 2026, 'epsEstimate': 1.97}
        e = earnings_event(row, 'Top company')
        assert e['description'] == 'AMD Earnings Report (Q3 2026 · after close · Est EPS $1.97)'
        assert (e['group'], e['importance']) == ('Top company', 'medium')
        assert earnings_event(row, 'Watchlist')['importance'] == 'high'

    def test_missing_fields(self):
        e = earnings_event({'symbol': 'X', 'date': '2026-12-01', 'hour': ''}, 'Watchlist')
        assert e['description'] == 'X Earnings Report'


class TestParseSymbols:
    def test_sanitises_and_dedupes(self):
        assert parse_symbols(' nvda, td,,NVDA, BRK.B, bad$, ') == ['NVDA', 'TD', 'BRK.B']

    def test_empty(self):
        assert parse_symbols(None) == [] and parse_symbols('') == []


def test_top_companies_match_large50_universe():
    from universe import UNIVERSES
    assert TOP_COMPANIES == UNIVERSES['large50']


def test_economic_events_are_weekdays_and_sorted():
    path = Path(__file__).parent.parent / 'backend' / 'economic_events.json'
    events = json.loads(path.read_text())['events']
    dates = [D.fromisoformat(e['date']) for e in events]
    assert dates == sorted(dates)
    assert all(d.weekday() < 5 for d in dates)
    assert all(e['type'] != 'Holiday' for e in events)  # holidays are generated now
