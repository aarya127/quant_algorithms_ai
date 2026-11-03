"""
tests/test_news_windows.py

Alpha Vantage news batches (extractor._fixed_windows) sit on fixed calendar
boundaries, so a closed window's cache file is reused by the next daily run and
only the still-open window costs an API call.
"""
import datetime

from extractor import _fixed_windows          # data_pipelines/ (conftest)


def _shift(iso, days):
    return (datetime.date.fromisoformat(iso) + datetime.timedelta(days=days)).isoformat()


class TestFixedWindows:
    def test_windows_cover_the_range_contiguously(self):
        w = _fixed_windows("2025-11-03", "2026-11-03")
        assert w[0][0] <= "2025-11-03" and w[-1][1] == "2026-11-03"
        for (_, prev_end), (nxt_start, _) in zip(w, w[1:]):
            assert _shift(prev_end, 1) == nxt_start
        assert all(_shift(s, 29) >= e for s, e in w)

    def test_next_days_run_reuses_every_closed_window(self):
        today = _fixed_windows("2025-11-03", "2026-11-03")
        tomorrow = _fixed_windows("2025-11-04", "2026-11-04")
        new = set(tomorrow) - set(today)
        assert len(new) == 1 and max(new)[1] == "2026-11-04"

    def test_single_day_range(self):
        [(s, e)] = _fixed_windows("2026-11-03", "2026-11-03")
        assert s <= e == "2026-11-03"
