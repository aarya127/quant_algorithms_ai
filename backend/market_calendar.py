"""
market_calendar.py — pure helpers behind /api/calendar (no network).

  nyse_holidays(year)       rule-generated market holidays, so they never run out
  chunk_ranges(...)         date windows for the Finnhub earnings sweep
  earnings_event(row, ...)  one Finnhub earnings row -> a calendar event
  parse_symbols(raw)        sanitise the ?symbols= watchlist parameter

FOMC / CPI / jobs-report dates can't be derived — the Fed and BLS announce them —
so those stay in economic_events.json and need a yearly refresh.
"""
import datetime
import re

_SYMBOL_RE = re.compile(r'^[A-Z0-9][A-Z0-9.\-]{0,11}$')
MAX_WATCHLIST = 60

_HOUR = {'bmo': 'before open', 'amc': 'after close', 'dmh': 'during market hours'}


def _easter(year):
    """Gregorian Easter Sunday (anonymous / Meeus algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return datetime.date(year, month, day)


def _nth_weekday(year, month, weekday, n):
    """n-th `weekday` (Mon=0) of a month; n=-1 means the last one."""
    if n > 0:
        first = datetime.date(year, month, 1)
        return first + datetime.timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))
    nxt = datetime.date(year + (month == 12), month % 12 + 1, 1)
    last = nxt - datetime.timedelta(days=1)
    return last - datetime.timedelta(days=(last.weekday() - weekday) % 7)


def _observed(day):
    """NYSE rule: Saturday holidays close the Friday before, Sunday ones the Monday after."""
    if day.weekday() == 5:
        return day - datetime.timedelta(days=1)
    if day.weekday() == 6:
        return day + datetime.timedelta(days=1)
    return day


def nyse_holidays(year):
    """Full-day NYSE closures plus the Black Friday early close, as calendar events."""
    thanksgiving = _nth_weekday(year, 11, 3, 4)
    closures = [
        ("New Year's Day", datetime.date(year, 1, 1)),
        ('Martin Luther King Jr. Day', _nth_weekday(year, 1, 0, 3)),
        ("Presidents' Day", _nth_weekday(year, 2, 0, 3)),
        ('Good Friday', _easter(year) - datetime.timedelta(days=2)),
        ('Memorial Day', _nth_weekday(year, 5, 0, -1)),
        ('Juneteenth', datetime.date(year, 6, 19)),
        ('Independence Day', datetime.date(year, 7, 4)),
        ('Labor Day', _nth_weekday(year, 9, 0, 1)),
        ('Thanksgiving Day', thanksgiving),
        ('Christmas Day', datetime.date(year, 12, 25)),
    ]
    events = []
    for name, day in closures:
        # A Saturday New Year's Day is not made up on Friday Dec 31 (NYSE Rule 7.2)
        if name == "New Year's Day" and day.weekday() == 5:
            continue
        events.append({'date': _observed(day), 'type': 'Holiday',
                       'description': f'{name} - Markets Closed', 'importance': 'medium'})
    events.append({'date': thanksgiving + datetime.timedelta(days=1), 'type': 'Holiday',
                   'description': 'Day after Thanksgiving - Early Market Close (1pm ET)',
                   'importance': 'low'})
    return events


def chunk_ranges(from_date, to_date, days):
    """Consecutive [start, end] windows of at most `days` days covering the range."""
    out, start = [], from_date
    while start <= to_date:
        end = min(start + datetime.timedelta(days=days - 1), to_date)
        out.append((start, end))
        start = end + datetime.timedelta(days=1)
    return out


def earnings_event(row, group):
    """Finnhub earningsCalendar row -> calendar event tagged Watchlist / Top company."""
    symbol = row['symbol']
    details = []
    if row.get('quarter') and row.get('year'):
        details.append(f"Q{row['quarter']} {row['year']}")
    if _HOUR.get(row.get('hour')):
        details.append(_HOUR[row['hour']])
    if row.get('epsEstimate') is not None:
        details.append(f"Est EPS ${row['epsEstimate']:.2f}")
    suffix = f" ({' · '.join(details)})" if details else ''
    return {
        'symbol': symbol,
        'date': row['date'],
        'type': 'Earnings',
        'description': f'{symbol} Earnings Report{suffix}',
        'importance': 'high' if group == 'Watchlist' else 'medium',
        'group': group,
    }


def parse_symbols(raw):
    """'nvda, td,,BAD$' -> ['NVDA', 'TD'] (deduped, capped at MAX_WATCHLIST)."""
    seen = []
    for part in (raw or '').split(','):
        sym = part.strip().upper()
        if _SYMBOL_RE.match(sym) and sym not in seen:
            seen.append(sym)
    return seen[:MAX_WATCHLIST]
