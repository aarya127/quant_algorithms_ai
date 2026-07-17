"""
common.py — shared helpers used by app.py and the route blueprints.

Kept dependency-light (flask + stdlib only) so any blueprint can import it
without dragging in the data layer.
"""
import os

from flask import jsonify, request


def keys_txt_value(header_substring):
    """Local-dev fallback: read an API key from the project-root keys.txt.

    Finds the first line containing `header_substring` (case-insensitive) and
    returns the next non-empty line, matching the convention finnhub.py and
    alphavantage.py already use. Returns '' if the file or section is missing —
    callers should try the env var first.
    """
    path = os.path.join(os.path.dirname(__file__), '..', 'keys.txt')
    try:
        with open(path) as fh:
            lines = [ln.strip() for ln in fh.readlines()]
        for i, line in enumerate(lines):
            if header_substring.lower() in line.lower():
                for nxt in lines[i + 1:]:
                    if nxt:
                        return nxt
    except OSError:
        pass
    return ''

# Configuration
DEFAULT_STOCKS = ["NVDA", "TD", "ACDVF", "MSFT", "ENB", "RCI", "CVE", "HUBS", "MU", "CNSWF", "AMD"]

TIMEFRAMES = {
    '1W': {'days': 7, 'label': '1 Week'},
    '1M': {'days': 30, 'label': '1 Month'},
    '3M': {'days': 90, 'label': '3 Months'},
    '6M': {'days': 180, 'label': '6 Months'},
    '1Y': {'days': 365, 'label': '1 Year'},
}

# Dashboard "Market Overview" cards — single source of truth for both
# /api/indices (quotes) and /api/indices/history (sparklines).
MARKET_INDICES = [
    ('^GSPC',   'S&P 500'),
    ('^IXIC',   'NASDAQ'),
    ('^DJI',    'Dow Jones'),
    ('^GSPTSE', 'TSX'),
    ('^RUT',    'Russell 2000'),
    ('^VIX',    'VIX'),
    ('XIC.TO',  'iShares TSX Composite'),
    ('^FTSE',   'FTSE 100'),
    ('^N225',   'Nikkei 225'),
    ('^TNX',    'US 10Y Yield'),
]

# Map US tickers to TSX equivalents for Canadian stocks (to get CAD prices)
CANADIAN_STOCKS_MAP = {
    'TD': 'TD.TO',
    'ACDVF': 'AC.TO',  # Air Canada
    'ENB': 'ENB.TO',
    'RCI': 'RCI-B.TO',  # Rogers Class B
    'CVE': 'CVE.TO',
    'CNSWF': 'CSU.TO'  # Constellation Software
}


def get_ticker_for_charts(symbol):
    """Get the appropriate ticker symbol for charts (TSX for Canadian stocks)"""
    return CANADIAN_STOCKS_MAP.get(symbol, symbol)


def is_canadian_stock(symbol):
    """Check if a stock is Canadian"""
    return symbol in CANADIAN_STOCKS_MAP


# Valid yfinance chart parameters — reject anything else with a 400 instead of
# passing garbage through to the data layer (uncaught 500s otherwise).
VALID_PERIODS   = {'1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'}
VALID_INTERVALS = {'1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'}


def _int_arg(name, default, max_val=None):
    """Parse an int query param safely: bad input falls back to the default
    instead of raising an uncaught ValueError (→ 500)."""
    try:
        val = int(request.args.get(name, default))
    except (TypeError, ValueError):
        val = default
    if max_val is not None:
        val = min(val, max_val)
    return max(val, 1)


def _validate_chart_args(period, interval):
    """Return an error response tuple for invalid chart params, or None if OK."""
    if period not in VALID_PERIODS:
        return jsonify({'success': False,
                        'error': f'invalid period {period!r}',
                        'valid_periods': sorted(VALID_PERIODS)}), 400
    if interval not in VALID_INTERVALS:
        return jsonify({'success': False,
                        'error': f'invalid interval {interval!r}',
                        'valid_intervals': sorted(VALID_INTERVALS)}), 400
    return None
