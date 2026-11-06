"""
services.py — shared data-access layer for app.py and the route blueprints.

Owns two things that used to live at the top of the app.py monolith:

1. The guarded data-module imports. A single bad import must never crash the
   server: if anything fails the error is printed to the deploy logs, the server
   still starts, /health passes, and the affected endpoints return errors.
   Import from here (`from services import get_stock_quote, yf, ...`) instead of
   importing `data.*` directly in route modules.

2. The TTL caches over the hot Finnhub/yfinance getters, deduplicated into one
   decorator (the monolith repeated the same lock/lookup/store pattern five times).

Quotes: 30s | Profiles: 1h | Peers: 24h | Financials/news: 10min | yf info: 5min
SEC cash-flow summary: 24h
"""
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

from cachetools import TTLCache

# project root (for the `data`/`ai_platform` packages)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

_DATA_MODULES_LOADED = False
av = None
yf = None

# Safe no-op fallbacks so `from services import X` never raises even when the
# data layer failed to import; callers get a clear error at call time instead.
def _unavailable(*_a, **_kw):
    raise RuntimeError("data modules failed to import at startup — see deploy logs")

get_company_news = get_basic_financials = get_earnings_surprises = _unavailable
get_insider_transactions = get_insider_sentiment = get_earnings_calendar = _unavailable
get_company_profile = get_company_peers = get_stock_quote = _unavailable
get_company_overview_llm = _unavailable
get_market_tweets = get_financial_news_feed = _unavailable
get_recent_news = _unavailable
get_company_statistics = format_statistics_for_display = has_any_statistic = _unavailable
get_cash_flow_summary = _unavailable


try:
    import yfinance as yf
    from data.finnhub import (
        get_company_news,
        get_basic_financials,
        get_earnings_surprises,
        get_insider_transactions,
        get_insider_sentiment,
        get_earnings_calendar,
        get_company_profile,
        get_company_peers,
        get_stock_quote
    )
    from data.alphavantage import AlphaVantage
    from ai_platform.nvidia_llm import get_company_overview_llm
    from data.twitter_feed import get_market_tweets, get_financial_news_feed
    from data.alpaca_news import get_recent_news
    from data.company_statistics import get_company_statistics, format_statistics_for_display, has_any_statistic
    from data.financial_statements import get_cash_flow_summary
    av = AlphaVantage()
    _DATA_MODULES_LOADED = True
    print("[STARTUP] All data modules loaded successfully.", flush=True)
except Exception:
    import traceback as _tb
    print(f"[STARTUP ERROR] One or more data modules failed to import:\n{_tb.format_exc()}", flush=True)


# ---------------------------------------------------------------------------
# TTL caches — keyed by args, thread-safe for concurrent requests.
# ---------------------------------------------------------------------------
_cache_lock = threading.Lock()


def _ttl_cached(cache, valid=bool):
    """Cache decorator: lock-guarded lookup/store around the wrapped getter.

    Concurrent misses for one key share a single upstream call (the page fires
    several endpoints that need the same data at once), and results failing
    `valid` (None, {}, a throttled stub) are returned but never cached, so a
    bad response isn't pinned for the whole TTL.
    """
    def deco(fn):
        key_locks = {}

        @wraps(fn)
        def wrapper(*args):
            key = args if len(args) > 1 else args[0]
            with _cache_lock:
                if key in cache:
                    return cache[key]
                lock = key_locks.setdefault(key, threading.Lock())
            with lock:
                with _cache_lock:
                    if key in cache:
                        return cache[key]
                result = fn(*args)
                with _cache_lock:
                    if valid(result):
                        cache[key] = result
                    key_locks.pop(key, None)
            return result
        return wrapper
    return deco


@_ttl_cached(TTLCache(maxsize=256, ttl=30))
def _cached_get_stock_quote(symbol):
    return get_stock_quote(symbol)


@_ttl_cached(TTLCache(maxsize=256, ttl=3600))
def _cached_get_company_profile(symbol):
    return get_company_profile(symbol)


@_ttl_cached(TTLCache(maxsize=256, ttl=86400))
def _cached_get_company_peers(symbol):
    return get_company_peers(symbol)


@_ttl_cached(TTLCache(maxsize=256, ttl=600))
def _cached_get_basic_financials(symbol):
    return get_basic_financials(symbol)


@_ttl_cached(TTLCache(maxsize=256, ttl=600))
def _cached_get_company_news(symbol, from_date, to_date):
    return get_company_news(symbol, from_date, to_date)


# Filings change quarterly; only the parsed summary is kept (the raw as-reported
# payload is ~1 MB). {} (no SEC filings, e.g. a TSX-only listing) is cached too,
# so it isn't refetched on every view; a raised error (429) is not.
@_ttl_cached(TTLCache(maxsize=256, ttl=86400), valid=lambda r: r is not None)
def _cached_cash_flow_summary(symbol):
    return get_cash_flow_summary(symbol)


def _is_real_yf_info(info):
    """A throttled .info call can come back as a stub like {'trailingPegRatio': None}."""
    return bool(info) and any(info.get(k) is not None
                              for k in ('quoteType', 'marketCap', 'regularMarketPrice', 'longName'))


@_ttl_cached(TTLCache(maxsize=256, ttl=300), valid=_is_real_yf_info)
def _cached_yf_info(ticker_symbol):
    return yf.Ticker(ticker_symbol).info


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception:
        return None


MAX_PEERS = 8


def graded_fundamentals(symbol):
    """(grades, industry, peers): the company's Finnhub fundamentals graded against
    up to MAX_PEERS industry peers (grading.py). Shared by /api/metrics and the
    Recommendations quality vote. A peer whose fetch fails (e.g. a 429) drops out."""
    from grading import grade_fundamentals
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_fins = ex.submit(_cached_get_basic_financials, symbol)
        f_profile = ex.submit(_safe, _cached_get_company_profile, symbol)
        f_peers = ex.submit(_safe, _cached_get_company_peers, symbol)
        metric = (f_fins.result() or {}).get('metric') or {}
        industry = (f_profile.result() or {}).get('finnhubIndustry')
        peers = [p for p in (f_peers.result() or []) if p != symbol][:MAX_PEERS]
    with ThreadPoolExecutor(max_workers=MAX_PEERS) as ex:
        fins = ex.map(lambda p: _safe(_cached_get_basic_financials, p), peers)
        fetched = {p: f['metric'] for p, f in zip(peers, fins) if f and f.get('metric')}
    # another share class of the same company (GOOG for GOOGL) isn't a peer
    peer_metrics = {p: m for p, m in fetched.items() if not same_company(metric, m)}
    grades = grade_fundamentals(metric, list(peer_metrics.values()), industry)
    # peers whose fetch failed (often a Finnhub 429) — grades can shift on a retry
    grades['peers_unavailable'] = [p for p in peers if p not in fetched]
    return grades, industry, list(peer_metrics)


_SAME_COMPANY_KEYS = ('revenueGrowthTTMYoy', 'grossMarginTTM', 'netProfitMarginTTM',
                      'operatingMarginTTM', 'roeTTM')


def same_company(a, b):
    """True when two Finnhub metric dicts agree on every company-level fundamental
    both report (at least three) — what two share classes of one company look like."""
    shared = [k for k in _SAME_COMPANY_KEYS if a.get(k) is not None and b.get(k) is not None]
    return len(shared) >= 3 and all(a[k] == b[k] for k in shared)
