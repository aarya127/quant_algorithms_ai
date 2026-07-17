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

Quotes: 30s | Profiles: 1h | Financials/news: 10min | yf info: 5min
"""
import os
import sys
import threading
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
get_company_profile = get_stock_quote = _unavailable
get_company_overview_llm = _unavailable
get_market_tweets = get_financial_news_feed = _unavailable
get_recent_news = _unavailable
get_company_statistics = format_statistics_for_display = _unavailable
StockAnalyzer = None


def start_news_stream(*_a, **_kw):  # overwritten below on successful import
    pass


def stop_news_stream():  # overwritten below on successful import
    pass


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
        get_stock_quote
    )
    from data.alphavantage import AlphaVantage
    from ai_platform.nvidia_llm import get_company_overview_llm
    from data.twitter_feed import get_market_tweets, get_financial_news_feed
    from data.alpaca_news import get_recent_news, start_news_stream, stop_news_stream
    from data.company_statistics import get_company_statistics, format_statistics_for_display
    from stock_analyzer import StockAnalyzer
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


def _ttl_cached(cache):
    """Cache decorator: one lock-guarded lookup/store around the wrapped getter."""
    def deco(fn):
        @wraps(fn)
        def wrapper(*args):
            key = args if len(args) > 1 else args[0]
            with _cache_lock:
                if key in cache:
                    return cache[key]
            result = fn(*args)
            with _cache_lock:
                cache[key] = result
            return result
        return wrapper
    return deco


@_ttl_cached(TTLCache(maxsize=256, ttl=30))
def _cached_get_stock_quote(symbol):
    return get_stock_quote(symbol)


@_ttl_cached(TTLCache(maxsize=256, ttl=3600))
def _cached_get_company_profile(symbol):
    return get_company_profile(symbol)


@_ttl_cached(TTLCache(maxsize=256, ttl=600))
def _cached_get_basic_financials(symbol):
    return get_basic_financials(symbol)


@_ttl_cached(TTLCache(maxsize=256, ttl=600))
def _cached_get_company_news(symbol, from_date, to_date):
    return get_company_news(symbol, from_date, to_date)


@_ttl_cached(TTLCache(maxsize=256, ttl=300))
def _cached_yf_info(ticker_symbol):
    return yf.Ticker(ticker_symbol).info
