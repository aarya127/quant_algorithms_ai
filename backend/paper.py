"""
paper.py — paper-trading fill rules, shared by the web routes (routes/paper.py)
and the ML model's account (scripts/paper_model_account.py).

Prices are in the symbol's listing currency; quotes carry `fx_usd` so accounts
(kept in USD) can convert, e.g. TSX listings in CAD.

An order fills only from bars that START at or after the moment it was placed, so
its price never comes from data the trader could already see:
  market — the first such bar's open, with SLIPPAGE against the trader
  limit  — the first such bar that trades through the limit, at the limit or at
           that bar's open if it gapped through (buy: min, sell: max)
No commission (like most retail brokers today).
"""
import datetime
import threading

import pandas as pd
from cachetools import TTLCache

SLIPPAGE = 0.0005           # 5 bps on market orders
INTRADAY_DAYS = 59          # yfinance serves 5-minute bars for the last 60 days

_cache = TTLCache(maxsize=256, ttl=60)
_cache_lock = threading.Lock()


def fill_order(order: dict, bars: pd.DataFrame):
    """(price, bar_time) of the order's fill from `bars`, or None if not filled yet.

    order: side ('buy'|'sell'), type ('market'|'limit'), limit (limit orders),
           placed_at (tz-aware Timestamp)
    bars:  Open/High/Low columns, index = tz-aware bar start times, ascending
    """
    after = bars[bars.index >= order["placed_at"]]
    if after.empty:
        return None
    buy = order["side"] == "buy"
    if order["type"] == "market":
        px = float(after["Open"].iloc[0]) * (1 + SLIPPAGE if buy else 1 - SLIPPAGE)
        return px, after.index[0]
    lim = float(order["limit"])
    hit = after[after["Low"] <= lim] if buy else after[after["High"] >= lim]
    if hit.empty:
        return None
    opn = float(hit["Open"].iloc[0])
    return (min(lim, opn) if buy else max(lim, opn)), hit.index[0]


def _cached(key, fetch):
    with _cache_lock:
        if key in _cache:
            return _cache[key]
    val = fetch()
    with _cache_lock:
        _cache[key] = val
    return val


def bars_since(symbol: str, since: pd.Timestamp) -> pd.DataFrame:
    """OHLC bars covering `since` → now: 5-minute bars when yfinance still has
    them, daily bars for older orders."""
    import yfinance as yf
    now = pd.Timestamp.now(tz="UTC")
    if since >= now - pd.Timedelta(days=INTRADAY_DAYS):
        return _cached((symbol, "5m"), lambda: yf.Ticker(symbol).history(
            period=f"{INTRADAY_DAYS}d", interval="5m", auto_adjust=False))
    start = (since - pd.Timedelta(days=1)).date().isoformat()
    return _cached((symbol, "1d", start), lambda: yf.Ticker(symbol).history(
        start=start, interval="1d", auto_adjust=False))


def fx_usd(currency: str) -> float:
    """USD value of one unit of `currency` (yfinance FX; raises if unknown)."""
    if currency == "USD":
        return 1.0
    import yfinance as yf
    return _cached((currency, "fx"), lambda: float(
        yf.Ticker(f"{currency}USD=X").fast_info["last_price"]))


def quote(symbol: str) -> dict:
    """Latest price, listing currency and its USD rate (yfinance; may be ~15 min delayed)."""
    import yfinance as yf

    def fetch():
        fi = yf.Ticker(symbol).fast_info
        return {"price": float(fi["last_price"]),
                "prev_close": float(fi["previous_close"]),
                "currency": fi["currency"],
                "fx_usd": fx_usd(fi["currency"]),
                "as_of": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    return _cached((symbol, "quote"), fetch)
