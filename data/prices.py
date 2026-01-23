from typing import List, Optional
import logging
import sys
import pandas as pd
import yfinance as yf

#!/usr/bin/env python3
"""
prices.py

Simple utilities to fetch current and historical stock prices using yfinance.

Install dependency:
    pip install yfinance pandas
"""



# (imports at top handle ImportError automatically) Removed a broken try/except that caused a syntax error.

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_historical_prices(
    ticker: str,
    period: str = "1mo",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Return historical OHLCV data for `ticker`.
    period examples: '1d', '5d', '1mo', '3mo', '1y', '5y', 'max'
    interval examples: '1m', '5m', '1h', '1d', '1wk', '1mo'
    """
    logger.debug("Downloading historical for %s period=%s interval=%s", ticker, period, interval)
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        logger.warning("No historical data returned for %s", ticker)
    return df


def get_latest_price(ticker: str) -> Optional[float]:
    """
    Try to get the most recent known price for `ticker`.
    Uses history fallback for reliability.
    Returns None if no price is available.
    """
    t = yf.Ticker(ticker)
    # Try fast info first (may not always be present)
    try:
        fast = getattr(t, "fast_info", None) or {}
        last = fast.get("last_price") or fast.get("last")
        if last:
            return float(last)
    except Exception:
        # ignore and fallback to history
        pass

    # Fallback: take the last Close from recent history
    df = t.history(period="5d", interval="1d")
    if not df.empty:
        # take last non-null Close
        closes = df["Close"].dropna()
        if not closes.empty:
            return float(closes.iloc[-1])
    logger.warning("Could not determine latest price for %s", ticker)
    return None


def get_multiple_latest(tickers: List[str]) -> pd.Series:
    """
    Download 'Close' for multiple tickers for the last traded day.
    Returns a pandas Series indexed by ticker.
    """
    if not tickers:
        return pd.Series(dtype=float)

    logger.debug("Downloading latest prices for tickers: %s", tickers)
    # yf.download returns a DataFrame with columns as tickers when multiple are provided
    df = yf.download(tickers, period="2d", interval="1d", progress=False)["Close"]
    if isinstance(df, pd.Series):
        # only one ticker was passed; make it a Series indexed by ticker
        return pd.Series({tickers[0]: float(df.dropna().iloc[-1])}) if not df.dropna().empty else pd.Series({tickers[0]: None})
    latest = {}
    for t in tickers:
        try:
            col = df[t].dropna()
            latest[t] = float(col.iloc[-1]) if not col.empty else None
        except Exception:
            latest[t] = None
    return pd.Series(latest)


def print_summary(tickers: List[str]) -> None:
    """
    Print a short summary: latest price and last 5 rows of history for each ticker.
    """
    prices = get_multiple_latest(tickers)
    print("Latest prices:")
    print(prices.to_string())

    for t in tickers:
        print("\nHistory for", t)
        hist = get_historical_prices(t, period="7d", interval="1d")
        if hist.empty:
            print("  No historical data available.")
        else:
            # show last 5 rows
            print(hist.tail(5).to_string())


if __name__ == "__main__":
    # Example usage:
    # python prices.py AAPL MSFT TSLA
    if len(sys.argv) > 1:
        tickers = sys.argv[1:]
    else:
        # default examples
        tickers = ["AAPL", "MSFT", "TSLA"]

    print_summary(tickers)