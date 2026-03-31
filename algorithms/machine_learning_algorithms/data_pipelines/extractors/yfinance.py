"""
extractors/yfinance.py — yfinance extractor.

Primary role  : TECHNICALS  (unlimited, built-in indicator calculations)
Fallback roles: OHLCV, PROFILE, FUNDAMENTALS, OPTIONS (basic surface)

Wraps data/prices.py, data/charts.py, and data/company_statistics.py
using the project's existing yfinance integrations.

Rate limit: None (unofficial scraper — do not hammer in tight loops).
"""

import logging
from typing import List, Optional

from ..base import DataRole, ExtractorBase, FundamentalsRecord, NewsRecord, OHLCVRecord, ProfileRecord

try:
    import yfinance as yf
    from data.prices import get_historical_prices
    from data.charts import get_technical_indicators
    from data.company_statistics import get_company_statistics
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

logger = logging.getLogger(__name__)


def _require() -> None:
    if not _AVAILABLE:
        raise RuntimeError(
            "yfinance is not installed or data/prices.py is not on sys.path. "
            "Run: pip install yfinance"
        )


class YFinanceExtractor(ExtractorBase):
    """
    yfinance extractor — backbone fallback for prices, profile, and fundamentals.

    OHLCV      : auto-adjusted daily / intraday bars via yfinance Ticker.history()
    TECHNICALS : SMA20/50/200, EMA12/26, RSI14, MACD, Bollinger via data/charts.py
    PROFILE    : company info (sector, industry, description, employees, website)
    FUNDAMENTALS: composite stats from data/company_statistics.py
                  (yfinance primary, Finnhub + Alpha Vantage as its own fallbacks)
    OPTIONS    : basic option surface snapshot (expiry, strike, IV, OI)
    """

    SOURCE_NAME     = "yfinance"
    SUPPORTED_ROLES = [
        DataRole.OHLCV,
        DataRole.TECHNICALS,
        DataRole.PROFILE,
        DataRole.FUNDAMENTALS,
        DataRole.OPTIONS,
    ]

    # -------------------------------------------------------------------
    # OHLCV
    # -------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol:   str,
        start:    str,
        end:      str,
        interval: str = "1d",
    ) -> List[OHLCVRecord]:
        """
        Fetch auto-adjusted OHLCV bars for the given date range.

        Parameters
        ----------
        interval : '1d' | '1h' | '30m' | '15m' | '5m' | '1m'
                   (intraday goes back max 60 days for free accounts)
        """
        _require()
        try:
            ticker = yf.Ticker(symbol)
            df     = ticker.history(
                start=start, end=end, interval=interval, auto_adjust=True
            )
            if df.empty:
                return []
            records: List[OHLCVRecord] = []
            for dt, row in df.iterrows():
                records.append(OHLCVRecord(
                    date=           str(dt.date()),
                    open=           round(float(row["Open"]),   4),
                    high=           round(float(row["High"]),   4),
                    low=            round(float(row["Low"]),    4),
                    close=          round(float(row["Close"]),  4),
                    volume=         int(row["Volume"]),
                    adjusted_close= round(float(row["Close"]),  4),  # auto_adjust = already adjusted
                ))
            return records
        except Exception as exc:
            logger.warning("YFinanceExtractor.fetch_ohlcv failed for %s: %s", symbol, exc)
            return []

    # -------------------------------------------------------------------
    # TECHNICALS  (primary role)
    # -------------------------------------------------------------------

    def fetch_technicals(
        self,
        symbol:   str,
        period:   str = "1y",
        interval: str = "1d",
    ) -> dict:
        """
        Return standard technical indicators via data/charts.py.

        Returns dict with keys: sma_20, sma_50, sma_200, ema_12, ema_26,
        rsi, macd, macd_signal, macd_hist, bb_upper, bb_middle, bb_lower,
        close, dates, symbol, period, interval.
        """
        _require()
        try:
            return get_technical_indicators(symbol, period, interval)
        except Exception as exc:
            logger.warning("YFinanceExtractor.fetch_technicals failed for %s: %s", symbol, exc)
            raise

    # -------------------------------------------------------------------
    # PROFILE
    # -------------------------------------------------------------------

    def fetch_profile(self, symbol: str) -> ProfileRecord:
        _require()
        try:
            info = yf.Ticker(symbol).info
            return ProfileRecord(
                symbol=      symbol,
                name=        info.get("longName", info.get("shortName", symbol)),
                exchange=    info.get("exchange",   ""),
                sector=      info.get("sector",     ""),
                industry=    info.get("industry",   ""),
                market_cap=  info.get("marketCap"),
                employees=   info.get("fullTimeEmployees"),
                description= info.get("longBusinessSummary"),
                website=     info.get("website"),
                country=     info.get("country"),
            )
        except Exception as exc:
            logger.warning("YFinanceExtractor.fetch_profile failed for %s: %s", symbol, exc)
            raise

    # -------------------------------------------------------------------
    # FUNDAMENTALS
    # -------------------------------------------------------------------

    def fetch_fundamentals(self, symbol: str) -> FundamentalsRecord:
        """
        Uses data/company_statistics.py which already composites yfinance
        with Finnhub and Alpha Vantage fallbacks internally.
        """
        _require()
        try:
            stats = get_company_statistics(symbol)
            val   = stats.get("valuation_ttm",    {})
            mgn   = stats.get("margins",           {})
            fh    = stats.get("financial_health",  {})
            prof  = stats.get("profile",           {})
            return FundamentalsRecord(
                symbol=         symbol,
                period=         "ttm",
                revenue=        prof.get("revenue_ttm"),
                eps=            None,    # not returned by company_statistics
                pe_ratio=       val.get("pe_ratio"),
                pb_ratio=       val.get("pb_ratio"),
                ev_ebitda=      val.get("ev_to_ebitda"),
                gross_margin=   mgn.get("gross"),
                debt_to_equity= fh.get("debt_to_equity"),
                free_cash_flow= mgn.get("fcf"),
                raw=            stats,
            )
        except Exception as exc:
            logger.warning("YFinanceExtractor.fetch_fundamentals failed for %s: %s", symbol, exc)
            raise

    # -------------------------------------------------------------------
    # OPTIONS  (basic surface — Polygon is primary for full chain + greeks)
    # -------------------------------------------------------------------

    def fetch_options_surface(self, symbol: str) -> dict:
        """
        Return a basic options snapshot (all available expiry dates + one
        chain per expiry).  Use PolygonExtractor.fetch_options_chain()
        for full IV surface with greeks.
        """
        _require()
        try:
            ticker   = yf.Ticker(symbol)
            expirations = ticker.options
            surface  = {}
            for exp in expirations[:6]:   # cap at 6 expiries to avoid hammering
                chain = ticker.option_chain(exp)
                surface[exp] = {
                    "calls": chain.calls.to_dict(orient="records"),
                    "puts":  chain.puts.to_dict(orient="records"),
                }
            return surface
        except Exception as exc:
            logger.warning("YFinanceExtractor.fetch_options_surface failed for %s: %s", symbol, exc)
            return {}

    # -------------------------------------------------------------------
    # Unsupported
    # -------------------------------------------------------------------

    def fetch_news(self, symbol: str, start: str, end: str, limit: int = 50) -> List[NewsRecord]:
        raise NotImplementedError("YFinanceExtractor: use FinnhubExtractor for NEWS")
