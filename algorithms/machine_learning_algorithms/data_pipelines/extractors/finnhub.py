"""
extractors/finnhub.py — Finnhub extractor.

Primary roles  : PROFILE, NEWS, SENTIMENT
Secondary roles: FUNDAMENTALS, CALENDAR (earnings dates)

Wraps the existing data/finnhub.py module — no API client duplication.

Rate limit: 60 calls / minute on the free tier.
"""

import logging
from typing import List

from ..base import DataRole, ExtractorBase, FundamentalsRecord, NewsRecord, OHLCVRecord, ProfileRecord

# Thin pass-through to the existing project integration.
try:
    from data.finnhub import (
        get_company_profile,
        get_company_news,
        get_basic_financials,
        get_insider_sentiment,
        get_earnings_surprises,
        get_earnings_calendar,
    )
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

logger = logging.getLogger(__name__)


def _require() -> None:
    if not _AVAILABLE:
        raise RuntimeError(
            "finnhub-python is not installed or data/finnhub.py is not on sys.path. "
            "Run: pip install finnhub-python"
        )


class FinnhubExtractor(ExtractorBase):
    """
    Finnhub extractor for company metadata, news, and sentiment signals.

    PROFILE    — richest free-tier company metadata (sector, exchange, logo, country)
    NEWS       — company news with full article text and timestamps
    SENTIMENT  — insider MSPR score and net change in insider transactions
    FUNDAMENTALS — basic financial metrics (PE, PB, margins, FCF) as fallback
    CALENDAR   — earnings dates via get_earnings_calendar as fallback
    """

    SOURCE_NAME     = "finnhub"
    SUPPORTED_ROLES = [
        DataRole.PROFILE,
        DataRole.NEWS,
        DataRole.SENTIMENT,
        DataRole.FUNDAMENTALS,
        DataRole.CALENDAR,
    ]

    # -------------------------------------------------------------------
    # PROFILE
    # -------------------------------------------------------------------

    def fetch_profile(self, symbol: str) -> ProfileRecord:
        _require()
        try:
            p = get_company_profile(symbol)
            if not p:
                raise ValueError(f"Finnhub returned empty profile for {symbol}")
            return ProfileRecord(
                symbol=      symbol,
                name=        p.get("name", ""),
                exchange=    p.get("exchange", ""),
                sector=      p.get("finnhubIndustry", ""),
                industry=    p.get("finnhubIndustry", ""),
                market_cap=  p.get("marketCapitalization"),   # USD millions
                employees=   p.get("employeeTotal"),
                description= None,                           # not in Finnhub profile
                logo_url=    p.get("logo"),
                website=     p.get("weburl"),
                country=     p.get("country"),
            )
        except Exception as exc:
            logger.warning("FinnhubExtractor.fetch_profile failed for %s: %s", symbol, exc)
            raise

    # -------------------------------------------------------------------
    # NEWS
    # -------------------------------------------------------------------

    def fetch_news(
        self,
        symbol: str,
        start:  str,
        end:    str,
        limit:  int = 50,
    ) -> List[NewsRecord]:
        _require()
        try:
            articles = get_company_news(symbol, start, end) or []
            records: List[NewsRecord] = []
            for a in articles[:limit]:
                records.append(NewsRecord(
                    headline=     a.get("headline", ""),
                    summary=      a.get("summary",  ""),
                    source=       a.get("source",   "finnhub"),
                    url=          a.get("url",       ""),
                    published_at= str(a.get("datetime", "")),
                    symbols=      [symbol],
                ))
            return records
        except Exception as exc:
            logger.warning("FinnhubExtractor.fetch_news failed for %s: %s", symbol, exc)
            return []

    # -------------------------------------------------------------------
    # FUNDAMENTALS  (secondary — Fiscal.ai is primary)
    # -------------------------------------------------------------------

    def fetch_fundamentals(self, symbol: str) -> FundamentalsRecord:
        _require()
        try:
            raw = get_basic_financials(symbol, metric="all")
            m   = raw.get("metric", {})
            return FundamentalsRecord(
                symbol=         symbol,
                period=         "ttm",
                pe_ratio=       m.get("peBasicExclExtraTTM"),
                pb_ratio=       m.get("pbQuarterly"),
                ev_ebitda=      m.get("currentEv/freeCashFlowTTM"),
                gross_margin=   m.get("grossMarginTTM"),
                free_cash_flow= m.get("freeCashFlowTTM"),
                raw=            raw,
            )
        except Exception as exc:
            logger.warning("FinnhubExtractor.fetch_fundamentals failed for %s: %s", symbol, exc)
            raise

    # -------------------------------------------------------------------
    # SENTIMENT  (direct call — not dispatched via generic DataRole routing)
    # -------------------------------------------------------------------

    def fetch_sentiment(self, symbol: str, start: str, end: str) -> dict:
        """
        Return Finnhub insider sentiment signals.
        Keys: change (net insider buying), mspr (Monthly Share Purchase Ratio).
        """
        _require()
        try:
            return get_insider_sentiment(symbol, start, end) or {}
        except Exception as exc:
            logger.warning("FinnhubExtractor.fetch_sentiment failed for %s: %s", symbol, exc)
            return {}

    # -------------------------------------------------------------------
    # CALENDAR (earnings dates)
    # -------------------------------------------------------------------

    def fetch_earnings_calendar(self, start: str, end: str, symbol: str = "") -> dict:
        _require()
        try:
            return get_earnings_calendar(start, end, symbol or None) or {}
        except Exception as exc:
            logger.warning("FinnhubExtractor.fetch_earnings_calendar failed: %s", exc)
            return {}

    # -------------------------------------------------------------------
    # Unsupported role
    # -------------------------------------------------------------------

    def fetch_ohlcv(self, symbol: str, start: str, end: str, interval: str = "1d"):
        raise NotImplementedError("FinnhubExtractor: use AlpacaExtractor for OHLCV")
