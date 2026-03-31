"""
extractors/alpaca.py — Alpaca extractor.

Primary roles  : OHLCV (equity bars via REST), CALENDAR (market hours)
Supplement role: NEWS  (REST news feed — Finnhub is primary for news)

Required env vars
-----------------
ALPACA_API_KEY
ALPACA_SECRET_KEY

Note: The existing data/alpaca_news.py handles WebSocket streaming news.
      This extractor uses Alpaca's REST endpoints for historical bars
      and news, so both can coexist without conflict.
"""

import logging
import os
from typing import List, Optional

import requests

from ..base import DataRole, ExtractorBase, FundamentalsRecord, NewsRecord, OHLCVRecord, ProfileRecord

logger = logging.getLogger(__name__)

_DATA_BASE    = "https://data.alpaca.markets"
_BROKER_BASE  = "https://api.alpaca.markets"


def _auth_headers() -> dict:
    return {
        "APCA-API-KEY-ID":     os.environ.get("ALPACA_API_KEY", ""),
        "APCA-API-SECRET-KEY": os.environ.get("ALPACA_SECRET_KEY", ""),
    }


class AlpacaExtractor(ExtractorBase):
    """
    Alpaca REST extractor for equity OHLCV bars and market-hours calendar.

    OHLCV notes
    -----------
    Uses the /v2/stocks/{symbol}/bars endpoint with IEX feed (free tier).
    Upgrade feed="sip" for consolidated SIP tape (requires Alpaca subscription).
    Bars are fully adjusted (split + dividend) via adjustment="all".

    Calendar notes
    --------------
    Uses /v2/calendar to retrieve open/close times per trading day.
    Call fetch_calendar() directly; it is not routed through DataRole.OHLCV.
    """

    SOURCE_NAME     = "alpaca"
    SUPPORTED_ROLES = [DataRole.OHLCV, DataRole.CALENDAR, DataRole.NEWS]

    _TF_MAP = {
        "1d":    "1Day",
        "1h":    "1Hour",
        "30min": "30Min",
        "15min": "15Min",
        "5min":  "5Min",
        "1min":  "1Min",
    }

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
        Fetch historical equity bars.

        Parameters
        ----------
        start, end : ISO-8601 date strings, e.g. "2024-01-01"
        interval   : one of 1d | 1h | 30min | 15min | 5min | 1min
        """
        timeframe = self._TF_MAP.get(interval, "1Day")
        url    = f"{_DATA_BASE}/v2/stocks/{symbol.upper()}/bars"
        params = {
            "start":      start,
            "end":        end,
            "timeframe":  timeframe,
            "adjustment": "all",    # split + dividend adjusted
            "feed":       "iex",    # IEX = free; "sip" for premium consolidated tape
            "limit":      10000,
        }

        records: List[OHLCVRecord] = []
        try:
            resp = requests.get(url, headers=_auth_headers(), params=params, timeout=15)
            resp.raise_for_status()
            for bar in resp.json().get("bars") or []:
                records.append(OHLCVRecord(
                    date=bar["t"][:10],         # ISO timestamp → date part
                    open=bar["o"],
                    high=bar["h"],
                    low=bar["l"],
                    close=bar["c"],
                    volume=int(bar["v"]),
                    adjusted_close=bar.get("vw"),  # VWAP as adj-close proxy
                ))
        except Exception as exc:
            logger.warning("AlpacaExtractor.fetch_ohlcv failed for %s: %s", symbol, exc)
        return records

    # -------------------------------------------------------------------
    # CALENDAR  (direct call — not dispatched via DataRole)
    # -------------------------------------------------------------------

    def fetch_calendar(self, start: str, end: str) -> List[dict]:
        """
        Return market-hours calendar between start and end.
        Each entry: {"date": "YYYY-MM-DD", "open": "HH:MM", "close": "HH:MM"}.
        """
        url    = f"{_BROKER_BASE}/v2/calendar"
        params = {"start": start, "end": end}
        try:
            resp = requests.get(url, headers=_auth_headers(), params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.warning("AlpacaExtractor.fetch_calendar failed: %s", exc)
            return []

    # -------------------------------------------------------------------
    # NEWS  (supplement — Finnhub is primary)
    # -------------------------------------------------------------------

    def fetch_news(
        self,
        symbol: str,
        start:  str,
        end:    str,
        limit:  int = 50,
    ) -> List[NewsRecord]:
        """
        Pull Alpaca-aggregated news via REST (v1beta1 endpoint).
        Multiple publishers with optional full-text content.
        """
        url    = f"{_DATA_BASE}/v1beta1/news"
        params = {
            "symbols":          symbol.upper(),
            "start":            start,
            "end":              end,
            "limit":            min(limit, 50),
            "include_content":  "false",
            "sort":             "desc",
        }
        records: List[NewsRecord] = []
        try:
            resp = requests.get(url, headers=_auth_headers(), params=params, timeout=10)
            resp.raise_for_status()
            for a in resp.json().get("news") or []:
                records.append(NewsRecord(
                    headline=     a.get("headline", ""),
                    summary=      a.get("summary",  ""),
                    source=       a.get("source",   "alpaca"),
                    url=          a.get("url",       ""),
                    published_at= a.get("created_at", ""),
                    symbols=      a.get("symbols",  [symbol]),
                ))
        except Exception as exc:
            logger.warning("AlpacaExtractor.fetch_news failed for %s: %s", symbol, exc)
        return records

    # -------------------------------------------------------------------
    # Unsupported roles — explicit refusals keep fallback chain clean
    # -------------------------------------------------------------------

    def fetch_profile(self, symbol: str) -> ProfileRecord:
        raise NotImplementedError("AlpacaExtractor: use FinnhubExtractor for PROFILE")

    def fetch_fundamentals(self, symbol: str) -> FundamentalsRecord:
        raise NotImplementedError("AlpacaExtractor: use FiscalAIExtractor for FUNDAMENTALS")
