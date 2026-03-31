"""
extractors/fiscal_ai.py — Fiscal.ai extractor.

Primary roles: FUNDAMENTALS, FILINGS

Primary for:
- Structured financial statement data (income, balance sheet, cash flow)
- Accounting-derived signals (accruals ratio, quality score, Altman Z)
- Earnings enrichment (actual vs. estimate, surprise %, revision trend)
- SEC filings parsed into structured JSON

Required env var: FISCAL_AI_API_KEY

TODO
----
API endpoints marked TODO — update once Fiscal.ai credentials are
provisioned and endpoint schemas confirmed.  The response-field mapping
in fetch_fundamentals() and fetch_filings() must be adapted to match
the actual Fiscal.ai JSON structure.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import requests

from ..base import DataRole, ExtractorBase, FundamentalsRecord, NewsRecord, OHLCVRecord, ProfileRecord

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.fiscal.ai/v1"    # TODO: confirm once account is provisioned


class FiscalAIExtractor(ExtractorBase):
    """
    Fiscal.ai extractor for structured fundamentals and SEC filings.

    FUNDAMENTALS (primary)
    ----------------------
    Provides accounting-derived signals beyond what yfinance or Finnhub
    expose on free tiers:
      - Accruals ratio (earnings quality signal)
      - Altman Z-score (distress signal)
      - Sloan ratio
      - Multi-period growth rates derived from GAAP filings

    FILINGS (primary)
    -----------------
    Parsed 10-K, 10-Q, 8-K filings as structured JSON.
    Useful for factor discovery and NLP pipelines.

    CALENDAR (supplementary)
    ------------------------
    Earnings date + EPS estimate + actual + surprise %.
    """

    SOURCE_NAME     = "fiscal_ai"
    SUPPORTED_ROLES = [DataRole.FUNDAMENTALS, DataRole.FILINGS, DataRole.CALENDAR]

    def __init__(self):
        self._api_key = os.environ.get("FISCAL_AI_API_KEY", "")
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type":  "application/json",
        }

    def _require_key(self) -> None:
        if not self._api_key:
            raise RuntimeError(
                "FISCAL_AI_API_KEY environment variable is not set. "
                "Obtain a key from https://fiscal.ai and export it."
            )

    def _get(self, path: str, params: Optional[Dict] = None) -> Any:
        self._require_key()
        resp = requests.get(
            f"{_BASE_URL}{path}",
            headers=self._headers,
            params=params or {},
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()

    # -------------------------------------------------------------------
    # FUNDAMENTALS  (primary)
    # -------------------------------------------------------------------

    def fetch_fundamentals(self, symbol: str) -> FundamentalsRecord:
        """
        Fetch structured fundamentals from Fiscal.ai.

        TODO: Replace endpoint path and field mapping once API schema is confirmed.
        Expected fields in JSON response listed in the mapping below.
        """
        try:
            data = self._get(f"/fundamentals/{symbol}")   # TODO: confirm path
            # TODO: adapt field names to match Fiscal.ai's actual response schema
            return FundamentalsRecord(
                symbol=         symbol,
                period=         data.get("period", "ttm"),
                revenue=        data.get("revenue"),
                net_income=     data.get("net_income"),
                eps=            data.get("eps"),
                pe_ratio=       data.get("pe_ratio"),
                pb_ratio=       data.get("pb_ratio"),
                ev_ebitda=      data.get("ev_ebitda"),
                gross_margin=   data.get("gross_margin"),
                debt_to_equity= data.get("debt_to_equity"),
                free_cash_flow= data.get("free_cash_flow"),
                raw=            data,
            )
        except Exception as exc:
            logger.warning("FiscalAIExtractor.fetch_fundamentals failed for %s: %s", symbol, exc)
            raise

    # -------------------------------------------------------------------
    # FILINGS  (primary — direct call, not in generic DataRole dispatch)
    # -------------------------------------------------------------------

    def fetch_filings(
        self,
        symbol:    str,
        form_type: str = "10-K",
        limit:     int = 5,
    ) -> List[Dict]:
        """
        Retrieve parsed SEC filings from Fiscal.ai.

        Parameters
        ----------
        symbol    : equity ticker
        form_type : '10-K' | '10-Q' | '8-K' | 'DEF 14A' | ...
        limit     : maximum number of filings to return

        Returns list of filing dicts.  Schema depends on Fiscal.ai response.
        TODO: confirm endpoint path and response structure once API key is available.
        """
        try:
            data = self._get(
                f"/filings/{symbol}",   # TODO: confirm path
                params={"form_type": form_type, "limit": limit},
            )
            return data.get("filings", [])
        except Exception as exc:
            logger.warning("FiscalAIExtractor.fetch_filings failed for %s: %s", symbol, exc)
            return []

    # -------------------------------------------------------------------
    # CALENDAR  (earnings dates — supplementary)
    # -------------------------------------------------------------------

    def fetch_earnings_calendar(self, symbol: str, limit: int = 8) -> List[Dict]:
        """
        Return earnings events: date, EPS estimate, EPS actual, surprise %.
        TODO: confirm endpoint path once API key is provisioned.
        """
        try:
            data = self._get(
                f"/earnings/{symbol}",   # TODO: confirm path
                params={"limit": limit},
            )
            return data.get("earnings", [])
        except Exception as exc:
            logger.warning("FiscalAIExtractor.fetch_earnings_calendar failed for %s: %s", symbol, exc)
            return []

    # -------------------------------------------------------------------
    # Unsupported
    # -------------------------------------------------------------------

    def fetch_ohlcv(self, symbol: str, start: str, end: str, interval: str = "1d"):
        raise NotImplementedError("FiscalAIExtractor: use AlpacaExtractor for OHLCV")

    def fetch_profile(self, symbol: str) -> ProfileRecord:
        raise NotImplementedError("FiscalAIExtractor: use FinnhubExtractor for PROFILE")

    def fetch_news(self, symbol: str, start: str, end: str, limit: int = 50) -> List[NewsRecord]:
        raise NotImplementedError("FiscalAIExtractor: use FinnhubExtractor for NEWS")
