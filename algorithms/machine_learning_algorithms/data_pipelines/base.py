"""
base.py — Abstract contract for every data-source extractor.

DataRole  : enum of all data types the pipeline can serve
*Record   : lightweight dataclasses for standardised inter-extractor output
ExtractorBase : abstract class every provider extractor must subclass
"""

from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data roles
# ---------------------------------------------------------------------------

class DataRole(Enum):
    OHLCV        = "ohlcv"        # daily / intraday price bars
    PROFILE      = "profile"      # company name, sector, description
    NEWS         = "news"         # news articles and event text
    FUNDAMENTALS = "fundamentals" # financial ratios, income / BS / CF
    TECHNICALS   = "technicals"   # SMA, RSI, MACD, Bollinger, etc.
    OPTIONS      = "options"      # options chain, IV surface, greeks
    FILINGS      = "filings"      # SEC filings, accounting-derived signals
    MACRO        = "macro"        # GDP, Fed rate, CPI, energy prices
    CALENDAR     = "calendar"     # earnings dates, market-hours calendar
    SENTIMENT    = "sentiment"    # aggregated sentiment scores


# ---------------------------------------------------------------------------
# Standardised output records
# ---------------------------------------------------------------------------

@dataclass
class OHLCVRecord:
    date:           str
    open:           float
    high:           float
    low:            float
    close:          float
    volume:         int
    adjusted_close: Optional[float] = None


@dataclass
class ProfileRecord:
    symbol:      str
    name:        str
    exchange:    str = ""
    sector:      str = ""
    industry:    str = ""
    market_cap:  Optional[float] = None
    employees:   Optional[int]   = None
    description: Optional[str]   = None
    logo_url:    Optional[str]   = None
    website:     Optional[str]   = None
    country:     Optional[str]   = None


@dataclass
class NewsRecord:
    headline:        str
    summary:         str
    source:          str
    url:             str
    published_at:    str
    symbols:         List[str]      = field(default_factory=list)
    sentiment_score: Optional[float] = None  # -1.0 (bearish) … +1.0 (bullish)


@dataclass
class FundamentalsRecord:
    symbol:          str
    period:          str            # e.g. "ttm", "2024-Q4", "2024-FY"
    revenue:         Optional[float] = None
    net_income:      Optional[float] = None
    eps:             Optional[float] = None
    pe_ratio:        Optional[float] = None
    pb_ratio:        Optional[float] = None
    ev_ebitda:       Optional[float] = None
    gross_margin:    Optional[float] = None   # 0–1 fraction
    debt_to_equity:  Optional[float] = None
    free_cash_flow:  Optional[float] = None
    raw:             Dict[str, Any]  = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base extractor
# ---------------------------------------------------------------------------

class ExtractorBase(ABC):
    """
    Abstract base for all provider extractors.

    Subclasses MUST declare:
        SOURCE_NAME     : str   — unique provider slug  (e.g. "alpaca")
        SUPPORTED_ROLES : list  — DataRole members the provider can serve

    Every method raises NotImplementedError by default.
    Subclasses override only the methods their provider actually supports,
    so the pipeline's fallback chain knows to skip unsuitable providers.
    """

    SOURCE_NAME:     str            = ""
    SUPPORTED_ROLES: List[DataRole] = []

    def supports(self, role: DataRole) -> bool:
        return role in self.SUPPORTED_ROLES

    def fetch_ohlcv(
        self,
        symbol:   str,
        start:    str,
        end:      str,
        interval: str = "1d",
    ) -> List[OHLCVRecord]:
        raise NotImplementedError(f"{self.SOURCE_NAME} does not support OHLCV")

    def fetch_profile(self, symbol: str) -> ProfileRecord:
        raise NotImplementedError(f"{self.SOURCE_NAME} does not support PROFILE")

    def fetch_news(
        self,
        symbol: str,
        start:  str,
        end:    str,
        limit:  int = 50,
    ) -> List[NewsRecord]:
        raise NotImplementedError(f"{self.SOURCE_NAME} does not support NEWS")

    def fetch_fundamentals(self, symbol: str) -> FundamentalsRecord:
        raise NotImplementedError(f"{self.SOURCE_NAME} does not support FUNDAMENTALS")
