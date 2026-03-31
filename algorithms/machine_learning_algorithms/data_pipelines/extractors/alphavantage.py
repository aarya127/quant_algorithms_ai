"""
extractors/alphavantage.py — Alpha Vantage extractor.

Primary role : MACRO  (GDP, Fed funds rate, CPI, oil, natural gas)
NO OTHER ROLES — see budget warning below.

Wraps the existing data/alphavantage.py AlphaVantage class.

BUDGET WARNING
--------------
Free tier : 25 API calls / day — shared across ALL running services.
Other services in this project (backend, stock_analyzer, etc.) already
consume calls from this budget.  Treat remaining headroom as ~10-15/day.

Rules:
  1. AlphaVantage is ONLY in the MACRO role chain.
  2. Never add it to OHLCV / TECHNICALS / FUNDAMENTALS / FILINGS fallbacks.
  3. Cache macro responses wherever possible — GDP and CPI update monthly,
     Fed Funds and yields update weekly.
  4. If the day's budget is likely exhausted, the call will raise HTTP 429;
     the pipeline will surface that exception rather than silently retrying.
"""

import logging
from typing import List, Optional

from ..base import DataRole, ExtractorBase, FundamentalsRecord, NewsRecord, OHLCVRecord, ProfileRecord

try:
    from data.alphavantage import AlphaVantage
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

logger = logging.getLogger(__name__)

_RATE_LIMIT_WARNING = (
    "AlphaVantage: free tier is limited to 25 req/day. "
    "This call costs 1 request — use sparingly."
)


def _to_float(v) -> Optional[float]:
    try:
        return float(v) if v not in (None, "None", "N/A", "") else None
    except (TypeError, ValueError):
        return None


def _safe_margin(numerator, denominator) -> Optional[float]:
    n, d = _to_float(numerator), _to_float(denominator)
    if n is None or d is None or d == 0:
        return None
    return n / d


class AlphaVantageExtractor(ExtractorBase):
    """
    Alpha Vantage extractor for macro data and technical indicators.

    MACRO      : GDP, Federal Funds Rate, CPI, WTI crude, Brent, Natural Gas
    TECHNICALS : SMA, EMA, RSI, ADX (fallback — yfinance is primary)
    FILINGS    : Income statement, balance sheet, cash flow (fallback)
    OHLCV      : Global Quote snapshot only — NOT a range; last-resort backup
    """

    SOURCE_NAME     = "alphavantage"
    SUPPORTED_ROLES = [
        DataRole.MACRO,
        DataRole.TECHNICALS,
        DataRole.FILINGS,
        DataRole.OHLCV,   # backup only — single snapshot, not a range
    ]

    def __init__(self):
        self._client: Optional[AlphaVantage] = AlphaVantage() if _AVAILABLE else None

    def _require(self) -> AlphaVantage:
        if self._client is None:
            raise RuntimeError(
                "data/alphavantage.py not importable or ALPHAVANTAGE_API_KEY not set."
            )
        return self._client

    # -------------------------------------------------------------------
    # MACRO  (primary role)
    # -------------------------------------------------------------------

    def fetch_macro(self, indicator: str, interval: str = "annual") -> dict:
        """
        Fetch a macro time-series from Alpha Vantage.

        Parameters
        ----------
        indicator : one of "gdp" | "fed_funds" | "cpi" | "inflation" |
                    "wti" | "brent" | "natural_gas" | "treasury_yield"
        interval  : "annual" | "quarterly" | "monthly" | "daily" | "weekly"

        Returns raw Alpha Vantage response dict.
        """
        logger.info(_RATE_LIMIT_WARNING)
        client = self._require()
        try:
            handlers = {
                "gdp":           lambda: client.get_real_gdp(interval),
                "fed_funds":     lambda: client.get_federal_funds_rate(interval),
                "cpi":           lambda: client.get_inflation(interval),
                "inflation":     lambda: client.get_inflation(interval),
                "wti":           lambda: client.get_wti_crude_oil(interval),
                "brent":         lambda: client.get_brent_crude_oil(interval),
                "natural_gas":   lambda: client.get_natural_gas(interval),
                "treasury_yield": lambda: client.get_treasury_yield(interval, maturity="10year"),
            }
            fn = handlers.get(indicator.lower())
            if fn is None:
                raise ValueError(
                    f"Unknown macro indicator '{indicator}'. "
                    f"Valid options: {list(handlers.keys())}"
                )
            return fn()
        except Exception as exc:
            logger.warning("AlphaVantageExtractor.fetch_macro failed for %s: %s", indicator, exc)
            raise

    # -------------------------------------------------------------------
    # TECHNICALS  (fallback — yfinance is primary)
    # -------------------------------------------------------------------

    def fetch_technicals(
        self,
        symbol:      str,
        indicator:   str   = "rsi",
        interval:    str   = "daily",
        time_period: int   = 14,
    ) -> dict:
        """
        Fetch a technical indicator from Alpha Vantage.

        Parameters
        ----------
        indicator : "sma" | "ema" | "rsi" | "adx"
        interval  : "daily" | "weekly" | "monthly" | "60min" | "30min" | "15min"
        """
        logger.info(_RATE_LIMIT_WARNING)
        client = self._require()
        try:
            handlers = {
                "sma": lambda: client.get_sma(symbol, interval, time_period, "close"),
                "ema": lambda: client.get_ema(symbol, interval, time_period, "close"),
                "rsi": lambda: client.get_rsi(symbol, interval, time_period, "close"),
                "adx": lambda: client.get_adx(symbol, interval, time_period),
            }
            fn = handlers.get(indicator.lower())
            if fn is None:
                raise ValueError(f"Unknown indicator '{indicator}'. Valid: {list(handlers.keys())}")
            return fn()
        except Exception as exc:
            logger.warning(
                "AlphaVantageExtractor.fetch_technicals failed for %s %s: %s",
                symbol, indicator, exc
            )
            raise

    # -------------------------------------------------------------------
    # FILINGS / FUNDAMENTALS  (fallback — Fiscal.ai is primary)
    # -------------------------------------------------------------------

    def fetch_fundamentals(self, symbol: str) -> FundamentalsRecord:
        """Income statement + balance sheet combined into FundamentalsRecord."""
        logger.info(_RATE_LIMIT_WARNING)
        client = self._require()
        try:
            inc    = client.get_income_statement(symbol)
            bs     = client.get_balance_sheet(symbol)
            annual = (inc.get("annualReports") or [{}])[0]
            bs_ann = (bs.get("annualReports")  or [{}])[0]
            return FundamentalsRecord(
                symbol=         symbol,
                period=         annual.get("fiscalDateEnding", "annual"),
                revenue=        _to_float(annual.get("totalRevenue")),
                net_income=     _to_float(annual.get("netIncome")),
                eps=            _to_float(annual.get("reportedEPS")),
                gross_margin=   _safe_margin(
                                    annual.get("grossProfit"),
                                    annual.get("totalRevenue"),
                                ),
                debt_to_equity= _safe_margin(
                                    bs_ann.get("totalLiabilities"),
                                    bs_ann.get("totalShareholderEquity"),
                                ),
                raw={"income_annual": annual, "balance_sheet_annual": bs_ann},
            )
        except Exception as exc:
            logger.warning("AlphaVantageExtractor.fetch_fundamentals failed for %s: %s", symbol, exc)
            raise

    # -------------------------------------------------------------------
    # OHLCV  (last-resort backup — single snapshot, not a range)
    # -------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol:   str,
        start:    str,
        end:      str,
        interval: str = "1d",
    ) -> List[OHLCVRecord]:
        """
        Returns the latest single-day Global Quote snapshot.
        Not a date range — use Alpaca or yfinance for historical bars.
        Reserved for environments where neither Alpaca nor yfinance is available.
        """
        logger.warning(
            "AlphaVantageExtractor.fetch_ohlcv returns a SINGLE snapshot, not a range. "
            "Prefer AlpacaExtractor or YFinanceExtractor for historical bars."
        )
        logger.info(_RATE_LIMIT_WARNING)
        client = self._require()
        try:
            raw = client.get_global_quote(symbol)
            q   = raw.get("Global Quote", {})
            if not q:
                return []
            return [OHLCVRecord(
                date=           q.get("07. latest trading day", ""),
                open=           float(q.get("02. open",   0) or 0),
                high=           float(q.get("03. high",   0) or 0),
                low=            float(q.get("04. low",    0) or 0),
                close=          float(q.get("05. price",  0) or 0),
                volume=         int(float(q.get("06. volume", 0) or 0)),
                adjusted_close= float(q.get("05. price",  0) or 0),
            )]
        except Exception as exc:
            logger.warning("AlphaVantageExtractor.fetch_ohlcv failed for %s: %s", symbol, exc)
            return []

    # -------------------------------------------------------------------
    # Unsupported
    # -------------------------------------------------------------------

    def fetch_profile(self, symbol: str) -> ProfileRecord:
        raise NotImplementedError("AlphaVantageExtractor: use FinnhubExtractor for PROFILE")

    def fetch_news(self, symbol: str, start: str, end: str, limit: int = 50) -> List[NewsRecord]:
        raise NotImplementedError("AlphaVantageExtractor: use FinnhubExtractor for NEWS")
