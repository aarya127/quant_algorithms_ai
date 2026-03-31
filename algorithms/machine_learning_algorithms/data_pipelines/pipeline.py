"""
pipeline.py — MarketDataPipeline: unified fetch interface with role-based routing.

Usage
-----
    from algorithms.machine_learning_algorithms.data_pipelines import (
        MarketDataPipeline, DataRole
    )

    pipe = MarketDataPipeline()

    # Single role
    bars    = pipe.fetch_ohlcv("AAPL", start="2024-01-01", end="2024-12-31")
    profile = pipe.fetch_profile("AAPL")
    news    = pipe.fetch_news("AAPL", start="2024-01-01", end="2024-12-31")

    # Batch — dict keyed by DataRole; failed roles store the Exception
    result = pipe.fetch(
        "AAPL",
        roles=[DataRole.OHLCV, DataRole.PROFILE, DataRole.NEWS],
        start="2024-01-01",
        end="2024-12-31",
    )

Provider chain
--------------
See registry.py for per-role primary / fallback assignments.
The pipeline walks the chain left-to-right and returns the first
successful result.  NotImplementedError (provider doesn't support the
role) → skip silently.  Any other exception → log warning + try next.
"""

import logging
from typing import Any, Dict, List, Optional

from .base import DataRole, ExtractorBase, FundamentalsRecord, NewsRecord, OHLCVRecord, ProfileRecord
from .registry import get_provider_chain
from .extractors.alpaca       import AlpacaExtractor
from .extractors.alphavantage import AlphaVantageExtractor
from .extractors.finnhub      import FinnhubExtractor
from .extractors.fiscal_ai    import FiscalAIExtractor
from .extractors.polygon      import PolygonExtractor
from .extractors.yfinance     import YFinanceExtractor

logger = logging.getLogger(__name__)

# Lazy singleton registry — extractors are instantiated only once per process.
_EXTRACTOR_REGISTRY: Dict[str, ExtractorBase] = {}

_EXTRACTOR_CLASSES: Dict[str, type] = {
    "alpaca":       AlpacaExtractor,
    "finnhub":      FinnhubExtractor,
    "alphavantage": AlphaVantageExtractor,
    "yfinance":     YFinanceExtractor,
    "fiscal_ai":    FiscalAIExtractor,
    "polygon":      PolygonExtractor,
}


def _get_extractor(name: str) -> ExtractorBase:
    if name not in _EXTRACTOR_REGISTRY:
        cls = _EXTRACTOR_CLASSES.get(name)
        if cls is None:
            raise ValueError(f"Unknown provider name: '{name}'")
        _EXTRACTOR_REGISTRY[name] = cls()
    return _EXTRACTOR_REGISTRY[name]


class MarketDataPipeline:
    """
    Unified fetch interface with automatic role-based routing and fallback.

    All public methods accept a symbol string and optional date range
    (ISO-8601 'YYYY-MM-DD' strings).  The pipeline resolves the provider
    chain from registry.py and returns the first successful result.

    The fetch() batch method returns a Dict[DataRole, Any] where failed
    roles store the Exception instance rather than raising, ensuring
    partial success is still usable downstream.
    """

    # -------------------------------------------------------------------
    # Internal: walk the provider chain for a role
    # -------------------------------------------------------------------

    def _run_with_fallback(
        self,
        role:   DataRole,
        method: str,
        *args,
        **kwargs,
    ) -> Any:
        """
        Try each provider in get_provider_chain(role) in order.

        - NotImplementedError / KeyError  → provider skipped silently
        - Any other Exception             → log warning, try next provider
        - All exhausted                   → raise RuntimeError
        """
        chain        = get_provider_chain(role)
        last_exc: Optional[Exception] = None

        for name in chain:
            try:
                extractor = _get_extractor(name)
                return getattr(extractor, method)(*args, **kwargs)
            except (NotImplementedError, KeyError):
                # Provider doesn't implement this role — skip without noise.
                continue
            except Exception as exc:
                logger.warning(
                    "[pipeline] %s.%s failed (%s), trying next provider",
                    name, method, exc,
                )
                last_exc = exc
                continue

        raise RuntimeError(
            f"All providers exhausted for role={role.value} "
            f"method={method}: {last_exc}"
        )

    # -------------------------------------------------------------------
    # Per-role public methods
    # -------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol:   str,
        start:    str,
        end:      str,
        interval: str = "1d",
    ) -> List[OHLCVRecord]:
        """
        Equity OHLCV bars.
        Primary: Alpaca (IEX tape, REST bars)  |  Fallback: yfinance
        """
        return self._run_with_fallback(
            DataRole.OHLCV, "fetch_ohlcv", symbol, start, end, interval
        )

    def fetch_profile(self, symbol: str) -> ProfileRecord:
        """
        Company metadata (name, sector, market cap, employees, …).
        Primary: Finnhub  |  Fallback: yfinance
        """
        return self._run_with_fallback(DataRole.PROFILE, "fetch_profile", symbol)

    def fetch_news(
        self,
        symbol: str,
        start:  str,
        end:    str,
        limit:  int = 50,
    ) -> List[NewsRecord]:
        """
        News articles.
        Primary: Finnhub  |  Fallback: Alpaca REST news
        """
        return self._run_with_fallback(
            DataRole.NEWS, "fetch_news", symbol, start, end, limit
        )

    def fetch_fundamentals(self, symbol: str) -> FundamentalsRecord:
        """
        Financial ratios and statements.
        Primary: Fiscal.ai  |  Fallbacks: Finnhub → AlphaVantage → yfinance
        """
        return self._run_with_fallback(DataRole.FUNDAMENTALS, "fetch_fundamentals", symbol)

    def fetch_calendar(self, start: str, end: str) -> List[dict]:
        """
        Market-hours trading calendar.
        Primary: Alpaca  |  Fallback: Finnhub earnings calendar
        """
        # fetch_calendar has a different signature (no symbol) — call Alpaca directly.
        try:
            return _get_extractor("alpaca").fetch_calendar(start, end)   # type: ignore[attr-defined]
        except Exception as exc:
            logger.warning("[pipeline] alpaca.fetch_calendar failed (%s)", exc)
            return []

    # -------------------------------------------------------------------
    # Batch convenience
    # -------------------------------------------------------------------

    def fetch(
        self,
        symbol: str,
        roles:  List[DataRole],
        start:  str = "",
        end:    str = "",
    ) -> Dict[DataRole, Any]:
        """
        Fetch multiple data roles in one call.

        Returns
        -------
        Dict[DataRole, Any]
            Successful roles map to their result objects.
            Failed roles map to the Exception that caused the failure.
            This lets callers handle partial results gracefully.

        Example
        -------
            result = pipe.fetch(
                "NVDA",
                roles=[DataRole.OHLCV, DataRole.PROFILE, DataRole.NEWS],
                start="2024-01-01", end="2024-06-30",
            )
            bars    = result[DataRole.OHLCV]
            profile = result[DataRole.PROFILE]
        """
        _dispatch = {
            DataRole.OHLCV:        lambda: self.fetch_ohlcv(symbol, start, end),
            DataRole.NEWS:         lambda: self.fetch_news(symbol, start, end),
            DataRole.PROFILE:      lambda: self.fetch_profile(symbol),
            DataRole.FUNDAMENTALS: lambda: self.fetch_fundamentals(symbol),
        }

        result: Dict[DataRole, Any] = {}
        for role in roles:
            fn = _dispatch.get(role)
            if fn is None:
                logger.warning(
                    "[pipeline] fetch() has no dispatch entry for role=%s; skipping",
                    role.value,
                )
                continue
            try:
                result[role] = fn()
            except Exception as exc:
                result[role] = exc

        return result
