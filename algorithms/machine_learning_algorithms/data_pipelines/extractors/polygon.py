"""
extractors/polygon.py — Polygon.io extractor  ("Massive" high-breadth provider).

Primary roles: OPTIONS, OHLCV (SIP consolidated tape), PROFILE

Primary for:
- Full options chain with IV surface and per-contract greeks
- SIP consolidated tape (not IEX subset) — full market depth
- Long historical breadth: 20+ years of daily bars
- Aggregates, tick data, real-time snapshots (paid tiers)

Required env var: POLYGON_API_KEY

Free-tier limits: 5 API calls / min, ~2 years history, 15-min delayed snapshots.
Starter+ plan unlocks real-time + full 20-year history.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from ..base import DataRole, ExtractorBase, FundamentalsRecord, NewsRecord, OHLCVRecord, ProfileRecord

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.polygon.io"

_INTERVAL_MAP: Dict[str, tuple] = {
    "1d":   (1, "day"),
    "1h":   (1, "hour"),
    "30min":(30, "minute"),
    "15min":(15, "minute"),
    "5min": (5,  "minute"),
    "1min": (1,  "minute"),
}


class PolygonExtractor(ExtractorBase):
    """
    Polygon.io extractor — the "Massive" market-data provider role.

    Why Polygon is the OPTIONS primary
    -----------------------------------
    Polygon's /v3/snapshot/options/{symbol} endpoint returns the full
    options surface per symbol in one call:
      • strike, expiration, contract_type, exercise_style
      • open_interest, volume, day OHLC
      • greeks (delta, gamma, theta, vega) — Starter+ plan
      • implied_volatility, theoretical value

    Why Polygon is the SIP-tape OHLCV source
    -----------------------------------------
    Alpaca's free data feed is IEX-only (~15% of volume).  Polygon's
    aggregates endpoint uses the full SIP consolidated tape, giving
    official OHLCV including dark-pool volume.  Critical for backtests
    that need accurate volume and accurate VWAP.
    """

    SOURCE_NAME     = "polygon"
    SUPPORTED_ROLES = [DataRole.OHLCV, DataRole.OPTIONS, DataRole.PROFILE]

    def __init__(self):
        self._api_key = os.environ.get("POLYGON_API_KEY", "")

    def _require_key(self) -> None:
        if not self._api_key:
            raise RuntimeError(
                "POLYGON_API_KEY environment variable is not set. "
                "Obtain a key at https://polygon.io and export it."
            )

    def _get(self, path: str, params: Optional[Dict] = None) -> Any:
        self._require_key()
        p = {"apiKey": self._api_key, **(params or {})}
        resp = requests.get(f"{_BASE_URL}{path}", params=p, timeout=15)
        resp.raise_for_status()
        return resp.json()

    # -------------------------------------------------------------------
    # OHLCV  (SIP consolidated tape — primary for long-history backtests)
    # -------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol:   str,
        start:    str,
        end:      str,
        interval: str = "1d",
    ) -> List[OHLCVRecord]:
        """
        Polygon aggregates bars using the SIP consolidated tape.

        Parameters
        ----------
        interval : '1d' | '1h' | '30min' | '15min' | '5min' | '1min'
        start    : 'YYYY-MM-DD'
        end      : 'YYYY-MM-DD'

        Polygon returns timestamps in milliseconds UTC epoch.
        Max 50,000 bars per request (sufficient for daily over 20 years).
        """
        mult, span = _INTERVAL_MAP.get(interval, (1, "day"))
        path = f"/v2/aggs/ticker/{symbol.upper()}/range/{mult}/{span}/{start}/{end}"
        records: List[OHLCVRecord] = []
        try:
            data = self._get(path, {"adjusted": "true", "sort": "asc", "limit": 50000})
            for bar in data.get("results") or []:
                # bar["t"] is milliseconds UTC epoch
                dt = datetime.fromtimestamp(bar["t"] / 1000.0, tz=timezone.utc)
                records.append(OHLCVRecord(
                    date=          dt.strftime("%Y-%m-%d"),
                    open=          bar["o"],
                    high=          bar["h"],
                    low=           bar["l"],
                    close=         bar["c"],
                    volume=        int(bar["v"]),
                    adjusted_close=bar.get("vw"),   # VWAP; separate adj_close not in free tier
                ))
        except Exception as exc:
            logger.warning("PolygonExtractor.fetch_ohlcv failed for %s: %s", symbol, exc)
        return records

    # -------------------------------------------------------------------
    # OPTIONS CHAIN  (primary)
    # -------------------------------------------------------------------

    def fetch_options_chain(
        self,
        symbol:          str,
        expiration_date: Optional[str] = None,     # 'YYYY-MM-DD' filter or None for all
        contract_type:   Optional[str] = None,     # 'call' | 'put' | None
        limit:           int           = 250,
    ) -> List[Dict]:
        """
        Full options surface via Polygon snapshot API.

        Returns raw contract dicts including:
          implied_volatility, greeks (delta/gamma/theta/vega on paid tiers),
          open_interest, volume, day OHLC, theoretical value

        Parameters
        ----------
        expiration_date : filter to a specific expiry (optional)
        contract_type   : 'call' | 'put' | None (returns both)
        limit           : contracts per page (max 250; pagination not implemented)
        """
        params: Dict = {"limit": limit, "order": "asc", "sort": "expiration_date"}
        if expiration_date:
            params["expiration_date"] = expiration_date
        if contract_type:
            params["contract_type"] = contract_type.lower()
        try:
            data = self._get(f"/v3/snapshot/options/{symbol.upper()}", params)
            return data.get("results") or []
        except Exception as exc:
            logger.warning(
                "PolygonExtractor.fetch_options_chain failed for %s: %s", symbol, exc
            )
            return []

    # -------------------------------------------------------------------
    # PROFILE  (Polygon ticker details: SIC, exchange, description)
    # -------------------------------------------------------------------

    def fetch_profile(self, symbol: str) -> ProfileRecord:
        """
        Polygon reference ticker endpoint returns:
        naics_code, sic_code, sic_description, primary_exchange,
        description, homepage_url, market_cap, total_employees.
        """
        try:
            data = self._get(f"/v3/reference/tickers/{symbol.upper()}")
            r = data.get("results") or {}
            return ProfileRecord(
                symbol=      symbol,
                name=        r.get("name", ""),
                exchange=    r.get("primary_exchange", ""),
                sector=      r.get("sic_description", ""),
                industry=    r.get("sic_description", ""),
                market_cap=  r.get("market_cap"),
                employees=   r.get("total_employees"),
                description= r.get("description"),
                website=     r.get("homepage_url"),
                country=     (r.get("locale") or "").upper(),
            )
        except Exception as exc:
            logger.warning("PolygonExtractor.fetch_profile failed for %s: %s", symbol, exc)
            raise

    # -------------------------------------------------------------------
    # Unsupported
    # -------------------------------------------------------------------

    def fetch_fundamentals(self, symbol: str) -> FundamentalsRecord:
        raise NotImplementedError("PolygonExtractor: use FiscalAIExtractor for FUNDAMENTALS")

    def fetch_news(self, symbol: str, start: str, end: str, limit: int = 50) -> List[NewsRecord]:
        raise NotImplementedError("PolygonExtractor: use FinnhubExtractor for NEWS")
