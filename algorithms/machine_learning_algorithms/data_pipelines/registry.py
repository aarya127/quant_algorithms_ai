"""
registry.py — Role-to-provider assignments.

Each DataRole maps to exactly one primary provider and an ordered
fallback chain.  The pipeline walks this chain until a call succeeds.

Role assignment rationale
-------------------------
Alpaca       primary: OHLCV (equity bars), CALENDAR (market hours)
             supplement: NEWS (REST news endpoint)

Finnhub      primary: PROFILE, NEWS, SENTIMENT
             secondary: FUNDAMENTALS, CALENDAR (earnings dates)

AlphaVantage primary: MACRO (GDP, rates, CPI, energy) — ONLY role assigned.
             25 req/day on free tier, shared with other running services.
             Do NOT put in any fallback chain other than MACRO.

yfinance     primary: TECHNICALS (unlimited, built-in indicators)
             fallback: OHLCV, PROFILE, FUNDAMENTALS, OPTIONS (basic surface)

Fiscal.ai    primary: FUNDAMENTALS, FILINGS
             (structured earnings / accounting-derived signals)

Polygon      primary: OPTIONS (full chain, IV, greeks)
             also: OHLCV (SIP tape, 20+ yr history), PROFILE (SIC metadata)
"""

from typing import Dict, List
from .base import DataRole

# ---------------------------------------------------------------------------
# Registry: DataRole → {primary, fallbacks}
# ---------------------------------------------------------------------------

ROLE_REGISTRY: Dict[DataRole, Dict] = {
    DataRole.OHLCV: {
        "primary":   "alpaca",
        "fallbacks": ["yfinance"],
        "note": "Alpaca REST bars (IEX feed, adjusted); yfinance as resilient backup",
    },
    DataRole.PROFILE: {
        "primary":   "finnhub",
        "fallbacks": ["yfinance"],
        "note": "Finnhub company/profile richest on free tier; yfinance covers basics",
    },
    DataRole.NEWS: {
        "primary":   "finnhub",
        "fallbacks": ["alpaca"],
        "note": "Finnhub company news + event calendar; Alpaca REST news as supplement",
    },
    DataRole.FUNDAMENTALS: {
        "primary":   "fiscal_ai",
        "fallbacks": ["yfinance"],
        "note": "Fiscal.ai (250/day) structured accounting signals; yfinance unlimited backbone. "
                "AV removed: 25-call/day budget reserved for MACRO only.",
    },
    DataRole.TECHNICALS: {
        "primary":   "yfinance",
        "fallbacks": [],
        "note": "yfinance covers all standard indicators with no rate limit. "
                "AV removed: 25-call/day budget reserved for MACRO only.",
    },
    DataRole.OPTIONS: {
        "primary":   "polygon",
        "fallbacks": ["yfinance"],
        "note": "Polygon full options chain with IV surface and greeks (Starter+ plan). "
                "Free tier: 5 req/min — avoid tight loops. yfinance for basic surface fallback.",
    },
    DataRole.FILINGS: {
        "primary":   "fiscal_ai",
        "fallbacks": [],
        "note": "Fiscal.ai (250/day) parsed SEC filings only. "
                "AV removed: 25-call/day budget reserved for MACRO only.",
    },
    DataRole.MACRO: {
        "primary":   "alphavantage",
        "fallbacks": ["yfinance"],
        "note": "AV macro (GDP, Fed funds, CPI, oil) — only 25 req/day shared across services. "
                "Cache results aggressively; macro series rarely update intraday.",
    },
    DataRole.CALENDAR: {
        "primary":   "alpaca",
        "fallbacks": ["finnhub", "yfinance"],
        "note": "Alpaca market-hours calendar; Finnhub earnings calendar as fallback",
    },
    DataRole.SENTIMENT: {
        "primary":   "finnhub",
        "fallbacks": ["alpaca"],
        "note": "Finnhub insider sentiment + news; Alpaca WebSocket news as supplement",
    },
}


def get_provider_chain(role: DataRole) -> List[str]:
    """Return [primary, *fallbacks] for a given DataRole."""
    entry = ROLE_REGISTRY[role]
    return [entry["primary"]] + entry.get("fallbacks", [])


def describe_role(role: DataRole) -> str:
    """Human-readable note for a given DataRole assignment."""
    return ROLE_REGISTRY[role].get("note", "")
