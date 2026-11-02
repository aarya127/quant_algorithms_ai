"""
membership.py — point-in-time S&P 500 membership, for a survivorship-aware universe.

Source: github.com/fja05680/sp500 (MIT), sp500_ticker_start_end.csv — one row
per membership spell (ticker, start_date, end_date; no end = still a member).
Cached for 7 days in .pipeline_cache/.

A company that left the index is kept for the dates it was a member: that's
what removes survivorship bias. Prices come from yfinance, which drops many
delisted tickers; RENAMES maps an old ticker to the one yfinance files its
history under, and a mapping is used only if that ticker's prices cover the old
membership. What's still missing (mostly acquired or failed companies) is
reported by `coverage_report`, so results can say how much of the index they
don't see.
"""
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".pipeline_cache"
SOURCE = "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"
CACHE_TTL = 7 * 86400

# old index ticker → the ticker yfinance keeps the company's history under
# (renames and ticker changes, not mergers into a different company)
RENAMES = {
    "ABC": "COR",      # AmerisourceBergen → Cencora (2023)
    "ANTM": "ELV",     # Anthem → Elevance Health (2022)
    "BLL": "BALL",     # Ball Corp (2022)
    "COG": "CTRA",     # Cabot Oil & Gas → Coterra (2021)
    "FBHS": "FBIN",    # Fortune Brands (2022)
    "FLT": "CPAY",     # FleetCor → Corpay (2024)
    "PEAK": "DOC",     # Healthpeak (2024)
    "PKI": "RVTY",     # PerkinElmer → Revvity (2023)
    "RE": "EG",        # Everest Re → Everest Group (2023)
    "NLOK": "GEN",     # NortonLifeLock → Gen Digital (2022)
    "WLTW": "WTW",     # Willis Towers Watson (2022)
    "HFC": "DINO",     # HollyFrontier → HF Sinclair (2022)
    "DISCA": "WBD",    # Discovery → Warner Bros. Discovery (2022)
    "VIAC": "PARA",    # ViacomCBS → Paramount (2022)
    "GPS": "GAP",      # Gap Inc. (2024)
    "CDAY": "DAY",     # Ceridian → Dayforce (2024)
    "FI": "FISV",      # Fiserv's ticker changes
    "FB": "META",      # Facebook → Meta Platforms (2022)
    "LB": "BBWI",      # L Brands → Bath & Body Works (2021)
}


def _yf(ticker: str) -> str:
    return ticker.replace(".", "-")


def load_spells(refresh: bool = False) -> pd.DataFrame:
    """Membership spells: ticker, start, end (NaT = still a member)."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / "sp500_ticker_start_end.csv"
    if refresh or not path.exists() or time.time() - path.stat().st_mtime > CACHE_TTL:
        import requests
        resp = requests.get(SOURCE, timeout=60)
        resp.raise_for_status()
        path.write_text(resp.text)
    df = pd.read_csv(path, parse_dates=["start_date", "end_date"])
    return df.rename(columns={"start_date": "start", "end_date": "end"})[["ticker", "start", "end"]]


def spells_in_window(spells: pd.DataFrame, start, end) -> pd.DataFrame:
    """Spells overlapping [start, end], clipped to it, with yfinance-style tickers."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    s = spells[(spells["start"] <= end) & (spells["end"].isna() | (spells["end"] >= start))].copy()
    s["ticker"] = s["ticker"].map(_yf)
    s["start"] = s["start"].clip(lower=start)
    s["end"] = s["end"].fillna(end).clip(upper=end)
    return s.reset_index(drop=True)


def price_symbols(spells: pd.DataFrame, available: dict) -> pd.DataFrame:
    """Add `price_symbol`: the ticker itself if yfinance has it, else its
    RENAMES target if that ticker's history covers the spell, else None.
    available: {ticker: (first_date, last_date)} of downloaded histories."""
    def pick(row):
        for cand in (row["ticker"], RENAMES.get(row["ticker"])):
            if cand and cand in available:
                first, last = available[cand]
                # must reach back to the spell's start (with a week's slack) and past it
                if first <= row["start"] + pd.Timedelta(days=7) and last >= row["start"]:
                    return cand
        return None
    out = spells.copy()
    out["price_symbol"] = out.apply(pick, axis=1) if len(out) else []
    return out


def in_index(dates, spells: pd.DataFrame) -> pd.DataFrame:
    """Long frame (Date, symbol, in_index=True) for every date a price symbol
    was a member (spells merged per price symbol)."""
    dates = pd.DatetimeIndex(dates)
    parts = []
    for sym, g in spells.dropna(subset=["price_symbol"]).groupby("price_symbol"):
        mask = pd.Series(False, index=dates)
        for _, r in g.iterrows():
            mask |= (dates >= r["start"]) & (dates <= r["end"])
        d = dates[mask.to_numpy()]
        parts.append(pd.DataFrame({"Date": d, "symbol": sym, "in_index": True}))
    if not parts:
        return pd.DataFrame(columns=["Date", "symbol", "in_index"])
    return pd.concat(parts, ignore_index=True)


def coverage_report(spells: pd.DataFrame, dates) -> dict:
    """How much of the index the panel can't see: spells with no price symbol,
    as a count and as a share of all member-days in the window."""
    dates = pd.DatetimeIndex(dates)

    def days(r):
        return int(((dates >= r["start"]) & (dates <= r["end"])).sum())

    s = spells.copy()
    s["member_days"] = s.apply(days, axis=1) if len(s) else []
    missing = s[s["price_symbol"].isna()]
    total = int(s["member_days"].sum())
    return {
        "companies": int(s["ticker"].nunique()),
        "priced": int(s.loc[s["price_symbol"].notna(), "ticker"].nunique()),
        "missing": sorted(missing["ticker"].unique().tolist()),
        "missing_member_day_share": float(missing["member_days"].sum() / total) if total else 0.0,
        "renamed": {t: RENAMES[t] for t in s.loc[s["price_symbol"].ne(s["ticker"]) & s["price_symbol"].notna(), "ticker"]},
    }
