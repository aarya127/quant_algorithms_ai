"""
fundamentals_pit.py — point-in-time fundamentals and earnings surprises for a
universe panel.

    python fundamentals_pit.py [UNIVERSE]      # default large50; run universe.py first

Adds columns to UNIVERSE_<name>_panel.csv, each value usable only from the
business day after it became public:

  from SEC filings (Finnhub "financials as reported", with each filing's date):
    gross_profitability  gross profit TTM / total assets           (Novy-Marx 2013)
    roa, roe             net income TTM / assets, / equity (equity > 0)
    accruals             (net income - operating cash flow) TTM / assets   (Sloan 1996)
    asset_growth         total assets vs a year earlier            (Cooper et al. 2008)
    revenue_growth       revenue TTM vs a year earlier
    leverage             total liabilities / total assets
    earnings_yield       net income TTM / market cap
    book_to_market       equity / market cap (equity > 0)
    sales_to_price       revenue TTM / market cap
  from earnings announcements (yfinance: estimate, reported EPS):
    eps_surprise         latest (reported - estimate) / |estimate|, clipped to ±1
    days_since_earnings  sessions since that announcement

Point in time means: each filing's own numbers as first reported (no later
restatements), a 10-Q's year-to-date figures turned into trailing twelve months
only with filings that came before it, and a value a filing lacks (a balance
sheet, or a TTM whose prior-year piece is missing from the data) carried
forward from an earlier filing. Share counts not tagged in the income
statement are net income / diluted EPS for the same period. Market cap uses yfinance's
split-adjusted (not dividend-adjusted) close and the filing's share count
converted to today's share basis with the splits after its filing date — a
change of units, not of information.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".pipeline_cache"
CACHE_TTL = 86400

# first concept present wins (companies tag the same line differently)
FLOWS = {
    "revenue": ("ic", ["us-gaap_Revenues", "us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax",
                       "us-gaap_RevenueFromContractWithCustomerIncludingAssessedTax",
                       "us-gaap_SalesRevenueNet", "us-gaap_SalesRevenueGoodsNet"]),
    "gross_profit": ("ic", ["us-gaap_GrossProfit"]),
    "cost_of_revenue": ("ic", ["us-gaap_CostOfRevenue", "us-gaap_CostOfGoodsAndServicesSold",
                               "us-gaap_CostOfGoodsSold"]),
    "net_income": ("ic", ["us-gaap_NetIncomeLoss", "us-gaap_ProfitLoss",
                          "us-gaap_NetIncomeLossAvailableToCommonStockholdersBasic"]),
    "cfo": ("cf", ["us-gaap_NetCashProvidedByUsedInOperatingActivities",
                   "us-gaap_NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"]),
}
INSTANTS = {
    "assets": ("bs", ["us-gaap_Assets"]),
    "equity": ("bs", ["us-gaap_StockholdersEquity",
                      "us-gaap_StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]),
    "liabilities": ("bs", ["us-gaap_Liabilities"]),
    "shares": ("ic", ["us-gaap_WeightedAverageNumberOfDilutedSharesOutstanding",
                      "us-gaap_WeightedAverageNumberOfSharesOutstandingBasic"]),
}
FUND_COLS = ["gross_profitability", "roa", "roe", "accruals", "asset_growth", "revenue_growth",
             "leverage", "earnings_yield", "book_to_market", "sales_to_price"]
EARN_COLS = ["eps_surprise", "days_since_earnings"]


def _value(report, section, concepts):
    items = {x.get("concept"): x.get("value") for x in report.get(section) or []}
    for c in concepts:
        v = items.get(c)
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            return v
    return np.nan


def filings_frame(quarterly, annual) -> pd.DataFrame:
    """Finnhub as-reported filings → one row per filing: fy, fq (4 = 10-K),
    end, filed, year-to-date flows and instants, as reported in that filing."""
    rows = []
    for rep, is_annual in [(r, False) for r in quarterly or []] + [(r, True) for r in annual or []]:
        try:
            row = {"fy": int(rep["year"]), "fq": 4 if is_annual else int(rep["quarter"]),
                   "end": pd.Timestamp(str(rep["endDate"])[:10]),
                   "filed": pd.Timestamp(str(rep["filedDate"])[:10])}
        except (KeyError, TypeError, ValueError):
            continue
        if not is_annual and row["fq"] not in (1, 2, 3):
            continue
        report = rep.get("report") or {}
        for name, (sec, concepts) in {**FLOWS, **INSTANTS}.items():
            row[name] = _value(report, sec, concepts)
        if np.isnan(row["gross_profit"]) and np.isfinite(row["revenue"]) and np.isfinite(row["cost_of_revenue"]):
            row["gross_profit"] = row["revenue"] - row["cost_of_revenue"]
        if not np.isfinite(row["shares"]):
            # weighted diluted shares = net income / diluted EPS, same period, same filing
            eps = _value(report, "ic", ["us-gaap_EarningsPerShareDiluted", "us-gaap_EarningsPerShareBasic"])
            if np.isfinite(eps) and abs(eps) > 1e-9 and np.isfinite(row["net_income"]):
                implied = row["net_income"] / eps
                row["shares"] = implied if implied > 0 else np.nan
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # an amended filing for the same period: keep the first one filed (what was known then)
    df = df.sort_values("filed").drop_duplicates(["fy", "fq"], keep="first")
    return df.sort_values(["filed", "end"]).reset_index(drop=True)


def ttm_frame(filings: pd.DataFrame) -> pd.DataFrame:
    """Per filing: trailing-twelve-month flows from that filing and earlier ones,
    instants carried forward from earlier filings when missing, and year-ago
    values for growth. `available` = the business day after filing."""
    if filings.empty:
        return pd.DataFrame()
    f = filings.set_index(["fy", "fq"])
    out = []
    seen = []
    for _, r in filings.iterrows():
        fy, fq = r["fy"], r["fq"]
        row = {"end": r["end"], "filed": r["filed"],
               "available": r["filed"] + pd.offsets.BDay(1)}
        for name in FLOWS:
            if fq == 4:
                row[name] = r[name]
            else:
                prev_annual = f[name].get((fy - 1, 4), np.nan)
                prev_ytd = f[name].get((fy - 1, fq), np.nan)
                known = all((k in f.index) and f.loc[k, "filed"] <= r["filed"]
                            for k in [(fy - 1, 4), (fy - 1, fq)])
                row[name] = r[name] + prev_annual - prev_ytd if known else np.nan
            if not np.isfinite(row[name]):          # carry the last TTM built
                row[name] = next((s[name + "_ttm"] for s in reversed(seen)
                                  if np.isfinite(s[name + "_ttm"])), np.nan)
        for name in INSTANTS:
            v = r[name]
            if not np.isfinite(v):                  # carry the last earlier value
                v = next((s[name] for s in reversed(seen) if np.isfinite(s[name])), np.nan)
            row[name] = v
        # a year earlier: the filing for the same fiscal quarter of the prior year
        ago = next((s for s in reversed(seen) if s["fy"] == fy - 1 and s["fq"] == fq), None)
        row["assets_year_ago"] = ago["assets"] if ago else np.nan
        row["revenue_year_ago"] = ago["revenue_ttm"] if ago else np.nan
        seen.append({"fy": fy, "fq": fq, **{k + "_ttm": row[k] for k in FLOWS},
                     **{k: row[k] for k in INSTANTS}})
        out.append(row)
    return pd.DataFrame(out)


def split_factor_after(splits: pd.Series, when: pd.Timestamp) -> float:
    """Product of split ratios dated after `when` (shares then → today's basis)."""
    if splits is None or len(splits) == 0:
        return 1.0
    idx = pd.to_datetime(splits.index).tz_localize(None) if getattr(splits.index, "tz", None) \
        else pd.to_datetime(splits.index)
    later = splits.to_numpy(dtype=float)[idx > when]
    return float(np.prod(later)) if len(later) else 1.0


def fundamentals_for(prices: pd.DataFrame, ttm: pd.DataFrame, splits=None) -> pd.DataFrame:
    """Daily point-in-time ratios for one ticker. prices: Date, Close_raw
    (split-adjusted, not dividend-adjusted). Each date uses the latest filing
    available on or before it."""
    base = prices[["Date", "Close_raw"]].sort_values("Date")
    if ttm.empty:
        return base.assign(**{c: np.nan for c in FUND_COLS})[["Date"] + FUND_COLS]
    t = ttm.sort_values("available").copy()
    t["shares_today_basis"] = [s * split_factor_after(splits, d) if np.isfinite(s) else np.nan
                               for s, d in zip(t["shares"], t["filed"])]
    m = pd.merge_asof(base, t, left_on="Date", right_on="available", direction="backward")
    pos_eq = m["equity"].where(m["equity"] > 0)
    assets = m["assets"].where(m["assets"] > 0)
    mcap = (m["Close_raw"] * m["shares_today_basis"]).where(lambda x: x > 0)
    m["gross_profitability"] = m["gross_profit"] / assets
    m["roa"] = m["net_income"] / assets
    m["roe"] = m["net_income"] / pos_eq
    m["accruals"] = (m["net_income"] - m["cfo"]) / assets
    m["asset_growth"] = m["assets"] / m["assets_year_ago"].where(m["assets_year_ago"] > 0) - 1
    m["revenue_growth"] = m["revenue"] / m["revenue_year_ago"].where(m["revenue_year_ago"] > 0) - 1
    m["leverage"] = m["liabilities"] / assets
    m["earnings_yield"] = m["net_income"] / mcap
    m["book_to_market"] = pos_eq / mcap
    m["sales_to_price"] = m["revenue"] / mcap
    m[FUND_COLS] = m[FUND_COLS].replace([np.inf, -np.inf], np.nan)
    return m[["Date"] + FUND_COLS]


def earnings_for(dates: pd.Series, events: pd.DataFrame) -> pd.DataFrame:
    """Daily latest-surprise features for one ticker. events: announced
    (tz-naive local timestamp), surprise (fraction). A report at or after the
    9:30 open is usable from the next session; a pre-market one the same day."""
    base = pd.DataFrame({"Date": pd.to_datetime(dates).sort_values().to_numpy()})
    if events is None or events.empty:
        return base.assign(eps_surprise=np.nan, days_since_earnings=np.nan)
    ev = events.dropna(subset=["surprise"]).copy()
    ts = pd.to_datetime(ev["announced"])
    pre_open = (ts.dt.hour * 60 + ts.dt.minute) < 9 * 60 + 30
    ev["available"] = np.where(pre_open, ts.dt.normalize(), ts.dt.normalize() + pd.offsets.BDay(1))
    ev["available"] = pd.to_datetime(ev["available"])
    ev = ev.sort_values("available")[["available", "surprise"]]
    m = pd.merge_asof(base, ev, left_on="Date", right_on="available", direction="backward")
    m["eps_surprise"] = m["surprise"].clip(-1, 1)
    sessions = pd.Series(np.arange(len(base)), index=base["Date"])
    avail_pos = np.searchsorted(base["Date"].to_numpy(), m["available"].to_numpy(dtype="datetime64[ns]"))
    m["days_since_earnings"] = np.where(m["available"].notna(), sessions.to_numpy() - avail_pos, np.nan)
    return m[["Date"] + EARN_COLS]


# ---------------------------------------------------------------- fetching

def _cached_json(name, fetch):
    CACHE.mkdir(exist_ok=True)
    path = CACHE / name
    if path.exists() and time.time() - path.stat().st_mtime < CACHE_TTL:
        try:
            return json.loads(path.read_text())
        except ValueError:
            pass
    data = fetch()
    path.write_text(json.dumps(data))
    return data


def fetch_filings(symbol: str):
    """(quarterly, annual) as-reported filings from Finnhub, cached 24 h."""
    sys.path.insert(0, str(HERE.parents[2]))
    from data.finnhub import _client
    out = []
    for freq in ("quarterly", "annual"):
        def call(freq=freq):
            time.sleep(1.1)                               # Finnhub free tier: 60/min
            return (_client.financials_reported(symbol=symbol.replace("-", "."), freq=freq) or {}).get("data", [])
        out.append(_cached_json(f"fh_reported_{symbol}_{freq}.json", call))
    return out


def fetch_earnings(symbol: str) -> pd.DataFrame:
    """Earnings announcements with surprise (fraction) from yfinance, cached 24 h."""
    def call():
        import yfinance as yf
        e = yf.Ticker(symbol).get_earnings_dates(limit=60)
        if e is None or e.empty:
            return []
        e = e.dropna(subset=["Reported EPS", "EPS Estimate"])
        return [{"announced": str(ix.tz_localize(None) if ix.tzinfo else ix),
                 "estimate": float(r["EPS Estimate"]), "reported": float(r["Reported EPS"])}
                for ix, r in e.iterrows()]
    rows = _cached_json(f"yf_earnings_{symbol}.json", call)
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["announced", "surprise"])
    est = df["estimate"].abs().where(df["estimate"].abs() > 1e-9)
    df["surprise"] = (df["reported"] - df["estimate"]) / est
    df["announced"] = pd.to_datetime(df["announced"])
    return df[["announced", "surprise"]]


def fetch_splits(symbol: str) -> pd.Series:
    def call():
        import yfinance as yf
        s = yf.Ticker(symbol).splits
        return {str(ix.date()): float(v) for ix, v in s.items()} if s is not None else {}
    d = _cached_json(f"yf_splits_{symbol}.json", call)
    return pd.Series(list(d.values()), index=pd.to_datetime(list(d.keys())), dtype=float)


def build(name="large50") -> pd.DataFrame:
    from universe import panel_path
    path = panel_path(name)
    if not path.exists():
        sys.exit(f"{path.name} not found — run universe.py {name} first")
    panel = pd.read_csv(path, parse_dates=["Date"])
    if "Close_raw" not in panel.columns:
        sys.exit(f"{path.name} has no Close_raw column — rebuild it with universe.py")
    panel = panel.drop(columns=[c for c in FUND_COLS + EARN_COLS if c in panel.columns])
    parts = []
    for i, (sym, g) in enumerate(panel.groupby("symbol"), 1):
        try:
            q, a = fetch_filings(sym)
            fund = fundamentals_for(g, ttm_frame(filings_frame(q, a)), fetch_splits(sym))
        except Exception as exc:
            print(f"  {sym}: filings unavailable ({exc})")
            fund = fundamentals_for(g, pd.DataFrame())
        try:
            earn = earnings_for(g["Date"], fetch_earnings(sym))
        except Exception as exc:
            print(f"  {sym}: earnings unavailable ({exc})")
            earn = earnings_for(g["Date"], None)
        parts.append(fund.merge(earn, on="Date").assign(symbol=sym))
        if i % 10 == 0:
            print(f"  {i}/{panel['symbol'].nunique()} tickers")
    extra = pd.concat(parts, ignore_index=True)
    panel = panel.merge(extra, on=["Date", "symbol"], how="left")
    panel.to_csv(path, index=False)
    cover = panel.groupby("symbol")[["gross_profitability", "earnings_yield", "eps_surprise"]].apply(
        lambda g: g.notna().mean()).mean()
    print(f"wrote {path.name}: coverage " + ", ".join(f"{k} {v:.0%}" for k, v in cover.items()))
    return panel


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "large50")
