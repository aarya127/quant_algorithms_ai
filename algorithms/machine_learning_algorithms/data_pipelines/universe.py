"""
universe.py — a multi-ticker daily panel for cross-sectional research.

    python universe.py [UNIVERSE] [PERIOD]      # large50 | sp500; default period 6y

Writes UNIVERSE_<name>_panel.csv.gz here (gitignored): one row per (Date, symbol)
with split- and dividend-adjusted OHLCV, Close_raw (split-adjusted only, for
market caps — fundamentals_pit.py), the price-only features the single-ticker pipeline builds
(DataTransformer's momentum, risk and microstructure blocks) and the six
canonical targets (targets.py), each computed within its own ticker.

Price-only on purpose: yfinance serves the whole universe in one call, while
the news / fundamentals / options sources behind the 84-column matrix have
daily quotas (Alpha Vantage: 25 calls) that one ticker already strains.

`in_index` marks the dates a row's company belonged to the universe; the
backtests rank and hold only those.

  large50 — a fixed list of 50 large caps picked in 2026, in_index always True.
            Survivorship-biased: companies that shrank or failed since 2020
            aren't in it, so every strategy looks better than it was.
  sp500   — every company that was in the S&P 500 at any time in the window,
            each flagged in_index only while it was a member (membership.py).
            Removes most of that bias; companies yfinance no longer prices
            (mostly acquired or failed) are listed in
            UNIVERSE_sp500_coverage.json with their share of member-days.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

UNIVERSES = {
    # 50 liquid US large caps across sectors, each listed for the whole window
    "large50": [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "BRK-B", "JPM",
        "V", "MA", "UNH", "XOM", "JNJ", "PG", "HD", "COST", "ABBV", "MRK",
        "PEP", "KO", "WMT", "BAC", "CVX", "LLY", "ORCL", "CSCO", "ADBE", "CRM",
        "AMD", "INTC", "QCOM", "TXN", "NFLX", "DIS", "MCD", "NKE", "PFE", "TMO",
        "ABT", "CAT", "GE", "HON", "UPS", "IBM", "AMGN", "GS", "MS", "LIN",
    ],
}

PRICE_COLS = ["Open", "High", "Low", "Close", "Volume"]


def ticker_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Features + targets for one ticker's bars (Date index, PRICE_COLS), using
    only that ticker's rows: rolling windows never reach into another symbol."""
    from algorithms.machine_learning_algorithms.data_pipelines.transform import DataTransformer
    from algorithms.machine_learning_algorithms.data_pipelines.targets import add_targets

    df = bars[PRICE_COLS + [c for c in ("Close_raw",) if c in bars.columns]].astype(float).copy()
    df = df[(df["Open"] > 0) & (df["Close"] > 0)]
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df = DataTransformer._add_momentum_features(df)
    df = DataTransformer._add_risk_features(df)
    df = DataTransformer._add_microstructure_features(df)
    c = df["Close"]
    for n in (20, 50, 200):
        df[f"price_to_sma{n}"] = c / c.rolling(n, min_periods=n).mean()
    return add_targets(df)


def to_panel(frames: dict) -> pd.DataFrame:
    """{symbol: per-ticker frame with a Date index} → long panel sorted by Date, symbol."""
    parts = []
    for sym, df in frames.items():
        out = df.reset_index().rename(columns={df.index.name or "index": "Date"})
        out.insert(1, "symbol", sym)
        parts.append(out)
    panel = pd.concat(parts, ignore_index=True)
    panel["Date"] = pd.to_datetime(panel["Date"]).dt.tz_localize(None)
    return panel.sort_values(["Date", "symbol"]).reset_index(drop=True)


MIN_ROWS = 60      # a short history still counts (a company that joined or left mid-window)


def panel_path(name: str) -> Path:
    return HERE / f"UNIVERSE_{name}_panel.csv.gz"


def load_panel(name: str) -> pd.DataFrame:
    path = panel_path(name)
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found — run data_pipelines/universe.py {name} first")
    return pd.read_csv(path, parse_dates=["Date"])


def download(tickers, period="6y") -> dict:
    """{symbol: daily OHLCV adjusted for splits and dividends, plus Close_raw
    (split-adjusted only)} from one yfinance bulk call; tickers with no data are
    left out (and reported)."""
    import yfinance as yf
    raw = yf.download(tickers, period=period, interval="1d", group_by="ticker",
                      auto_adjust=False, progress=False, threads=True)
    out, missing = {}, []
    for t in tickers:
        try:
            df = raw[t].dropna(subset=["Close", "Adj Close"]).copy()
        except KeyError:
            df = pd.DataFrame()
        if len(df) >= MIN_ROWS:
            factor = df["Adj Close"] / df["Close"]          # dividend adjustment
            df["Close_raw"] = df["Close"]
            for col in ("Open", "High", "Low", "Close"):
                df[col] = df[col] * factor
        if len(df) < MIN_ROWS:
            missing.append(t)
            continue
        df.index.name = "Date"
        out[t] = df
    if missing:
        print(f"no usable history for {len(missing)}: {', '.join(missing)}")
    return out


def _sp500_bars(period):
    """Bars for every S&P 500 member in the window (renamed tickers mapped),
    the membership spells with their price symbol, and the coverage report."""
    import json
    import membership as mb
    spells_all = mb.load_spells()
    window_end = pd.Timestamp.today().normalize()
    years = int(period.rstrip("y")) if period.endswith("y") else 6
    window_start = window_end - pd.DateOffset(years=years)
    spells = mb.spells_in_window(spells_all, window_start, window_end)
    tickers = sorted(spells["ticker"].unique())
    print(f"downloading {len(tickers)} S&P 500 members ({period})")
    bars = download(tickers, period)
    extra = sorted({mb.RENAMES[t] for t in tickers if t not in bars and t in mb.RENAMES} - set(bars))
    if extra:
        print(f"downloading {len(extra)} renamed tickers")
        bars.update(download(extra, period))
    avail = {t: (df.index.min().tz_localize(None), df.index.max().tz_localize(None)) for t, df in bars.items()}
    spells = mb.price_symbols(spells, avail)
    keep = set(spells["price_symbol"].dropna())
    dates = sorted({d.tz_localize(None) for df in bars.values() for d in df.index})
    report = mb.coverage_report(spells, dates)
    report.update(source=mb.SOURCE, window=[str(window_start.date()), str(window_end.date())])
    (HERE / "UNIVERSE_sp500_coverage.json").write_text(json.dumps(report, indent=2))
    print(f"priced {report['priced']}/{report['companies']} members; missing {len(report['missing'])} "
          f"({report['missing_member_day_share']:.1%} of member-days): {', '.join(report['missing'])}")
    return {t: df for t, df in bars.items() if t in keep}, spells


def build(name="large50", period="6y") -> pd.DataFrame:
    if name == "sp500":
        import membership as mb
        bars, spells = _sp500_bars(period)
    else:
        tickers = UNIVERSES[name]
        print(f"downloading {len(tickers)} tickers ({period})")
        bars, spells = download(tickers, period), None
    print(f"computing features for {len(bars)} tickers")
    panel = to_panel({t: ticker_features(df) for t, df in bars.items()})
    if spells is None:
        panel["in_index"] = True
    else:
        flags = mb.in_index(panel["Date"].unique(), spells)
        panel = panel.merge(flags, on=["Date", "symbol"], how="left")
        panel["in_index"] = panel["in_index"].fillna(False).astype(bool)
    out = panel_path(name)
    panel.to_csv(out, index=False)
    print(f"wrote {out.name}: {len(panel):,} rows, {panel['symbol'].nunique()} tickers, "
          f"{panel['Date'].min().date()} → {panel['Date'].max().date()}, "
          f"{int(panel.groupby('Date')['in_index'].sum().median())} members on a typical day")
    return panel


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "large50",
          sys.argv[2] if len(sys.argv) > 2 else "6y")
