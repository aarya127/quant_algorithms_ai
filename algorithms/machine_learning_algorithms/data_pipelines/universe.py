"""
universe.py — a multi-ticker daily panel for cross-sectional research.

    python universe.py [UNIVERSE] [PERIOD]      # defaults: large50, 6y

Writes UNIVERSE_<name>_panel.csv here (gitignored): one row per (Date, symbol)
with split- and dividend-adjusted OHLCV, Close_raw (split-adjusted only, for
market caps — fundamentals_pit.py), the price-only features the single-ticker pipeline builds
(DataTransformer's momentum, risk and microstructure blocks) and the six
canonical targets (targets.py), each computed within its own ticker.

Price-only on purpose: yfinance serves the whole universe in one call, while
the news / fundamentals / options sources behind the 84-column matrix have
daily quotas (Alpha Vantage: 25 calls) that one ticker already strains.

Survivorship bias: the list below was picked in 2026 from today's large caps.
Companies that shrank or failed since 2020 aren't in it, so every strategy
here, buy-and-hold included, looks better than it would have in real time.
Compare strategies with each other and with the equal-weight universe, not
with zero.
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
        if len(df) >= 300:
            factor = df["Adj Close"] / df["Close"]          # dividend adjustment
            df["Close_raw"] = df["Close"]
            for col in ("Open", "High", "Low", "Close"):
                df[col] = df[col] * factor
        if len(df) < 300:
            missing.append(t)
            continue
        df.index.name = "Date"
        out[t] = df
    if missing:
        print(f"no usable history, skipped: {', '.join(missing)}")
    return out


def build(name="large50", period="6y") -> pd.DataFrame:
    tickers = UNIVERSES[name]
    print(f"downloading {len(tickers)} tickers ({period})")
    bars = download(tickers, period)
    panel = to_panel({t: ticker_features(df) for t, df in bars.items()})
    out = HERE / f"UNIVERSE_{name}_panel.csv"
    panel.to_csv(out, index=False)
    print(f"wrote {out.name}: {len(panel):,} rows, {panel['symbol'].nunique()} tickers, "
          f"{panel['Date'].min().date()} → {panel['Date'].max().date()}")
    return panel


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "large50",
          sys.argv[2] if len(sys.argv) > 2 else "6y")
