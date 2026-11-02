"""
pooled_backtest.py — one model trained across the whole universe, judged as a
stock ranker.

    python backtesting/pooled_backtest.py [UNIVERSE] [--models ridge,lgb]

The single-ticker models see ~1,250 rows; pooling 50 tickers gives ~60,000.
What's learned is relative: the target is each ticker's next-month (21-session)
return ranked against the other tickers that day, and every feature is ranked
across tickers the same way, so a market-wide move — which a stock ranker can't
and needn't time — drops out, and NVDA's volatility and Walmart's sit on one
scale.

Walk-forward: refit monthly on every (date, ticker) row whose label was known
at the refit (label horizon = purge gap, simulator.retrain_blocks), the first
refit after MIN_TRAIN_DATES dates of features. At each refit the model scores
every ticker; the portfolio holds the top fifth until the next refit, traded at
the next open with slippage (cross_section.simulate_portfolio).

Compared on the same out-of-sample window with the equal-weight universe and
the plain factors from factor_backtest.py. Model hyperparameters are the
pipeline's defaults (supervised/models.py), untuned. The universe carries
survivorship bias (universe.py): compare rows, not absolute returns.

Writes backtesting/output/UNIVERSE_<name>_pooled.json.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "backend"))

from cross_section import (wide, top_weights, equal_weights, simulate_portfolio,  # noqa: E402
                           portfolio_stats, forward_returns, rank_ic, ic_summary)
from simulator import retrain_blocks                                               # noqa: E402

PIPES = ROOT / "algorithms" / "machine_learning_algorithms" / "data_pipelines"
OUT_DIR = HERE / "output"
START_CASH = 100_000.0
HORIZON = 21              # predict and rebalance monthly
TOP_FRAC = 0.2
MIN_TRAIN_DATES = 504     # two years of feature dates before the first refit
MIN_NAMES = 10            # tickers needed to rank a date

FEATURES = [
    "reversal_1w", "momentum_1m", "momentum_3m", "momentum_6m", "momentum_12m",
    "high_52w_pct", "low_52w_pct", "price_accel",
    "realized_vol_20d", "realized_vol_60d", "vol_of_vol", "skewness_20d", "kurtosis_20d",
    "downside_deviation", "max_drawdown_20d",
    "amihud_illiquidity", "bid_ask_spread_proxy", "volume_participation", "trade_imbalance",
    "price_to_sma20", "price_to_sma50", "price_to_sma200",
]


def cs_rank(panel: pd.DataFrame, cols) -> pd.DataFrame:
    """Each column ranked across tickers within its date, scaled to [-0.5, 0.5]
    (NaN stays NaN). Uses only that date's rows: no look-ahead."""
    g = panel.groupby("Date")[list(cols)]
    return g.rank(pct=True) - 0.5


def add_target(panel: pd.DataFrame, h: int = HORIZON) -> pd.DataFrame:
    """`fwd` = each ticker's close-to-close log return over the next h sessions;
    `y` = fwd ranked across tickers within the date ([-0.5, 0.5])."""
    out = panel.sort_values(["symbol", "Date"]).copy()
    out["fwd"] = np.log(out.groupby("symbol")["Close"].shift(-h) / out["Close"])
    out = out.sort_values(["Date", "symbol"]).reset_index(drop=True)
    out["y"] = out.groupby("Date")["fwd"].rank(pct=True) - 0.5
    return out


def prepare(panel: pd.DataFrame, h: int = HORIZON):
    """(rows, X, dates) — rows: Date, symbol, date_pos, y; X: ranked features
    with missing values at the cross-sectional median (0); dates: sorted."""
    df = add_target(panel, h)
    X = cs_rank(df, FEATURES).fillna(0.0).to_numpy(dtype=float)
    dates = np.sort(df["Date"].unique())
    pos = {d: i for i, d in enumerate(dates)}
    rows = df[["Date", "symbol", "y"]].copy()
    rows["date_pos"] = rows["Date"].map(pos).to_numpy()
    # a date where too few tickers have features can't be ranked meaningfully
    have = df[FEATURES].notna().sum(axis=1) >= len(FEATURES) // 2
    rows["usable"] = have.groupby(df["Date"]).transform("sum").to_numpy() >= MIN_NAMES
    return rows, X, dates


def first_refit(rows: pd.DataFrame) -> int:
    """Date position of the first refit: MIN_TRAIN_DATES usable dates in."""
    usable = np.sort(rows.loc[rows["usable"], "date_pos"].unique())
    if len(usable) <= MIN_TRAIN_DATES:
        raise ValueError(f"only {len(usable)} usable dates — need more than {MIN_TRAIN_DATES}")
    return int(usable[MIN_TRAIN_DATES])


def walk_forward_scores(rows, X, dates, make_model, start: int, h: int = HORIZON) -> pd.DataFrame:
    """Date × symbol scores at each refit date (NaN elsewhere). A model refit at
    date position s trains on rows with date_pos < s - h (labels complete by the
    close before s) and scores the tickers at s."""
    scores = pd.DataFrame(np.nan, index=pd.DatetimeIndex(dates),
                          columns=sorted(rows["symbol"].unique()))
    dpos = rows["date_pos"].to_numpy()
    y = rows["y"].to_numpy()
    usable = rows["usable"].to_numpy()
    for train_stop, s, _ in retrain_blocks(len(dates), start, h, h):
        tr = (dpos < train_stop) & usable & np.isfinite(y)
        now = (dpos == s) & usable
        if tr.sum() < 1000 or now.sum() < MIN_NAMES:
            continue
        model = make_model()
        model.fit(X[tr], y[tr])
        scores.loc[pd.Timestamp(dates[s]), rows.loc[now, "symbol"].to_numpy()] = model.predict(X[now])
    return scores


def evaluate(scores, opens, closes, every):
    """Top-fifth portfolio + IC for one score frame on the given window."""
    daily, reb = simulate_portfolio(opens, closes, top_weights(scores, TOP_FRAC, every),
                                    START_CASH, cost=_slippage())
    ic = ic_summary(rank_ic(scores, forward_returns(closes, every), every))
    return {**portfolio_stats(daily, reb, START_CASH), **ic}, daily


def _slippage():
    from paper import SLIPPAGE
    return SLIPPAGE


def run(panel: pd.DataFrame, model_names) -> dict:
    sys.path.insert(0, str(ROOT / "algorithms" / "machine_learning_algorithms" / "supervised"))
    from models import reg_model_set
    from factor_backtest import FACTORS

    rows, X, dates = prepare(panel)
    start = first_refit(rows)
    oos = pd.DatetimeIndex(dates[start:])
    opens, closes = wide(panel, "Open").loc[oos], wide(panel, "Close").loc[oos]

    results, curves = {}, {"dates": [d.date().isoformat() for d in oos]}
    bench, breb = simulate_portfolio(opens, closes, equal_weights(closes, HORIZON), START_CASH, _slippage())
    results["equal_weight_universe"] = portfolio_stats(bench, breb, START_CASH)
    curves["equal_weight_universe"] = bench["equity"].round(2).tolist()

    for name in ("momentum_12m", "low_volatility"):
        col, sign, every = FACTORS[name]
        stats, daily = evaluate(wide(panel, col).loc[oos] * sign, opens, closes, every)
        results[name] = stats
        curves[name] = daily["equity"].round(2).tolist()

    for m in model_names:
        print(f"  fitting pooled {m} …", flush=True)
        scores = walk_forward_scores(rows, X, dates, lambda: reg_model_set()[m], start)
        stats, daily = evaluate(scores.loc[oos], opens, closes, HORIZON)
        results[f"pooled_{m}"] = stats
        curves[f"pooled_{m}"] = daily["equity"].round(2).tolist()

    ew = results["equal_weight_universe"]["cagr"]
    for r in results.values():
        r["excess_cagr"] = r["cagr"] - ew
    return {"results": results, "equity": curves, "tickers": int(panel["symbol"].nunique()),
            "features": FEATURES, "horizon": HORIZON, "top_frac": TOP_FRAC,
            "train_rows_at_last_refit": int(((rows["date_pos"] < len(dates) - 2 * HORIZON)
                                              & rows["usable"] & rows["y"].notna()).sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("universe", nargs="?", default="large50")
    ap.add_argument("--models", default="ridge,lgb")
    args = ap.parse_args()
    path = PIPES / f"UNIVERSE_{args.universe}_panel.csv"
    if not path.exists():
        sys.exit(f"{path.name} not found — run data_pipelines/universe.py {args.universe} first")
    panel = pd.read_csv(path, parse_dates=["Date"])
    out = run(panel, [m.strip() for m in args.models.split(",") if m.strip()])
    res = out["results"]

    first = res["equal_weight_universe"]
    print(f"\n=== pooled model: {args.universe} ({out['tickers']} tickers, {len(FEATURES)} ranked "
          f"features, ~{out['train_rows_at_last_refit']:,} training rows by the last refit) "
          f"out of sample {first['start']} → {first['end']} ===")
    cols = [("cagr", "{:+.1%}"), ("excess_cagr", "{:+.1%}"), ("sharpe", "{:.2f}"),
            ("max_drawdown", "{:.1%}"), ("turnover_per_year", "{:.1f}x"),
            ("ic_mean", "{:+.3f}"), ("ic_t", "{:+.2f}"), ("ic_samples", "{:d}")]
    print(f"{'portfolio':<24}" + "".join(f"{c:>18}" for c, _ in cols))
    for k, s in res.items():
        cells = []
        for c, fmt in cols:
            v = s.get(c)
            cells.append("—" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v))
        print(f"{k:<24}" + "".join(f"{v:>18}" for v in cells))
    print("\nSame out-of-sample window for every row; untuned models; survivorship-biased universe.")

    OUT_DIR.mkdir(exist_ok=True)
    dest = OUT_DIR / f"UNIVERSE_{args.universe}_pooled.json"
    dest.write_text(json.dumps({"universe": args.universe, **out}, separators=(",", ":")))
    print(f"Wrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
