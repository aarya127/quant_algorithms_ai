"""
factor_backtest.py — cross-sectional factor portfolios on a ticker universe.

    python backtesting/factor_backtest.py [UNIVERSE]      # default large50

Reads data_pipelines/UNIVERSE_<name>_panel.csv (build it with
data_pipelines/universe.py). Each factor ranks every ticker on a date; the
portfolio holds the top fifth, equal-weighted, re-ranked on a fixed schedule,
traded at the next open with backend/paper.py's slippage on the traded value.
Benchmark: all tickers equal-weighted, rebalanced monthly. Textbook parameters,
fixed in advance:

  momentum_12m   — 12-month return skipping the last week (Jegadeesh-Titman), monthly
  low_volatility — lowest 60-day realized vol (Ang et al. 2006; Frazzini-Pedersen), monthly
  reversal_1w    — worst last-week return (Lehmann 1990; Jegadeesh 1990), weekly

Each factor's information coefficient (IC) is the rank correlation between its
score and the next rebalance period's return across tickers, on
non-overlapping dates; t > 2 is the usual bar for "not noise".

Survivorship bias (see universe.py) flatters every row equally; the comparison
that means something is factor vs the equal-weight universe.

Writes backtesting/output/UNIVERSE_<name>_factors.json.
"""
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
from paper import SLIPPAGE                                                         # noqa: E402

PIPES = ROOT / "algorithms" / "machine_learning_algorithms" / "data_pipelines"
OUT_DIR = HERE / "output"
START_CASH = 100_000.0
TOP_FRAC = 0.2

# name → (panel column, sign: +1 = higher is better, rebalance every N sessions)
FACTORS = {
    "momentum_12m":   ("momentum_12m", +1, 21),
    "low_volatility": ("realized_vol_60d", -1, 21),
    "reversal_1w":    ("reversal_1w", -1, 5),
}


def run(panel: pd.DataFrame) -> dict:
    opens, closes = wide(panel, "Open"), wide(panel, "Close")
    scores = {name: wide(panel, col) * sign for name, (col, sign, _) in FACTORS.items()}

    # common window: from the first date every factor can rank enough names
    ready = pd.concat([s.notna().sum(axis=1) >= 10 for s in scores.values()], axis=1).all(axis=1)
    if not ready.any():
        raise ValueError("no date where every factor has scores — is the panel long enough?")
    start = ready.idxmax()
    opens, closes = opens.loc[start:], closes.loc[start:]
    scores = {k: v.loc[start:] for k, v in scores.items()}

    results = {}
    bench_daily, bench_reb = simulate_portfolio(opens, closes, equal_weights(closes, 21),
                                                START_CASH, SLIPPAGE)
    results["equal_weight_universe"] = {**portfolio_stats(bench_daily, bench_reb, START_CASH),
                                        "rebalance_every": 21}
    curves = {"dates": [d.date().isoformat() for d in bench_daily["Date"]],
              "equal_weight_universe": bench_daily["equity"].round(2).tolist()}

    for name, (_, _, every) in FACTORS.items():
        w = top_weights(scores[name], TOP_FRAC, every)
        daily, reb = simulate_portfolio(opens, closes, w, START_CASH, SLIPPAGE)
        ic = ic_summary(rank_ic(scores[name], forward_returns(closes, every), every))
        stats = portfolio_stats(daily, reb, START_CASH)
        results[name] = {**stats, **ic, "rebalance_every": every,
                         "excess_cagr": stats["cagr"] - results["equal_weight_universe"]["cagr"]}
        curves[name] = daily["equity"].round(2).tolist()
    return {"results": results, "equity": curves,
            "tickers": int(panel["symbol"].nunique()), "top_frac": TOP_FRAC,
            "cost_per_side": SLIPPAGE}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "large50"
    path = PIPES / f"UNIVERSE_{name}_panel.csv"
    if not path.exists():
        sys.exit(f"{path.name} not found — run data_pipelines/universe.py {name} first")
    panel = pd.read_csv(path, parse_dates=["Date"])
    out = run(panel)
    res = out["results"]

    first = next(iter(res.values()))
    print(f"=== factor backtest: {name} ({out['tickers']} tickers, top {TOP_FRAC:.0%}, "
          f"{SLIPPAGE * 1e4:.0f} bps per side) {first['start']} → {first['end']} ===")
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
    print("\nSurvivorship-biased universe (today's large caps): compare rows with each other, "
          "not with zero.")

    OUT_DIR.mkdir(exist_ok=True)
    dest = OUT_DIR / f"UNIVERSE_{name}_factors.json"
    dest.write_text(json.dumps({"universe": name, **out}, separators=(",", ":")))
    print(f"Wrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
