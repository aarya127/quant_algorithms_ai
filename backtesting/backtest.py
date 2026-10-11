"""
backtest.py — walk-forward trading backtest of the supervised models.

Answers: had we traded TICKER on the model's signal, retraining as we went and
never training on a label we couldn't have known yet, what would we have made
after slippage — against buy-and-hold and a naive momentum rule?

  1. Walk forward from INIT_TRAIN (the first row the supervised pipeline ever
     validates on): every --retrain-every rows, refit on all rows whose
     target_5d label was known, with the pipeline's own fold preprocessing
     (train medians → scaler → Lasso; data.prepare_fold_data) and model ladder.
  2. Each row's forecast → target via the live rule (backend/predictor.py:
     long if predicted 5d return > +1%, short if < −1%, else cash).
  3. Fill at the next open with backend/paper.py's slippage (simulator.py).
  4. Run the rule-based factor strategies (strategies.py) over the same window.

Caveats printed with every run: the feature allow-list
(factor_discovery/output/recommended_features.txt) was picked on the full
history, which flatters every model strategy a little; models use default
hyperparameters (the daily retrain tunes them); production retrains daily.

Usage:
  python backtesting/backtest.py [TICKER] [--model lgb|ridge,...|all]
         [--features-csv PATH] [--retrain-every 21] [--threshold 0.01]
Writes backtesting/output/<TICKER>_backtest_{daily.csv,summary.json}.
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
ML   = ROOT / "algorithms" / "machine_learning_algorithms"
sys.path.insert(0, str(ML / "supervised"))
sys.path.insert(0, str(ROOT / "backend"))

from config import INIT_TRAIN, REGIME_COLS, FEAT_FILE, target_embargo  # noqa: E402
from data import prepare_fold_data                                     # noqa: E402
from models import reg_model_set                                       # noqa: E402
from paper import SLIPPAGE                                             # noqa: E402
from simulator import (simulate, summarize, retrain_blocks,            # noqa: E402
                       signal_from_forecast)
from strategies import FACTOR_STRATEGIES                               # noqa: E402

TARGET     = "target_5d"     # the live signal is driven by predicted_5d_return
THRESHOLD  = 0.01            # backend/predictor.py: ±1% on the 5d forecast
START_CASH = 100_000.0       # same as the model paper account
OUT_DIR    = HERE / "output"


def served_model(ticker: str):
    """The model type the registry is serving for TARGET, or None (not served)."""
    active = ML / "supervised" / "model_registry" / ticker / "active.json"
    try:
        return json.loads(active.read_text())["targets"][TARGET]["model_type"]
    except Exception:
        return None


def walk_forward_forecast(df, features, model_name, every):
    """Out-of-sample TARGET forecasts for rows INIT_TRAIN.. (NaN before)."""
    horizon = target_embargo(TARGET)
    pred = np.full(len(df), np.nan)
    blocks = retrain_blocks(len(df), INIT_TRAIN, every, horizon)
    for k, (train_stop, s, e) in enumerate(blocks):
        train_idx = [i for i in range(train_stop) if not np.isnan(df.at[i, TARGET])]
        X_tr, X_pred, y_tr, _, _, _, _, _ = prepare_fold_data(
            df, train_idx, list(range(s, e)), features, TARGET)
        model = reg_model_set()[model_name]
        model.fit(X_tr, y_tr)
        pred[s:e] = model.predict(X_pred)
        if k % 6 == 0 or k == len(blocks) - 1:
            print(f"    {model_name}: retrain {k+1}/{len(blocks)} "
                  f"(train rows {len(train_idx)}, predicting {df.at[s, 'Date'].date()}"
                  f"..{df.at[e-1, 'Date'].date()})")
    return pred


def forecast_ic(pred, actual):
    """Spearman rank IC of forecasts vs realised TARGET over rows with both."""
    ok = ~(np.isnan(pred) | np.isnan(actual))
    if ok.sum() < 3:
        return float("nan")
    return float(pd.Series(pred[ok]).rank().corr(pd.Series(actual[ok]).rank()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ticker", nargs="?", default="NVDA")
    ap.add_argument("--model", default=None,
                    help="model(s) from supervised/models.py reg_model_set, comma-separated, "
                         "or 'all' (default: the one the registry serves for target_5d, "
                         "else all)")
    ap.add_argument("--features-csv", type=Path, default=None,
                    help="feature matrix (default data_pipelines/<TICKER>_features_with_regimes.csv)")
    ap.add_argument("--retrain-every", type=int, default=21,
                    help="rows between refits (default 21 ≈ monthly)")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    args = ap.parse_args()
    ticker = args.ticker.upper()

    csv = args.features_csv or ML / "data_pipelines" / f"{ticker}_features_with_regimes.csv"
    df = (pd.read_csv(csv, parse_dates=["Date"])
            .sort_values("Date").reset_index(drop=True))
    if len(df) <= INIT_TRAIN + 20:
        sys.exit(f"{csv.name}: {len(df)} rows — need more than {INIT_TRAIN + 20} to backtest")

    allow = [ln.strip() for ln in FEAT_FILE.read_text().splitlines() if ln.strip()]
    # feature set B (allow-list + regime columns), as served. The regime models are
    # fit on rows < INIT_TRAIN only, so they're out of sample for every traded row.
    features = [f for f in allow if f in df.columns]
    features += [c for c in REGIME_COLS if c in df.columns and c not in features]

    ladder = list(reg_model_set())
    served = served_model(ticker)
    models = (args.model.split(",") if args.model and args.model != "all" else
              [served] if not args.model and served else ladder)
    unknown = [m for m in models if m not in ladder]
    if unknown:
        sys.exit(f"unknown model(s) {unknown}; choose from {ladder}")

    bars = df[["Date", "Open", "Close"]]
    traded = slice(INIT_TRAIN, None)     # rows with a decision; fills start a row later
    print(f"=== backtest: {ticker} ===")
    print(f"Rows {len(df)}  |  trading {df.at[INIT_TRAIN, 'Date'].date()} → "
          f"{df['Date'].iloc[-1].date()} ({len(df) - INIT_TRAIN} sessions)")
    print(f"Features {len(features)}  |  retrain every {args.retrain_every}  |  "
          f"signal ±{args.threshold:.2%} on {TARGET}  |  slippage {SLIPPAGE*1e4:.0f} bps")

    # strategies: name → per-row target sign (NaN before the first traded row)
    strategies, ic = {}, {}
    hold = np.full(len(df), np.nan)
    hold[traded] = 1.0
    strategies["buy_and_hold"] = hold

    # naive: assume the next 5 days repeat the last 5 (the gate's momentum baseline)
    past_5d = df["log_return"].rolling(5).sum().to_numpy()
    naive = signal_from_forecast(past_5d, args.threshold)
    naive[:INIT_TRAIN] = np.nan
    strategies["naive_momentum_5d"] = naive

    # rule-based factor strategies (strategies.py), same trading window
    for name, rule in FACTOR_STRATEGIES.items():
        tgt = rule(df)
        tgt[:INIT_TRAIN] = np.nan
        strategies[name] = tgt
    ic["naive_momentum_5d"] = forecast_ic(np.where(np.isnan(naive), np.nan, past_5d),
                                          df[TARGET].to_numpy())

    for m in models:
        pred = walk_forward_forecast(df, features, m, args.retrain_every)
        strategies[f"model_{m}"] = signal_from_forecast(pred, args.threshold)
        ic[f"model_{m}"] = forecast_ic(pred, df[TARGET].to_numpy())

    # every curve starts at START_CASH on the first decision's close; it fills next open
    first = INIT_TRAIN
    out = df.loc[first:, ["Date", "Close"]].reset_index(drop=True)
    summary = {}
    for name, tgt in strategies.items():
        daily, trades = simulate(bars.iloc[first:].reset_index(drop=True), tgt[first:],
                                 START_CASH, SLIPPAGE)
        summary[name] = {**summarize(daily, trades, START_CASH), "forecast_ic": ic.get(name)}
        out[f"{name}_equity"] = daily["equity"].round(2)
        out[f"{name}_shares"] = daily["shares"]

    # report
    cols = [("total_return", "{:+.1%}"), ("cagr", "{:+.1%}"), ("sharpe", "{:.2f}"),
            ("max_drawdown", "{:.1%}"), ("ann_vol", "{:.1%}"), ("n_trades", "{:d}"),
            ("long_frac", "{:.0%}"), ("short_frac", "{:.0%}"), ("forecast_ic", "{:+.3f}")]
    print(f"\n{'strategy':<20}" + "".join(f"{c:>14}" for c, _ in cols))
    for name, s in summary.items():
        cells = [("—" if s[c] is None or (isinstance(s[c], float) and np.isnan(s[c]))
                  else fmt.format(s[c])) for c, fmt in cols]
        print(f"{name:<20}" + "".join(f"{v:>14}" for v in cells))
    print("\nCaveats: feature allow-list chosen on the full history (flatters model_*); "
          "default hyperparameters; no borrow fees or dividends; one ticker, one path.")

    OUT_DIR.mkdir(exist_ok=True)
    out.to_csv(OUT_DIR / f"{ticker}_backtest_daily.csv", index=False)
    (OUT_DIR / f"{ticker}_backtest_summary.json").write_text(json.dumps({
        "ticker": ticker, "target": TARGET, "threshold": args.threshold,
        "retrain_every": args.retrain_every, "slippage": SLIPPAGE,
        "start_cash": START_CASH, "served_model": served, "features": features,
        "strategies": summary,
        # daily equity per strategy, for the app's chart (≈ 750 rows × strategies)
        "equity": {"dates": out["Date"].dt.strftime("%Y-%m-%d").tolist(),
                   **{name: out[f"{name}_equity"].tolist() for name in strategies}},
    }, separators=(",", ":")))
    print(f"Wrote {OUT_DIR.relative_to(ROOT)}/{ticker}_backtest_{{daily.csv,summary.json}}")


if __name__ == "__main__":
    main()
