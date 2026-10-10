"""
simulator.py — daily-bar trading simulator and performance metrics.

Pure numpy/pandas (no model or network deps) so the CI tests can import it.

Fill rules mirror the ML model's paper account (scripts/paper_model_account.py
and backend/paper.py), so a backtest reads like that account's track record:
  - a target (+1 long, -1 short, 0 cash) decided at row t's close fills as a
    market order at row t+1's open, with slippage against the trader
  - the whole position is re-sized from equity at the fill price, in whole
    shares; long → short is one trade
  - no commission, no short-borrow fee, no dividends
"""
import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def simulate(bars: pd.DataFrame, targets, start_cash: float = 100_000.0,
             slippage: float = 0.0005):
    """Run one strategy over daily bars.

    bars    : DataFrame with Date, Open, Close, one row per session, ascending
    targets : per-row target sign decided at that row's close (+1/-1/0);
              NaN = no decision (keep the current position)

    Returns (daily, trades):
      daily  — DataFrame: Date, Close, shares, cash, equity (marked at the close)
      trades — list of dicts: date, side, qty, price, target
    """
    opens  = bars["Open"].to_numpy(dtype=float)
    closes = bars["Close"].to_numpy(dtype=float)
    dates  = pd.to_datetime(bars["Date"]).to_numpy()
    targets = np.asarray(targets, dtype=float)
    if len(targets) != len(bars):
        raise ValueError(f"targets has {len(targets)} rows, bars has {len(bars)}")

    cash, shares, pending = float(start_cash), 0, None
    rows, trades = [], []
    for i in range(len(bars)):
        # 1. fill yesterday's decision at today's open
        if pending is not None:
            held = (shares > 0) - (shares < 0)
            buy = pending > held
            px = opens[i] * (1 + slippage if buy else 1 - slippage)
            equity = cash + shares * px
            new_shares = int(pending) * max(0, math.floor(equity / px))
            dq = new_shares - shares
            if dq:
                cash -= dq * px
                shares = new_shares
                trades.append({"date": pd.Timestamp(dates[i]).date().isoformat(),
                               "side": "buy" if dq > 0 else "sell", "qty": abs(dq),
                               "price": round(px, 4), "target": int(pending)})
            pending = None

        # 2. mark to the close
        rows.append((dates[i], closes[i], shares, cash, cash + shares * closes[i]))

        # 3. today's decision → order for the next open
        t = targets[i]
        if not np.isnan(t):
            held = (shares > 0) - (shares < 0)
            if int(np.sign(t)) != held:
                pending = int(np.sign(t))

    daily = pd.DataFrame(rows, columns=["Date", "Close", "shares", "cash", "equity"])
    return daily, trades


def max_drawdown(equity) -> float:
    """Largest peak-to-trough fall of an equity curve, as a negative fraction."""
    eq = np.asarray(equity, dtype=float)
    if len(eq) == 0:
        return 0.0
    return float(np.min(eq / np.maximum.accumulate(eq) - 1.0))


def summarize(daily: pd.DataFrame, trades: list, start_cash: float) -> dict:
    """Headline metrics for one simulated strategy (risk-free rate taken as 0)."""
    eq = daily["equity"].to_numpy(dtype=float)
    rets = np.diff(np.concatenate([[start_cash], eq])) / np.concatenate([[start_cash], eq[:-1]])
    n = len(eq)
    total = eq[-1] / start_cash - 1.0 if n else 0.0
    vol = float(np.std(rets, ddof=1)) if n > 1 else 0.0
    return {
        "days":          n,
        "start":         pd.Timestamp(daily["Date"].iloc[0]).date().isoformat() if n else None,
        "end":           pd.Timestamp(daily["Date"].iloc[-1]).date().isoformat() if n else None,
        "final_equity":  round(float(eq[-1]), 2) if n else start_cash,
        "total_return":  float(total),
        "cagr":          float((1 + total) ** (TRADING_DAYS / n) - 1) if n and total > -1 else -1.0,
        "ann_vol":       vol * math.sqrt(TRADING_DAYS),
        "sharpe":        float(np.mean(rets) / vol * math.sqrt(TRADING_DAYS)) if vol > 0 else 0.0,
        "max_drawdown":  max_drawdown(eq),
        "n_trades":      len(trades),
        "exposure":      float(np.mean(daily["shares"].to_numpy() != 0)) if n else 0.0,
        "long_frac":     float(np.mean(daily["shares"].to_numpy() > 0)) if n else 0.0,
        "short_frac":    float(np.mean(daily["shares"].to_numpy() < 0)) if n else 0.0,
    }


def retrain_blocks(n_rows: int, start: int, every: int, horizon: int):
    """Walk-forward schedule: (train_stop, pred_start, pred_stop) per retrain.

    A model retrained at row `pred_start` predicts rows pred_start..pred_stop-1,
    each at that row's close. It may train only on rows whose label was known at
    the close of pred_start - 1: a `horizon`-day label on row r uses rows up to
    r + horizon, so training stops at pred_start - horizon (exclusive).
    """
    blocks = []
    for s in range(start, n_rows, every):
        blocks.append((max(0, s - horizon), s, min(s + every, n_rows)))
    return blocks


def signal_from_forecast(pred, threshold: float):
    """Forecast return → target sign: long above +threshold, short below −threshold,
    cash in between; NaN stays NaN (no decision)."""
    p = np.asarray(pred, dtype=float)
    out = np.where(p > threshold, 1.0, np.where(p < -threshold, -1.0, 0.0))
    out[np.isnan(p)] = np.nan
    return out
