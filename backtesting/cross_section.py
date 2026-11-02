"""
cross_section.py — multi-asset portfolio simulation and factor diagnostics.

Pure numpy/pandas (CI-importable), like simulator.py. Inputs are "wide" frames:
one row per session (ascending dates), one column per ticker.

Execution matches the single-asset simulator and the paper account: weights
decided at row t's close are traded at row t+1's open; costs are charged on the
traded value (slippage per side). Between rebalances holdings drift with prices
instead of being reset to target every day, so turnover is what a real
portfolio would trade. A holding whose price series ends (an acquisition, a
delisting) is turned into cash at its last price.
"""
import math

import numpy as np
import pandas as pd

from simulator import TRADING_DAYS, max_drawdown


def wide(panel: pd.DataFrame, col: str) -> pd.DataFrame:
    """Long panel (Date, symbol, col) → Date × symbol frame."""
    return panel.pivot(index="Date", columns="symbol", values=col).sort_index()


def top_weights(scores: pd.DataFrame, top_frac: float = 0.2, every: int = 21,
                min_names: int = 5) -> pd.DataFrame:
    """Equal weights on the best-scored `top_frac` of tickers, decided every
    `every` rows (NaN rows = no decision, keep holdings). Tickers with no score
    that day can't be picked; a date with fewer than `min_names` scored tickers
    makes no decision."""
    w = pd.DataFrame(np.nan, index=scores.index, columns=scores.columns)
    for i in range(0, len(scores), every):
        row = scores.iloc[i].dropna()
        if len(row) < min_names:
            continue
        k = max(1, int(round(len(row) * top_frac)))
        picks = row.nlargest(k).index
        w.iloc[i] = 0.0
        w.loc[scores.index[i], picks] = 1.0 / k
    return w


def eligible_frame(panel: pd.DataFrame, like: pd.DataFrame) -> pd.DataFrame:
    """Date × symbol booleans from the panel's `in_index` (all True without it)."""
    if "in_index" not in panel.columns:
        return pd.DataFrame(True, index=like.index, columns=like.columns)
    return wide(panel, "in_index").reindex(index=like.index, columns=like.columns).fillna(False).astype(bool)


def equal_weights(closes: pd.DataFrame, every: int = 21, eligible: pd.DataFrame = None) -> pd.DataFrame:
    """Equal weight on every (eligible) ticker with a price, rebalanced every `every` rows."""
    w = pd.DataFrame(np.nan, index=closes.index, columns=closes.columns)
    for i in range(0, len(closes), every):
        have = closes.iloc[i].notna()
        if eligible is not None:
            have &= eligible.iloc[i]
        if have.any():
            w.iloc[i] = np.where(have, 1.0 / have.sum(), 0.0)
    return w


def simulate_portfolio(opens: pd.DataFrame, closes: pd.DataFrame, weights: pd.DataFrame,
                       start_cash: float = 100_000.0, cost: float = 0.0005):
    """Run a weight schedule. Returns (daily, rebalances).

    daily      — DataFrame: Date, equity, invested (gross holdings / equity),
                 names (positions held)
    rebalances — list of dicts: date, turnover (traded value / equity), cost
    A ticker with no price on a day doesn't move and can't be traded that day.
    """
    o = opens.to_numpy(dtype=float)
    c = closes.to_numpy(dtype=float)
    w = weights.reindex(index=opens.index, columns=opens.columns).to_numpy(dtype=float)
    dates = opens.index
    n, m = o.shape
    hold = np.zeros(m)                 # $ per ticker, valued at last_px
    cash = float(start_cash)
    last_px = np.where(np.isfinite(c[0]) & (c[0] > 0), c[0], np.nan)
    rows, rebalances = [(dates[0], cash, 0.0, 0)], []
    ok_any = np.isfinite(c) & (c > 0)
    last_row = np.where(ok_any.any(axis=0), n - 1 - np.argmax(ok_any[::-1], axis=0), -1)

    def move_to(px):
        nonlocal hold, last_px
        ok = np.isfinite(px) & (px > 0)
        factor = np.where(ok & np.isfinite(last_px), px / np.where(ok, last_px, 1.0), 1.0)
        hold = hold * factor
        last_px = np.where(ok, px, last_px)
        return ok

    for d in range(1, n):
        ended = (last_row < d) & (hold != 0)              # price series over: cash out
        if ended.any():
            cash += hold[ended].sum()
            hold = np.where(ended, 0.0, hold)
        tradable = move_to(o[d])                         # overnight, to today's open
        target = w[d - 1]                                # decided at yesterday's close
        if np.isfinite(target).any():
            equity = cash + hold.sum()
            want = np.where(tradable, np.nan_to_num(target) * equity, hold)  # no price: stays put
            traded = np.abs(want - hold).sum()
            fee = traded * cost
            cash = equity - want.sum() - fee
            hold = want
            rebalances.append({"date": pd.Timestamp(dates[d]).date().isoformat(),
                               "turnover": traded / equity if equity > 0 else 0.0,
                               "cost": fee})
        move_to(c[d])                                    # intraday, to today's close
        equity = cash + hold.sum()
        rows.append((dates[d], equity, np.abs(hold).sum() / equity if equity > 0 else 0.0,
                     int((hold != 0).sum())))
    daily = pd.DataFrame(rows, columns=["Date", "equity", "invested", "names"])
    return daily, rebalances


def portfolio_stats(daily: pd.DataFrame, rebalances: list, start_cash: float) -> dict:
    eq = daily["equity"].to_numpy(dtype=float)
    rets = np.diff(np.concatenate([[start_cash], eq])) / np.concatenate([[start_cash], eq[:-1]])
    n = len(eq)
    total = eq[-1] / start_cash - 1.0
    vol = float(np.std(rets, ddof=1)) if n > 1 else 0.0
    years = n / TRADING_DAYS
    turn = [r["turnover"] for r in rebalances[1:]]          # the first is the initial buy
    return {
        "start": pd.Timestamp(daily["Date"].iloc[0]).date().isoformat(),
        "end": pd.Timestamp(daily["Date"].iloc[-1]).date().isoformat(),
        "total_return": float(total),
        "cagr": float((1 + total) ** (1 / years) - 1) if years > 0 and total > -1 else -1.0,
        "ann_vol": vol * math.sqrt(TRADING_DAYS),
        "sharpe": float(np.mean(rets) / vol * math.sqrt(TRADING_DAYS)) if vol > 0 else 0.0,
        "max_drawdown": max_drawdown(eq),
        "rebalances": len(rebalances),
        "turnover_per_year": float(sum(turn) / years) if years > 0 else 0.0,
        "costs": float(sum(r["cost"] for r in rebalances)),
    }


def forward_returns(closes: pd.DataFrame, h: int) -> pd.DataFrame:
    """Return from row t's close to row t+h's close (NaN where unknown)."""
    return closes.shift(-h) / closes - 1


def rank_ic(scores: pd.DataFrame, fwd: pd.DataFrame, every: int, min_names: int = 10) -> pd.Series:
    """Spearman rank correlation across tickers between score and forward return,
    on every `every`-th date (non-overlapping when `every` = the return horizon)."""
    out = {}
    for i in range(0, len(scores), every):
        s, r = scores.iloc[i], fwd.iloc[i]
        ok = s.notna() & r.notna()
        if ok.sum() >= min_names:
            out[scores.index[i]] = s[ok].rank().corr(r[ok].rank())
    return pd.Series(out, dtype=float)


def ic_summary(ic: pd.Series) -> dict:
    """Mean IC, its t-statistic (IC samples taken as independent) and the share positive."""
    ic = ic.dropna()
    if len(ic) < 3:
        return {"ic_mean": None, "ic_t": None, "ic_positive": None, "ic_samples": len(ic)}
    sd = float(ic.std(ddof=1))
    return {"ic_mean": float(ic.mean()),
            "ic_t": float(ic.mean() / (sd / math.sqrt(len(ic)))) if sd > 0 else None,
            "ic_positive": float((ic > 0).mean()), "ic_samples": int(len(ic))}
