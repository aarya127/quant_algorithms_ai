"""
strategies.py — rule-based factor strategies for the backtest.

Each takes the feature matrix (one row per session, features known at that
row's close) and returns a per-row target for simulator.simulate(): +1 long,
-1 short, 0 cash, NaN = keep the current position. Parameters are the textbook
ones, fixed in advance — tuning them on the backtest window would just fit noise.

Time-series rules on one ticker only: a cross-sectional factor (rank a universe,
buy the top) needs several tickers, and a quality factor needs point-in-time
fundamentals (both on the roadmap).
"""
import numpy as np
import pandas as pd


def trend_sma200(df: pd.DataFrame) -> np.ndarray:
    """Long above the 200-day average, cash below (Faber 2007)."""
    x = df["price_to_sma200"].to_numpy(dtype=float)
    out = np.where(x > 1.0, 1.0, 0.0)
    out[np.isnan(x)] = np.nan
    return out


def tsmom_12m(df: pd.DataFrame) -> np.ndarray:
    """Time-series momentum: long if the 12-month return (skipping the last week)
    is positive, short if negative (Moskowitz, Ooi & Pedersen 2012)."""
    x = df["momentum_12m"].to_numpy(dtype=float)
    out = np.sign(x)
    out[np.isnan(x)] = np.nan
    return out


def rsi_mean_reversion(df: pd.DataFrame, enter: float = 30.0, exit_: float = 50.0) -> np.ndarray:
    """Buy when RSI(14) is oversold (< enter), go to cash once it recovers
    (> exit_), hold in between."""
    rsi = df["RSI_14"].to_numpy(dtype=float)
    out = np.full(len(rsi), np.nan)
    out[rsi < enter] = 1.0
    out[rsi > exit_] = 0.0
    return out


FACTOR_STRATEGIES = {
    "trend_sma200":       trend_sma200,
    "tsmom_12m":          tsmom_12m,
    "rsi_mean_reversion": rsi_mean_reversion,
}
