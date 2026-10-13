"""
scenario_engine.py — deterministic scenario numbers for the Scenarios /
Recommendations tabs. Numbers come from math; words come from llm_analyst.py.

Filtered historical simulation over 5 years of daily prices:
  - volatility: EWMA (RiskMetrics, λ = 0.94) for today's level, mean-reverting
    to the 5-year average over the horizon (persistence 0.97/day, GARCH-like),
    so a calm or a stressed week isn't extrapolated for a year
  - shape: this ticker's own daily returns standardised by their EWMA vol,
    resampled in 5-day blocks (keeps fat tails and short-run clustering)
  - drift: half the 5-year mean, half a 7%/yr equity prior — trailing returns
    are a weak forecast, and NVDA's at full weight would promise ~60%/yr
  - tickers with a trained registry (backend/predictor.py) use the model's
    5-day return/vol forecast for the first 5 days instead

Bull / base / bear are the 80th / 50th / 20th percentiles of the simulated
price, so each card's probability is true by construction (20% / 60% / 20%).
`calibration` checks the band against history: the share of past h-day moves
that landed inside it (ideal 60%), using only data available at each start.

Outputs are cached for 15 minutes per (symbol, timeframe); prices for 15 minutes
per symbol.
"""
import math
import os
import sys
import threading
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from cachetools import TTLCache

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from services import yf  # guarded import; None if the data layer failed
from common import get_ticker_for_charts

_REGISTRY_DIR = (Path(__file__).resolve().parents[1]
                 / 'algorithms' / 'machine_learning_algorithms'
                 / 'supervised' / 'model_registry')

# trading-day horizons for the UI timeframes
HORIZON_DAYS = {'1W': 5, '1M': 21, '3M': 63, '6M': 126, '1Y': 252}

EWMA_LAMBDA = 0.94        # RiskMetrics daily decay
VOL_PERSISTENCE = 0.97    # per-day pull of today's vol back toward the long-run level
BLOCK = 5                 # bootstrap block length (days)
N_PATHS = 4000
DRIFT_SHRINK = 0.5        # weight on the trailing mean; the rest goes to the prior
PRIOR_ANNUAL = 0.07       # equity-premium-like prior for the drift
MIN_INDEPENDENT = 10      # non-overlapping windows needed to report calibration
BANDS = (0.2, 0.5, 0.8)   # bear / base / bull percentiles

_cache = TTLCache(maxsize=256, ttl=900)
_prices_cache = TTLCache(maxsize=128, ttl=900)
_cache_lock = threading.Lock()


def has_model(symbol: str) -> bool:
    """True when a trained registry exists for this ticker."""
    return (_REGISTRY_DIR / symbol.upper() / 'active.json').exists()


def prices(symbol: str):
    """(~5y daily DataFrame [Date, Open, Close], listing currency). App symbols
    for Canadian listings map to the TSX ticker, matching the Overview tab."""
    symbol = symbol.upper()
    with _cache_lock:
        if symbol in _prices_cache:
            return _prices_cache[symbol]
    tk = yf.Ticker(get_ticker_for_charts(symbol))
    hist = tk.history(period='5y', interval='1d')
    if hist is None or hist.empty or len(hist) < 60:
        raise ValueError(f'insufficient history for {symbol}')
    df = hist[['Open', 'Close']].reset_index().rename(columns={hist.index.name or 'index': 'Date'})
    out = (df, (tk.history_metadata or {}).get('currency', 'USD'))
    with _cache_lock:
        _prices_cache[symbol] = out
    return out


def _history(symbol: str):
    """(daily closes oldest→newest, listing currency)."""
    df, currency = prices(symbol)
    return df['Close'].to_numpy(dtype=float), currency


def ewma_var(log_ret: np.ndarray) -> np.ndarray:
    """σ² for each day, from returns strictly before it (seeded with the first
    20 days' variance): element t is the forecast for return t; the last+1
    element is the forecast for tomorrow."""
    var = np.empty(len(log_ret) + 1)
    var[0] = np.var(log_ret[:20], ddof=1)
    for t, r in enumerate(log_ret):
        var[t + 1] = EWMA_LAMBDA * var[t] + (1 - EWMA_LAMBDA) * r * r
    return var


def var_path(var_now: float, var_lr: float, h: int) -> np.ndarray:
    """Daily variance for days 1..h, decaying from today's EWMA level to the long run."""
    return var_lr + (var_now - var_lr) * VOL_PERSISTENCE ** np.arange(1, h + 1)


def _bootstrap_sums(z: np.ndarray, sig: np.ndarray, rng) -> np.ndarray:
    """Simulated sums of sig_k · z over len(sig) days, z resampled in blocks."""
    h = len(sig)
    n_blocks = -(-h // BLOCK)
    starts = rng.integers(0, len(z), size=(N_PATHS, n_blocks))
    idx = (starts[:, :, None] + np.arange(BLOCK)) % len(z)          # circular blocks
    draws = z[idx.reshape(N_PATHS, -1)[:, :h]]
    return (draws * sig).sum(axis=1)


def _model_overlay(symbol: str):
    """Return (daily_drift, daily_vol, meta) from the registry, or None."""
    try:
        from predictor import predict_latest
        result = predict_latest(symbol.upper())
        pred = result.get('predictions') or {}
        r5 = pred.get('predicted_5d_return')
        v5 = pred.get('predicted_vol_5d')
        if r5 is None or v5 is None:
            return None
        meta = {
            'engine': 'ML registry',
            'model_backed': True,
            'predicted_5d_return': round(float(r5), 5),
            'predicted_vol_5d': round(float(v5), 5),
            'predicted_dir_1d': pred.get('predicted_dir_1d'),
            'model_signal': result.get('signal'),
            'prediction_date': result.get('date'),
        }
        # target_vol_5d is the std of DAILY returns over the next 5 days, so it
        # is already a daily vol
        return float(r5) / 5.0 * DRIFT_SHRINK, float(v5), meta
    except Exception as exc:
        print(f'[SCENARIOS] model overlay unavailable for {symbol}: {exc}', flush=True)
        return None


def _calibration(log_ret, var, z, mu, var_lr, h, rng):
    """Share of past h-day moves inside the 20–80% band, each band built from the
    EWMA vol known at its start (the return shape uses the full sample)."""
    starts = np.arange(252, len(log_ret) - h + 1)
    if len(log_ret) // h < MIN_INDEPENDENT or len(starts) == 0:
        return None
    unit = _bootstrap_sums(z, np.ones(h), rng) / math.sqrt(h)       # unit-variance shape
    q_lo, q_hi = np.quantile(unit, [BANDS[0], BANDS[2]])
    k = VOL_PERSISTENCE ** np.arange(1, h + 1)
    total_var = h * var_lr + (var[starts] - var_lr) * k.sum()
    csum = np.concatenate([[0.0], np.cumsum(log_ret)])
    realised = csum[starts + h] - csum[starts]
    centre = mu * h
    inside = (realised >= centre + q_lo * np.sqrt(total_var)) & \
             (realised <= centre + q_hi * np.sqrt(total_var))
    return {'inside_band': round(float(inside.mean()), 3), 'ideal': BANDS[2] - BANDS[0],
            'windows': int(len(starts)), 'independent_windows': int(len(log_ret) // h)}


def compute_scenarios(symbol: str, timeframe: str = '1M') -> dict:
    """Bull/base/bear targets (80th/50th/20th percentiles) for one timeframe."""
    symbol = symbol.upper()
    h = HORIZON_DAYS.get(timeframe, 21)
    key = (symbol, h)
    with _cache_lock:
        if key in _cache:
            return _cache[key]

    closes, currency = _history(symbol)
    last = float(closes[-1])
    log_ret = np.diff(np.log(closes))
    var = ewma_var(log_ret)
    z = log_ret / np.sqrt(var[:-1])                 # standardised by the vol known that day
    z = (z - z.mean()) / z.std(ddof=1)
    var_lr = float(np.var(log_ret, ddof=1))
    mu = DRIFT_SHRINK * float(np.mean(log_ret)) + (1 - DRIFT_SHRINK) * math.log(1 + PRIOR_ANNUAL) / 252

    mus = np.full(h, mu)
    sig = np.sqrt(var_path(float(var[-1]), var_lr, h))
    meta = {'engine': 'filtered historical simulation', 'model_backed': False}
    if has_model(symbol):
        overlay = _model_overlay(symbol)
        if overlay is not None:
            m_drift, m_vol, meta = overlay
            n = min(h, 5)
            mus[:n], sig[:n] = m_drift, m_vol
            meta['engine'] = 'ML registry (5d)' + (' + simulation beyond' if h > 5 else '')

    rng = np.random.default_rng(zlib.crc32(f'{symbol}:{h}'.encode()))   # same answer on refresh
    totals = mus.sum() + _bootstrap_sums(z, sig, rng)
    pct = np.quantile(totals, [0.05, *BANDS, 0.95])
    targets = last * np.exp(pct)

    result = {
        'symbol': symbol,
        'timeframe': timeframe,
        'horizon_days': h,
        'current_price': round(last, 2),
        'currency': currency,
        'annualized_vol': round(float(np.sqrt(np.mean(sig ** 2) * 252)), 4),
        'vol_now_annualized': round(float(np.sqrt(var[-1] * 252)), 4),
        'vol_longrun_annualized': round(math.sqrt(var_lr * 252), 4),
        'drift_annualized': round(math.exp(mu * 252) - 1, 4),
        'p_up': round(float(np.mean(totals > 0)), 3),
        'p_loss_10': round(float(np.mean(totals < math.log(0.9))), 3),
        'range_90': {'low': round(float(targets[0]), 2), 'high': round(float(targets[-1]), 2)},
        'probability_method': f'{N_PATHS:,} simulated paths from {len(log_ret):,} days of history',
        'calibration': _calibration(log_ret, var, z, mu, var_lr, h, rng),
        'scenarios': {
            'bull_case': {'price_target': round(float(targets[3]), 2), 'probability': 20.0},
            'base_case': {'price_target': round(float(targets[2]), 2), 'probability': 60.0},
            'bear_case': {'price_target': round(float(targets[1]), 2), 'probability': 20.0},
        },
        **meta,
    }
    for sc in result['scenarios'].values():
        sc['return'] = round((sc['price_target'] - last) / last * 100, 2)

    with _cache_lock:
        _cache[key] = result
    return result


def market_snapshot(symbol: str) -> dict:
    """Compact technical/price context for the LLM payload (all computed)."""
    closes, _ = _history(symbol.upper())
    last = float(closes[-1])

    def chg(n):
        return round((last / float(closes[-min(n + 1, len(closes))]) - 1) * 100, 2)

    log_ret = np.diff(np.log(closes))
    vol20 = float(np.std(log_ret[-20:], ddof=1)) * math.sqrt(252)
    # percentile of current 20d vol within the trailing year of rolling 20d vols
    windows = [np.std(log_ret[i:i + 20], ddof=1) for i in range(max(0, len(log_ret) - 272), len(log_ret) - 19)]
    vol_pct = round(float(np.mean([w <= np.std(log_ret[-20:], ddof=1) for w in windows])) * 100) if windows else None

    # RSI(14)
    delta = np.diff(closes)[-60:]
    gains = np.clip(delta, 0, None)
    losses = np.clip(-delta, 0, None)
    avg_g, avg_l = np.mean(gains[-14:]), np.mean(losses[-14:])
    rsi = round(100 - 100 / (1 + avg_g / avg_l), 1) if avg_l > 0 else 100.0

    lo, hi = float(np.min(closes[-252:])), float(np.max(closes[-252:]))
    sma50 = float(np.mean(closes[-50:]))
    sma200 = float(np.mean(closes[-200:])) if len(closes) >= 200 else None

    return {
        'last': round(last, 2),
        'chg_1w_pct': chg(5), 'chg_1m_pct': chg(21), 'chg_3m_pct': chg(63),
        'pos_in_52w_range_pct': round((last - lo) / (hi - lo) * 100) if hi > lo else None,
        'rsi14': rsi,
        'price_vs_sma50_pct': round((last / sma50 - 1) * 100, 2),
        'price_vs_sma200_pct': round((last / sma200 - 1) * 100, 2) if sma200 else None,
        'realized_vol_20d_ann': round(vol20, 3),
        'vol_1y_percentile': vol_pct,
    }
