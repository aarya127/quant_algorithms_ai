"""
scenario_engine.py — deterministic scenario numbers for the Scenarios /
Recommendations tabs. Numbers come from math; words come from llm_analyst.py.

Two tiers, selected automatically per ticker:

  Tier 1 — statistical (any ticker):
    targets from historical drift, band width from realized volatility scaled
    by sqrt(horizon), probabilities from the ticker's own empirical h-day
    return distribution (not fixed 25/50/25).

  Tier 2 — model-backed (tickers with a trained registry, currently NVDA):
    drift and volatility are overridden by the registry models' predicted
    5-day return / 5-day vol (backend/predictor.py); probabilities still come
    from the empirical distribution recentered on the model drift. Falls back
    to Tier 1 on any error.

All outputs are cached for 15 minutes per (symbol, timeframe).
"""
import math
import os
import sys
import threading
from pathlib import Path

import numpy as np
from cachetools import TTLCache

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from services import yf  # guarded import; None if the data layer failed

_REGISTRY_DIR = (Path(__file__).resolve().parents[1]
                 / 'algorithms' / 'machine_learning_algorithms'
                 / 'supervised' / 'model_registry')

# trading-day horizons for the UI timeframes
HORIZON_DAYS = {'1W': 5, '1M': 21, '3M': 63, '6M': 126, '1Y': 252}

_cache = TTLCache(maxsize=256, ttl=900)
_cache_lock = threading.Lock()


def has_model(symbol: str) -> bool:
    """True when a trained registry exists for this ticker."""
    return (_REGISTRY_DIR / symbol.upper() / 'active.json').exists()


def _history(symbol: str):
    """~2y of daily closes as a numpy array (oldest→newest)."""
    hist = yf.Ticker(symbol).history(period='2y', interval='1d')
    if hist is None or hist.empty or len(hist) < 60:
        raise ValueError(f'insufficient history for {symbol}')
    return hist['Close'].to_numpy(dtype=float)


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
        return float(r5) / 5.0, float(v5) / math.sqrt(5.0), meta
    except Exception as exc:
        print(f'[SCENARIOS] model overlay unavailable for {symbol}: {exc}', flush=True)
        return None


def compute_scenarios(symbol: str, timeframe: str = '1M') -> dict:
    """Bull/base/bear targets + empirical probabilities for one timeframe."""
    symbol = symbol.upper()
    h = HORIZON_DAYS.get(timeframe, 21)
    key = (symbol, h)
    with _cache_lock:
        if key in _cache:
            return _cache[key]

    closes = _history(symbol)
    last = float(closes[-1])
    log_ret = np.diff(np.log(closes))

    # Tier 1: historical drift (shrunk 50% toward zero — daily drift estimates
    # are noisy and momentum-chasing at full weight) and realized vol.
    stat_drift = float(np.mean(log_ret)) * 0.5
    stat_vol = float(np.std(log_ret, ddof=1))
    drift, vol = stat_drift, stat_vol
    total_drift = drift * h
    total_var = vol * vol * h
    meta = {'engine': 'vol-scaled statistical', 'model_backed': False}

    # Tier 2 overlay when a trained registry exists for this ticker. The model
    # predicts a 5-day return/vol — apply it over its NATIVE horizon only and
    # use statistical drift/vol for the remainder, otherwise a 5d prediction
    # extrapolated to 1Y would imply absurd targets.
    if has_model(symbol):
        overlay = _model_overlay(symbol)
        if overlay is not None:
            m_drift, m_vol, meta = overlay
            model_days = min(h, 5)
            rest = h - model_days
            total_drift = m_drift * model_days + stat_drift * rest
            total_var = m_vol ** 2 * model_days + stat_vol ** 2 * rest
            drift = total_drift / h
            if rest:
                meta['engine'] = 'ML registry (5d) + statistical beyond'

    # Targets: geometric drift over the horizon, bands at ±1σ√h
    band = math.sqrt(total_var)
    base = last * math.exp(total_drift)
    bull = last * math.exp(total_drift + band)
    bear = last * math.exp(total_drift - band)

    # Probabilities from the ticker's own h-day return distribution,
    # recentered on the engine drift: P(beyond ±1σ band) and the middle.
    if len(log_ret) > h + 20:
        h_rets = np.array([log_ret[i:i + h].sum() for i in range(len(log_ret) - h)])
        centered = h_rets - float(np.mean(h_rets)) + drift * h
        p_bull = float(np.mean(centered >= band))
        p_bear = float(np.mean(centered <= -band))
        p_up = float(np.mean(centered > 0))
    else:  # thin history — normal-approximation fallback
        p_bull = p_bear = 0.16
        p_up = 0.5 + drift * h / (band * 2.5) if band else 0.5
    p_base = max(0.0, 1.0 - p_bull - p_bear)

    result = {
        'symbol': symbol,
        'timeframe': timeframe,
        'horizon_days': h,
        'current_price': round(last, 2),
        'annualized_vol': round(vol * math.sqrt(252), 4),
        'p_up': round(p_up, 3),
        'scenarios': {
            'bull_case': {'price_target': round(bull, 2), 'probability': round(p_bull * 100, 1)},
            'base_case': {'price_target': round(base, 2), 'probability': round(p_base * 100, 1)},
            'bear_case': {'price_target': round(bear, 2), 'probability': round(p_bear * 100, 1)},
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
    closes = _history(symbol.upper())
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
