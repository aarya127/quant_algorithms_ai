"""
recommendation_engine.py — the Recommendations tab: a transparent score from
signals that can be checked against this ticker's own history.

Votes, each in [-1, +1]:
  price     — the mean of two rules that both read the past year's trend, so
              they count once: trend (close above / below its 200-day average,
              Faber 2007) and 12-month momentum skipping the last week
              (Moskowitz-Ooi-Pedersen 2012). +1 / -1 when they agree, 0 when not.
  model     — the ML registry's 5-day return forecast beyond ±1%, only where a
              model is served (it has passed the CV-stability gate); 1W and 1M
  quality   — the Metrics & Grades overall grade (vs industry peers):
              A/B +1, C 0, D/F -1; 3M and longer, with a weight that grows with
              the horizon. Fundamentals aren't point-in-time here, so this one
              can't be backtested.

Per horizon, score = weighted mean of the available votes; Buy above +0.3,
Sell below -0.3, else Hold. With every vote present, quality can turn a 3M Buy
into a Hold and decides 1Y when price signals are mixed. Nothing here is "Strong": the evidence table shows
how each price rule has actually done on this ticker — decide at the close,
fill the next open with backend/paper.py's slippage, via
backtesting/simulator.py — next to buy-and-hold. The rules there are long or
cash (a Sell means "be out", not "short").
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backtesting'))
from simulator import simulate, summarize   # noqa: E402

import scenario_engine                       # noqa: E402
from paper import SLIPPAGE                   # noqa: E402

THRESHOLD = 0.3                 # |score| needed for Buy / Sell
MODEL_THRESHOLD = 0.01          # same ±1% rule as backend/predictor.py
WARMUP = 257                    # rows before the 12-month momentum exists

WEIGHTS = {                     # horizon → signal weights (missing signals drop out)
    '1W': {'model': 0.6, 'price': 0.4},
    '1M': {'model': 0.5, 'price': 0.5},
    '3M': {'price': 0.6, 'quality': 0.4},
    '6M': {'price': 0.5, 'quality': 0.5},
    '1Y': {'price': 0.4, 'quality': 0.6},
}
LABELS = {'1W': 'Short-term trading', '1M': 'Swing trading', '3M': 'Medium term',
          '6M': 'Position', '1Y': 'Long term'}


def price_signals(closes: np.ndarray) -> dict:
    """Per-day votes (NaN during warm-up) for the price rules."""
    n = len(closes)
    sma200 = np.full(n, np.nan)
    c = np.cumsum(np.insert(closes, 0, 0.0))
    sma200[199:] = (c[200:] - c[:-200]) / 200
    trend = np.where(np.isnan(sma200), np.nan, np.where(closes > sma200, 1.0, -1.0))
    mom = np.full(n, np.nan)
    mom[WARMUP:] = np.sign(closes[WARMUP - 5:-5] / closes[:n - WARMUP] - 1)
    return {'trend': trend, 'momentum': mom}


def _long_or_cash(votes: np.ndarray) -> np.ndarray:
    out = np.where(votes > 0, 1.0, 0.0)
    out[np.isnan(votes)] = np.nan
    return out


def horizon_action(votes: dict, weights: dict):
    """(action, score, used weights) for one horizon from {signal: vote}."""
    used = {k: w for k, w in weights.items() if k in votes}
    total = sum(used.values())
    score = sum(votes[k] * w for k, w in used.items()) / total
    action = 'Buy' if score > THRESHOLD else 'Sell' if score < -THRESHOLD else 'Hold'
    return action, score, {k: w / total for k, w in used.items()}


def _evidence_row(name, daily, trades, start_cash):
    s = summarize(daily, trades, start_cash)
    return {'rule': name, 'cagr': round(s['cagr'], 4), 'sharpe': round(s['sharpe'], 2),
            'max_drawdown': round(s['max_drawdown'], 4), 'invested': round(s['exposure'], 2),
            'trades': s['n_trades']}


def evidence(bars, signals) -> dict:
    """Backtest each price rule (long or cash) and their equal-weight blend
    against buy-and-hold over the window where every rule is defined."""
    start = WARMUP
    window = bars.iloc[start:].reset_index(drop=True)
    rows = []
    hold = np.ones(len(window))
    rows.append(_evidence_row('Buy and hold', *simulate(window, hold, slippage=SLIPPAGE, fractional=True), 100_000.0))
    names = {'trend': 'Trend (200-day average)', 'momentum': '12-month momentum'}
    for key, label in names.items():
        tgt = _long_or_cash(signals[key][start:])
        rows.append(_evidence_row(label, *simulate(window, tgt, slippage=SLIPPAGE, fractional=True), 100_000.0))
    blend = (signals['trend'][start:] + signals['momentum'][start:]) / 2
    tgt = np.where(blend > THRESHOLD, 1.0, 0.0)
    rows.append(_evidence_row('Trend + momentum together',
                              *simulate(window, tgt, slippage=SLIPPAGE, fractional=True), 100_000.0))
    return {'start': str(window['Date'].iloc[0])[:10], 'end': str(window['Date'].iloc[-1])[:10],
            'rows': rows}


def _model_vote(symbol, last_date):
    """(vote, detail, backtest row) from the served 5-day model, or None — also
    None when the forecast is more than MAX_MODEL_AGE sessions older than the
    last close (a failed retrain must not keep voting with last week's view)."""
    if not scenario_engine.has_model(symbol):
        return None
    try:
        from predictor import predict_latest
        pred = predict_latest(symbol)
    except Exception:
        return None
    r5 = (pred.get('predictions') or {}).get('predicted_5d_return')
    if r5 is None:                    # not served (e.g. failed the stability gate)
        return None
    age = scenario_engine.forecast_age(pred.get('date'), last_date)
    if age is None or age > scenario_engine.MAX_MODEL_AGE:
        return None
    vote = 1.0 if r5 > MODEL_THRESHOLD else -1.0 if r5 < -MODEL_THRESHOLD else 0.0
    row = None
    summary = ROOT / 'backtesting' / 'output' / f'{symbol}_backtest_summary.json'
    try:
        bt = json.loads(summary.read_text())
        s = bt['strategies'][f"model_{bt['served_model']}"]
        row = {'rule': f"ML model ({bt['served_model']}, long/short/cash)",
               'cagr': round(s['cagr'], 4), 'sharpe': round(s['sharpe'], 2),
               'max_drawdown': round(s['max_drawdown'], 4),
               'invested': round(s['long_frac'] + s['short_frac'], 2),
               'trades': s['n_trades'], 'window': f"{s['start']} → {s['end']}"}
    except Exception:
        pass
    return vote, f"forecast {r5:+.1%} over 5 days ({pred.get('date')})", row


def _quality_vote(symbol):
    try:
        from services import graded_fundamentals
        m, industry, _ = graded_fundamentals(symbol)
    except Exception:
        return None
    g = m.get('overall_grade')
    if not g:
        return None
    return {'A': 1.0, 'B': 1.0, 'C': 0.0}.get(g, -1.0), f"overall grade {g} vs {industry or 'industry'} peers"


def recommend(symbol: str) -> dict:
    symbol = symbol.upper()
    bars, currency = scenario_engine.prices(symbol)
    closes = bars['Close'].to_numpy(dtype=float)
    if len(closes) <= WARMUP + 60:
        raise ValueError(f'{symbol}: needs over a year of price history for these signals')
    sig = price_signals(closes)

    last_close = closes[-1]
    sma200 = closes[-200:].mean()
    r12 = closes[-6] / closes[-WARMUP - 1] - 1
    trend, mom = float(sig['trend'][-1]), float(sig['momentum'][-1])
    votes = {
        'price': ((trend + mom) / 2,
                  f"trend {'up' if trend > 0 else 'down'} (price {last_close / sma200 - 1:+.1%} vs "
                  f"200-day average), 12-month momentum {'up' if mom > 0 else 'down' if mom < 0 else 'flat'} "
                  f"({r12:+.1%}, skipping last week)"),
    }
    ev = evidence(bars, sig)
    model = _model_vote(symbol, bars['Date'].iloc[-1])
    if model:
        votes['model'] = model[:2]
        if model[2]:
            ev['rows'].append(model[2])
    quality = _quality_vote(symbol)
    if quality:
        votes['quality'] = quality

    horizons = {}
    for tf, weights in WEIGHTS.items():
        action, score, used = horizon_action({k: v[0] for k, v in votes.items()}, weights)
        sc = scenario_engine.compute_scenarios(symbol, tf)
        horizons[tf] = {
            'label': LABELS[tf],
            'action': action,
            'score': round(score, 2),
            'votes': [{'signal': k, 'vote': votes[k][0], 'weight': round(w, 2),
                       'detail': votes[k][1]} for k, w in used.items()],
            'p_up': sc['p_up'],
            'base_return': sc['scenarios']['base_case']['return'],
        }
    return {'symbol': symbol, 'currency': currency, 'horizons': horizons, 'evidence': ev,
            'threshold': THRESHOLD}
