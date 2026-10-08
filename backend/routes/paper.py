"""
routes/paper.py — paper trading.

Portfolios live in the visitor's browser (Render's free tier has no durable
disk); the server only prices quotes and decides fills, using paper.py's rules.

GET  /api/paper/quote?symbols=NVDA,AAPL  → latest prices
POST /api/paper/fill  {orders: [{id, symbol, side, qty, type, limit?, placed_at}]}
                      → {results: [{id, status: filled|open|rejected, price?, filled_at?, reason?}]}
GET  /api/paper/model?ticker=NVDA → the ML model's paper account (traded by the daily retrain)
"""
import json
import re
from pathlib import Path

import pandas as pd
from flask import Blueprint, jsonify, request

import model_sync
import paper

bp = Blueprint('paper', __name__)

_LEDGERS = Path(__file__).resolve().parents[2] / 'paper'
_MAX_ORDERS = 50


def _symbol(raw):
    s = str(raw or '').strip().upper()
    return s if re.fullmatch(r'[A-Z0-9][A-Z0-9.\-]{0,9}', s) else None


def _check(o):
    """Normalised order, or an error string."""
    sym = _symbol(o.get('symbol'))
    if not sym:
        return 'invalid symbol'
    if o.get('side') not in ('buy', 'sell') or o.get('type') not in ('market', 'limit'):
        return 'invalid side or type'
    try:
        qty = float(o.get('qty'))
        placed = pd.Timestamp(o.get('placed_at'))
        limit = float(o['limit']) if o.get('type') == 'limit' else None
    except Exception:
        return 'invalid qty, limit or placed_at'
    if not (0 < qty <= 1e6) or (limit is not None and limit <= 0):
        return 'qty or limit out of range'
    placed = placed.tz_localize('UTC') if placed.tzinfo is None else placed
    if placed > pd.Timestamp.now(tz='UTC') + pd.Timedelta(minutes=1):
        return 'placed_at is in the future'
    return {'symbol': sym, 'side': o['side'], 'type': o['type'], 'qty': qty,
            'limit': limit, 'placed_at': placed}


@bp.route('/api/paper/quote')
def quote():
    syms = [s for s in (_symbol(x) for x in request.args.get('symbols', '').split(',')) if s]
    out = {}
    for sym in syms[:25]:
        try:
            out[sym] = paper.quote(sym)
        except Exception as e:
            out[sym] = {'error': f'no price for {sym}: {type(e).__name__}'}
    return jsonify({'success': True, 'quotes': out})


@bp.route('/api/paper/fill', methods=['POST'])
def fill():
    orders = (request.get_json(silent=True) or {}).get('orders') or []
    if not isinstance(orders, list) or len(orders) > _MAX_ORDERS:
        return jsonify({'success': False, 'error': f'send a list of at most {_MAX_ORDERS} orders'}), 400

    results = []
    for raw in orders:
        oid = raw.get('id') if isinstance(raw, dict) else None
        o = _check(raw) if isinstance(raw, dict) else 'invalid order'
        if isinstance(o, str):
            results.append({'id': oid, 'status': 'rejected', 'reason': o})
            continue
        try:
            q = paper.quote(o['symbol'])
            if q['currency'] != 'USD':
                results.append({'id': oid, 'status': 'rejected',
                                'reason': 'paper trading supports USD-listed symbols only'})
                continue
            got = paper.fill_order(o, paper.bars_since(o['symbol'], o['placed_at']))
        except Exception as e:
            # Unknown ticker or a data hiccup: leave the order open, it's retried next poll
            results.append({'id': oid, 'status': 'open', 'reason': type(e).__name__})
            continue
        if got is None:
            results.append({'id': oid, 'status': 'open'})
        else:
            px, ts = got
            results.append({'id': oid, 'status': 'filled', 'price': round(px, 4),
                            'filled_at': ts.isoformat()})
    return jsonify({'success': True, 'results': results})


@bp.route('/api/paper/model')
def model_account():
    t = _symbol(request.args.get('ticker', 'NVDA'))
    if not t:
        return jsonify({'success': False, 'error': 'invalid ticker'}), 400
    model_sync.ensure_fresh()
    ledger = _LEDGERS / f'{t}_model_account.json'
    if not ledger.exists():
        return jsonify({'success': False,
                        'error': 'The model account starts with the next daily retrain.'})
    return jsonify({'success': True, **json.loads(ledger.read_text())})
