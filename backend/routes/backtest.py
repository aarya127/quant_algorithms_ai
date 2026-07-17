"""
routes/backtest.py — backtesting blueprint, extracted verbatim from app.py.
"""
import os
import sys
import json
import datetime

from flask import Blueprint, jsonify, request

from services import yf

bp = Blueprint('backtest', __name__)

# backend/ directory (this file lives one level deeper than app.py did; moved
# code that used os.path.dirname(__file__) now uses _BACKEND instead)
_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@bp.route('/api/backtest', methods=['POST'])
def run_backtest_api():
    """Run a real options volatility backtest via VolatilityStrategyBacktestEngine."""
    try:
        from backtest_runner import run_backtest, STRATEGIES
        body = request.get_json(force=True) or {}

        strategy   = body.get('strategy', '')
        asset      = body.get('asset', '').upper().strip()
        start_date = body.get('start_date', '')
        end_date   = body.get('end_date', '')
        capital    = float(body.get('capital', 250_000))

        if not strategy or strategy not in STRATEGIES:
            return jsonify({'success': False,
                            'error': f"Invalid strategy. Choose from: {list(STRATEGIES.keys())}"}), 400
        if not asset:
            return jsonify({'success': False, 'error': 'Asset symbol is required'}), 400
        if not start_date or not end_date:
            return jsonify({'success': False, 'error': 'start_date and end_date are required'}), 400

        result = run_backtest(strategy=strategy, asset=asset,
                              start_date=start_date, end_date=end_date, capital=capital)
        return jsonify(result)

    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        import traceback
        print(f"[BACKTEST ERROR] {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/api/backtest/strategies')
def backtest_strategies():
    """Return the list of available backtest strategy names."""
    try:
        from backtest_runner import STRATEGIES
        return jsonify({'success': True, 'strategies': list(STRATEGIES.keys())})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

