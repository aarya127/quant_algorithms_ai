"""
routes/charts.py — chart-data blueprint (OHLCV, timeframes, comparison, indicators).

Extracted from app.py; behavior unchanged. Data layer is imported behind a guard so
the routes degrade to a clean 503 (instead of NameError → 500) if it fails to load.
"""
import os
import sys

from flask import Blueprint, jsonify, request

# project root (for the `data` package) — app.py normally sets this up already
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from common import (  # noqa: E402
    get_ticker_for_charts, is_canadian_stock, _validate_chart_args,
)

bp = Blueprint('charts', __name__)

try:
    from data.charts import (
        get_chart_data, get_multiple_timeframes,
        get_comparison_data, get_technical_indicators,
    )
    _CHARTS_LOADED = True
except Exception as _e:
    print(f"[CHARTS] data layer failed to import: {_e}", flush=True)
    _CHARTS_LOADED = False


def _unavailable():
    return jsonify({'success': False,
                    'error': 'chart data layer unavailable'}), 503


@bp.route('/api/charts/<symbol>')
def get_charts(symbol):
    """Get chart data for a stock

    Query Parameters:
    - period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max (default: 1y)
    - interval: 1m, 2m, 5m, 15m, 30m, 60m, 1h, 1d, 5d, 1wk, 1mo (default: 1d)
    """
    period = request.args.get('period', '1y')
    interval = request.args.get('interval', '1d')
    err = _validate_chart_args(period, interval)
    if err:
        return err
    if not _CHARTS_LOADED:
        return _unavailable()

    try:
        symbol_upper = symbol.upper()
        # Use TSX symbol for Canadian stocks to get CAD prices
        chart_symbol = get_ticker_for_charts(symbol_upper)

        data = get_chart_data(chart_symbol, period, interval)
        # Keep the original symbol in response
        data['display_symbol'] = symbol_upper
        data['currency'] = 'CAD' if is_canadian_stock(symbol_upper) else 'USD'
        return jsonify(data)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/charts/<symbol>/all-timeframes')
def get_all_timeframes(symbol):
    """Get chart data for all timeframes"""
    if not _CHARTS_LOADED:
        return _unavailable()

    try:
        symbol_upper = symbol.upper()
        # Use TSX symbol for Canadian stocks to get CAD prices
        chart_symbol = get_ticker_for_charts(symbol_upper)

        data = get_multiple_timeframes(chart_symbol)
        data['display_symbol'] = symbol_upper
        data['currency'] = 'CAD' if is_canadian_stock(symbol_upper) else 'USD'
        return jsonify(data)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/charts/compare')
def compare_charts():
    """Compare multiple stocks

    Query Parameters:
    - symbols: Comma-separated list of symbols (e.g., AAPL,MSFT,GOOGL)
    - period: Time period (default: 1y)
    - interval: Data interval (default: 1d)
    """
    symbols_param = request.args.get('symbols', '')
    symbols = [s.strip().upper() for s in symbols_param.split(',') if s.strip()]

    if not symbols:
        return jsonify({
            'success': False,
            'error': 'No symbols provided'
        }), 400

    period = request.args.get('period', '1y')
    interval = request.args.get('interval', '1d')
    err = _validate_chart_args(period, interval)
    if err:
        return err
    if not _CHARTS_LOADED:
        return _unavailable()

    try:
        data = get_comparison_data(symbols, period, interval)
        return jsonify(data)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/charts/<symbol>/indicators')
def get_indicators(symbol):
    """Get chart data with technical indicators (RSI, MACD, Bollinger Bands, Moving Averages)

    Query Parameters:
    - period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max (default: 1y)
    - interval: 1m, 2m, 5m, 15m, 30m, 60m, 1h, 1d, 5d, 1wk, 1mo (default: 1d)
    """
    period = request.args.get('period', '1y')
    interval = request.args.get('interval', '1d')
    err = _validate_chart_args(period, interval)
    if err:
        return err
    if not _CHARTS_LOADED:
        return _unavailable()

    try:
        symbol_upper = symbol.upper()
        # Use TSX symbol for Canadian stocks to get CAD prices
        chart_symbol = get_ticker_for_charts(symbol_upper)

        data = get_technical_indicators(chart_symbol, period, interval)
        # Keep the original symbol in response
        data['display_symbol'] = symbol_upper
        data['currency'] = 'CAD' if is_canadian_stock(symbol_upper) else 'USD'
        return jsonify(data)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
