"""
routes/trading.py — Trading tab blueprint.

Two chart engines behind the same UI (main.js tries them in order):
  1. POST /api/trading/ohlcv  — yfinance bars + server-computed indicators,
     rendered client-side with LightweightCharts. No key, no quota. RECOVERED:
     this route was added in 751d772/02ad36a and accidentally dropped in 4f32baa
     while the frontend kept calling it, silently degrading the tab to engine 2.
  2. POST /api/trading/chart  — chart-img.com TradingView-style PNG proxy
     (CHART_IMG_KEY, 1 req/sec, 50/day on the free plan). Fallback only.
"""
import os

from flask import Blueprint, jsonify, request, Response

from common import keys_txt_value

bp = Blueprint('trading', __name__)


@bp.route('/api/trading/ohlcv', methods=['POST'])
def trading_ohlcv():
    """Fetch OHLCV bars from yfinance for the interactive trading chart.

    POST body: { "ticker": "AAPL", "interval": "1d", "period": "1y" }
    interval values (yfinance format): 1m 2m 5m 15m 30m 60m 1h 1d 5d 1wk 1mo 3mo
    period values: 1d 5d 1mo 3mo 6mo 1y 2y 5y 10y ytd max — or "auto"
    """
    try:
        import yfinance as _yf
    except ImportError:
        return jsonify({'success': False, 'error': 'yfinance not installed'}), 503

    body     = request.get_json(force=True, silent=True) or {}
    ticker   = (body.get('ticker') or 'AAPL').upper().strip()[:20]
    interval = body.get('interval', '1d')
    period   = body.get('period', '1y')

    # Map the TV-style intervals from the UI to yfinance equivalents
    _iv_map = {
        '1m': '1m',  '5m': '5m',  '15m': '15m', '30m': '30m',
        '1h': '60m', '4h': '60m',
        '1D': '1d',  '1W': '1wk', '1M': '1mo',
    }
    # Map interval → sensible default period when not specified
    _period_map = {
        '1m': '1d',  '5m': '5d',  '15m': '5d',  '30m': '60d',
        '60m': '60d', '1h': '60d', '4h': '6mo',
        '1d': '1y',  '1D': '1y',  '1wk': '5y', '1W': '5y',
        '1mo': 'max', '1M': 'max',
    }
    yf_interval = _iv_map.get(interval, interval)
    yf_period   = period if period != 'auto' else _period_map.get(yf_interval, '1y')

    try:
        import numpy as _np
        import pandas as _pd
        tk   = _yf.Ticker(ticker)
        hist = tk.history(period=yf_period, interval=yf_interval)
        if hist.empty:
            return jsonify({'success': False, 'error': f'No data returned for {ticker}'}), 404

        close = hist['Close']
        high  = hist['High']
        low   = hist['Low']

        # ── Indicators (NaN → None so JSON serialises cleanly) ──────────────
        def _s(series):
            """Round a pandas Series, coerce NaN→None for JSON."""
            return {ts: (None if _np.isnan(v) else round(float(v), 4))
                    for ts, v in series.items()}

        # EMAs
        ema20  = _s(close.ewm(span=20,  adjust=False).mean())
        ema50  = _s(close.ewm(span=50,  adjust=False).mean())
        ema200 = _s(close.ewm(span=200, adjust=False).mean())

        # Bollinger Bands (20, 2σ)
        bb_mid   = close.rolling(20).mean()
        bb_std   = close.rolling(20).std()
        bb_upper = _s(bb_mid + 2 * bb_std)
        bb_lower = _s(bb_mid - 2 * bb_std)
        bb_mid   = _s(bb_mid)

        # RSI (14)
        delta = close.diff()
        gain  = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rsi   = _s(100 - 100 / (1 + gain / loss.replace(0, _np.nan)))

        # MACD (12, 26, 9)
        ema12      = close.ewm(span=12, adjust=False).mean()
        ema26      = close.ewm(span=26, adjust=False).mean()
        macd_line  = ema12 - ema26
        macd_sig   = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist_s = _s(macd_line - macd_sig)
        macd_line  = _s(macd_line)
        macd_sig   = _s(macd_sig)

        # Stochastic (14, 3)
        low14  = low.rolling(14).min()
        high14 = high.rolling(14).max()
        stk    = 100 * (close - low14) / (high14 - low14).replace(0, _np.nan)
        std    = _s(stk.rolling(3).mean())
        stk    = _s(stk)

        # ATR (14)
        tr  = _pd.concat([(high - low).abs(),
                           (high - close.shift()).abs(),
                           (low  - close.shift()).abs()], axis=1).max(axis=1)
        atr = _s(tr.rolling(14).mean())

        # CCI (20)
        tp  = (high + low + close) / 3
        cci = _s((tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std()))

        # ── Build bars array ─────────────────────────────────────────────────
        bars = []
        for ts, row in hist.iterrows():
            t = int(ts.timestamp())
            bar = {
                'time':   t,
                'open':   round(float(row['Open']),   4),
                'high':   round(float(row['High']),   4),
                'low':    round(float(row['Low']),    4),
                'close':  round(float(row['Close']),  4),
                'volume': int(row['Volume']),
                # overlays
                'ema20':  ema20.get(ts),
                'ema50':  ema50.get(ts),
                'ema200': ema200.get(ts),
                'bb_upper': bb_upper.get(ts),
                'bb_mid':   bb_mid.get(ts),
                'bb_lower': bb_lower.get(ts),
                # sub-pane indicators
                'rsi':        rsi.get(ts),
                'macd':       macd_line.get(ts),
                'macd_sig':   macd_sig.get(ts),
                'macd_hist':  macd_hist_s.get(ts),
                'stoch_k':    stk.get(ts),
                'stoch_d':    std.get(ts),
                'atr':        atr.get(ts),
                'cci':        cci.get(ts),
            }
            bars.append(bar)

        info = tk.fast_info
        currency = getattr(info, 'currency', 'USD') or 'USD'

        return jsonify({
            'success':  True,
            'ticker':   ticker,
            'interval': interval,
            'period':   yf_period,
            'currency': currency,
            'bars':     bars,
        })
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


@bp.route('/api/trading/chart', methods=['POST'])
def trading_chart():
    """Proxy chart-img.com TradingView chart image requests.

    Tries the v2 POST endpoint first (paid plan); falls back to v1 GET (free plan).
    Auth: x-api-key header — CHART_IMG_KEY env var, falling back to keys.txt
    (section header containing "chart-img") for local dev.
    """
    import requests as _req

    CHART_IMG_KEY = os.environ.get('CHART_IMG_KEY', '') or keys_txt_value('chart-img')
    if not CHART_IMG_KEY:
        return jsonify({'error': 'chart-img API key not configured',
                        'details': 'Set CHART_IMG_KEY (env or Render dashboard) or add a chart-img section to keys.txt.'}), 503

    data = request.get_json(force=True) or {}
    symbol   = data.get('symbol', 'NASDAQ:AAPL')
    interval = data.get('interval', '1D')
    style    = data.get('style', 'candle')
    width    = int(data.get('width', 800))
    height   = int(data.get('height', 600))
    headers  = {'x-api-key': CHART_IMG_KEY, 'content-type': 'application/json'}

    # Convert study name strings to v2 object array.
    # Supports "EMA:N" shorthand → {"name": "Moving Average Exponential", "input": {"length": N}}
    raw_studies = data.get('studies', [])
    studies_v2 = []
    for s in raw_studies:
        if s.startswith('EMA:'):
            try:
                length = int(s.split(':')[1])
            except (IndexError, ValueError):
                length = 20
            studies_v2.append({'name': 'Moving Average Exponential', 'input': {'length': length}})
        else:
            studies_v2.append({'name': s})

    try:
        payload = {
            'symbol': symbol, 'interval': interval,
            'style': style, 'theme': 'dark',
            'studies': studies_v2, 'width': width, 'height': height,
            'timezone': 'America/New_York',
        }
        resp = _req.post(
            'https://api.chart-img.com/v2/tradingview/advanced-chart',
            json=payload, headers=headers, timeout=45,
        )
        if resp.status_code == 200:
            return Response(resp.content, mimetype='image/png')
        try:
            err_body = resp.json()
        except Exception:
            err_body = resp.text
        status_msg = str(err_body)
        if resp.status_code == 403:
            status_msg = 'chart-img access denied — check API key or daily quota.'
        elif resp.status_code == 429:
            status_msg = 'chart-img rate/limit exceeded. Try again shortly.'
        return jsonify({'error': 'chart-img API error', 'status': resp.status_code, 'details': status_msg}), 502
    except Exception as e:
        return jsonify({'error': str(e)}), 500
