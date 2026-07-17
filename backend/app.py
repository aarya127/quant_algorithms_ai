"""
Invest.ai Web Application — Flask app entrypoint.

All API routes live in blueprints under backend/routes/ (one module per domain).
Shared config is in common.py; the guarded data layer + TTL caches in services.py.
"""

import os
import sys

from flask import Flask, render_template, jsonify

# Add parent directory to path (for the data/ai_platform packages)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Create the Flask app first — this guarantees /health always responds even if
# downstream data-module imports fail, keeping deploy healthchecks green.
app = Flask(__name__)

from common import DEFAULT_STOCKS          # noqa: E402
from services import stop_news_stream      # noqa: E402  (no-op if data layer failed)

from routes.pipeline import bp as _pipeline_bp    # noqa: E402
from routes.charts import bp as _charts_bp        # noqa: E402
from routes.news import bp as _news_bp            # noqa: E402
from routes.trading import bp as _trading_bp      # noqa: E402
from routes.market import bp as _market_bp        # noqa: E402
from routes.stock import bp as _stock_bp          # noqa: E402
from routes.backtest import bp as _backtest_bp    # noqa: E402
from routes.research import bp as _research_bp    # noqa: E402

for _bp in (_pipeline_bp, _charts_bp, _news_bp, _trading_bp,
            _market_bp, _stock_bp, _backtest_bp, _research_bp):
    app.register_blueprint(_bp)


@app.route('/health')
def health():
    """Deploy healthcheck endpoint — always returns 200 immediately"""
    return jsonify({'status': 'ok'}), 200


@app.route('/')
def index():
    """Main dashboard"""
    resp = app.make_response(render_template('index.html', default_stocks=DEFAULT_STOCKS))
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    return resp


if __name__ == '__main__':
    try:
        port = int(os.environ.get('PORT', 5001))
        debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
        app.run(debug=debug, host='0.0.0.0', port=port)
    finally:
        # Clean up: stop the Alpaca news stream when app closes
        stop_news_stream()
