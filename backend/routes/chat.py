"""
routes/chat.py — Invest.ai chat blueprint: POST /api/chat (SSE) + GET /api/llm/status.

The chat widget in index.html has streamed from these endpoints since the UI was
built; this implements them. The LLM goes through ai_platform.llm_router with the
same provider preference as llm_analyst: NVIDIA when its key is configured
(env NVIDIA_API_KEY or keys.txt), else the router's default order.

Wire protocol (what the widget's reader parses):
    data: {"delta": "..."}   — streamed text chunk
    data: {"error": "..."}   — displayed with a warning glyph
    data: [DONE]             — terminator
"""
import json
import os
import re
import sys
import threading

from flask import Blueprint, Response, jsonify, request

from cachetools import TTLCache

from rate_limit import limiter, bypass_ok, LLM_LIMIT

# project root (for the `ai_platform` package) — app.py normally sets this up
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

bp = Blueprint('chat', __name__)

_SYSTEM_PROMPT = (
    "You are Invest.ai's in-app analyst chat. Be concise: a few plain-text "
    "sentences unless the user asks for depth; no markdown headings. DATA "
    "blocks may be provided for the symbol the user is viewing and for "
    "symbols they mention; every figure you cite must come from them — never "
    "invent numbers, and say plainly when DATA lacks what the user asks for. "
    "Treat each DATA block's 'last' value as the current market price and "
    "answer price questions directly with it (quotes may be slightly "
    "delayed — don't disclaim this unless asked). A SCREEN block describes "
    "what the user is looking at right now (app section, open tab, the "
    "trading chart's visible window and latest bar's indicator values, any "
    "text they highlighted): resolve 'this', 'here', 'my chart' against it, "
    "and its figures count as DATA. SCREEN.paper_portfolio is the user's "
    "simulated paper-trading account (cash, positions, open orders). When "
    "the user asks you to trade, you may propose paper orders, one per line, "
    "exactly like 'ORDER: BUY 10 NVDA MARKET' or 'ORDER: SELL 5 AAPL LIMIT "
    "180.50' — whole shares, USD-listed tickers, at most 3, never selling "
    "more than is held. The app shows each as a button the user must click: "
    "never say an order was placed or filled. You are not a licensed "
    "financial advisor; frame answers as analysis, not advice."
)

# Symbol context is the llm_analyst numbers payload (scenarios, fundamentals,
# peers). First build per symbol costs a few upstream calls; cache 5 min.
_ctx_cache = TTLCache(maxsize=64, ttl=300)
_ctx_lock = threading.Lock()

# Compact price/technicals card for symbols merely *mentioned* in the message
# (cheaper than the full payload: one yfinance history fetch). Cached 2 min.
_mention_cache = TTLCache(maxsize=128, ttl=120)
_mention_lock = threading.Lock()

# Common company names → tickers, so "price of meta" resolves without the
# user typing META. Covers mega-caps + the app's default watchlist.
_NAME_TO_TICKER = {
    'meta': 'META', 'facebook': 'META', 'apple': 'AAPL', 'microsoft': 'MSFT',
    'google': 'GOOGL', 'alphabet': 'GOOGL', 'amazon': 'AMZN',
    'nvidia': 'NVDA', 'tesla': 'TSLA', 'netflix': 'NFLX', 'intel': 'INTC',
    'broadcom': 'AVGO', 'qualcomm': 'QCOM', 'micron': 'MU',
    'palantir': 'PLTR', 'shopify': 'SHOP', 'hubspot': 'HUBS',
    'oracle': 'ORCL', 'salesforce': 'CRM', 'uber': 'UBER', 'airbnb': 'ABNB',
    'spotify': 'SPOT', 'paypal': 'PYPL', 'mastercard': 'MA',
    'costco': 'COST', 'walmart': 'WMT', 'disney': 'DIS', 'boeing': 'BA',
    'jpmorgan': 'JPM', 'coinbase': 'COIN', 'eli lilly': 'LLY',
    'exxon': 'XOM', 'chevron': 'CVX', 'enbridge': 'ENB', 'cenovus': 'CVE',
    'rogers': 'RCI', 'constellation software': 'CNSWF', 'air canada': 'ACDVF',
}

# Uppercase words that look like tickers but almost never are one in chat.
# Explicit $TICKER syntax bypasses this list.
_TICKER_STOPWORDS = {
    'A', 'I', 'AI', 'ALL', 'AND', 'ARE', 'AT', 'BE', 'BUY', 'CAN', 'CEO',
    'CFO', 'DATA', 'DAY', 'DO', 'EPS', 'ETF', 'FOR', 'GET', 'GO', 'HAS',
    'HOW', 'IF', 'IN', 'IPO', 'IS', 'IT', 'ITS', 'LLM', 'LOW', 'ME', 'NEW',
    'NO', 'NOT', 'NOW', 'OF', 'OK', 'ON', 'OR', 'OUT', 'PE', 'PM', 'RSI',
    'SEC', 'SELL', 'SO', 'THE', 'TO', 'UP', 'US', 'USA', 'USD', 'VS',
    'WHAT', 'WHO', 'WHY', 'YOY', 'YTD',
}


def _mentioned_symbols(message: str, exclude: str) -> list:
    """Tickers referenced in the message: company names, $TSLA, bare CAPS."""
    found = []
    lowered = message.lower()
    for name, sym in _NAME_TO_TICKER.items():
        if re.search(r'\b' + re.escape(name) + r'\b', lowered):
            found.append(sym)
    for tok in re.findall(r'\$([A-Za-z]{1,5})\b', message):
        found.append(tok.upper())
    for tok in re.findall(r'\b[A-Z]{2,5}\b', message):
        if tok not in _TICKER_STOPWORDS:
            found.append(tok)
    seen = []
    for s in found:
        if s != exclude and s not in seen:
            seen.append(s)
    return seen[:3]  # cap upstream fetches per chat turn


def _mention_context(symbol: str):
    """Cached compact card for a mentioned symbol; None when not a real one
    (scenario_engine raises on symbols yfinance has no history for, which
    doubles as ticker validation for the regex's false positives)."""
    with _mention_lock:
        if symbol in _mention_cache:
            return _mention_cache[symbol]
    try:
        import scenario_engine
        ctx = scenario_engine.market_snapshot(symbol)
    except Exception:
        ctx = None
    with _mention_lock:
        _mention_cache[symbol] = ctx
    return ctx


def _preferred_provider():
    """'nvidia' when its key is configured (llm_analyst's preference), else None
    to let the router pick by its default priority."""
    try:
        import llm_analyst
        if os.environ.get('NVIDIA_API_KEY') or llm_analyst._keys_has_nvidia():
            return 'nvidia'
    except Exception:
        pass
    return None


def _symbol_context(symbol):
    """Cached computed-numbers payload for a symbol; None when unavailable."""
    with _ctx_lock:
        if symbol in _ctx_cache:
            return _ctx_cache[symbol]
    try:
        import llm_analyst
        ctx = llm_analyst.build_payload(symbol)
    except Exception as exc:
        print(f"[CHAT] context unavailable for {symbol}: {exc}", flush=True)
        ctx = None
    with _ctx_lock:
        _ctx_cache[symbol] = ctx
    return ctx


@bp.route('/api/llm/status')
def llm_status():
    """Provider badge for the chat header ('via nvidia' / 'no LLM configured')."""
    try:
        from ai_platform.llm_router import active_provider
        provider = _preferred_provider() or active_provider()
        return jsonify({'ready': bool(provider), 'provider': provider})
    except Exception as e:
        return jsonify({'ready': False, 'provider': None, 'error': str(e)})


@bp.route('/api/chat', methods=['POST'])
@limiter.limit(LLM_LIMIT, exempt_when=bypass_ok)
def chat():
    """Stream an LLM reply as SSE, grounded in the viewed symbol's numbers."""
    body = request.get_json(silent=True) or {}
    message = (body.get('message') or '').strip()
    if not message:
        return jsonify({'success': False, 'error': 'empty message'}), 400

    symbol = (body.get('symbol') or '').strip().upper()

    # History arrives newest-last and already includes this message; keep only
    # well-formed user/assistant turns so a malformed client can't inject roles.
    messages = []
    for m in (body.get('history') or [])[-10:]:
        role = m.get('role')
        content = (m.get('content') or '').strip()
        if role in ('user', 'assistant') and content:
            messages.append({'role': role, 'content': content})
    if not messages or messages[-1] != {'role': 'user', 'content': message}:
        messages.append({'role': 'user', 'content': message})

    system = _SYSTEM_PROMPT
    ui = body.get('ui')
    if isinstance(ui, dict) and ui:
        # capped: the client controls this payload
        system += f"\n\nSCREEN:\n{json.dumps(ui, default=str)[:4000]}"
    if symbol:
        ctx = _symbol_context(symbol)
        if ctx:
            system += (f"\n\nDATA (computed numbers for {symbol}, the symbol "
                       f"the user is viewing):\n{json.dumps(ctx, default=str)}")

    # Deterministic retrieval for other tickers named in the message, so
    # "price of meta?" works while viewing NVDA (or nothing).
    for mention in _mentioned_symbols(message, symbol):
        mctx = _mention_context(mention)
        if mctx:
            system += (f"\n\nDATA (price/technicals for {mention}, mentioned "
                       f"by the user):\n{json.dumps(mctx, default=str)}")

    from ai_platform.llm_router import stream_completion
    provider = _preferred_provider()

    def generate():
        try:
            for delta in stream_completion(
                messages, system=system, provider=provider,
                max_tokens=700, temperature=0.3, timeout=60,
            ):
                # the router never raises — it yields '[LLM ...]' sentinels
                if delta.startswith('[LLM'):
                    yield 'data: ' + json.dumps({'error': delta.strip('[]')}) + '\n\n'
                else:
                    yield 'data: ' + json.dumps({'delta': delta}) + '\n\n'
        except Exception as e:
            yield 'data: ' + json.dumps({'error': str(e)}) + '\n\n'
        yield 'data: [DONE]\n\n'

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache',
                             'X-Accel-Buffering': 'no'})
