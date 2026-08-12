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
    "sentences unless the user asks for depth; no markdown headings. When a "
    "DATA block for the symbol the user is viewing is provided, every figure "
    "you cite must come from it — never invent numbers, and say so when DATA "
    "lacks what the user asks for. Quotes may be delayed. You are not a "
    "licensed financial advisor; frame answers as analysis, not advice."
)

# Symbol context is the llm_analyst numbers payload (scenarios, fundamentals,
# peers). First build per symbol costs a few upstream calls; cache 5 min.
_ctx_cache = TTLCache(maxsize=64, ttl=300)
_ctx_lock = threading.Lock()


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
    if symbol:
        ctx = _symbol_context(symbol)
        if ctx:
            system += (f"\n\nDATA (computed numbers for {symbol}, the symbol "
                       f"the user is viewing):\n{json.dumps(ctx, default=str)}")

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
