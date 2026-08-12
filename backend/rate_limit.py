"""
rate_limit.py — per-IP rate limiting for the public Render deployment.

Enablement (the "only on Render" contract):
    * Active automatically when running on Render — detected via the RENDER /
      RENDER_EXTERNAL_URL / RENDER_SERVICE_ID env vars Render sets on every
      service. Nothing to configure.
    * A plain local run (`python app.py`) is NEVER rate-limited.
    * RATE_LIMIT_ENABLED=true|false overrides the auto-detection either way
      (true is how you test the limiter on your own machine).

Personal bypass on the deployed URL:
    Set RATE_LIMIT_BYPASS_TOKEN in Render's env, then send the same value in
    an `X-RateLimit-Bypass` request header (browser header-modifier extension,
    curl -H, etc.). Matching requests are exempt from every limit.

Tiers:
    LLM_LIMIT      — endpoints that trigger a paid/queued LLM round-trip
    AV_LIMIT       — endpoints backed by Alpha Vantage (25 calls/day upstream)
    DEFAULT_LIMIT  — everything else, applied app-wide per IP

Storage is in-process memory — correct for the single-worker gthread model
(threads share one process). If GUNICORN_WORKERS is ever raised above 1,
point storage_uri at a shared Redis instance or each worker gets its own
independent budget.

Import is guarded like services.py: if flask-limiter isn't installed the app
still starts, routes work unlimited, and a warning lands in the deploy logs.
"""
import os

from flask import jsonify, request

try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    _LIMITER_AVAILABLE = True
except ImportError:
    _LIMITER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Enablement
# ---------------------------------------------------------------------------

def on_render() -> bool:
    """True when running on a Render service (Render sets these itself)."""
    return bool(os.environ.get('RENDER')
                or os.environ.get('RENDER_EXTERNAL_URL')
                or os.environ.get('RENDER_SERVICE_ID'))


def _enabled() -> bool:
    override = os.environ.get('RATE_LIMIT_ENABLED', '').strip().lower()
    if override in ('1', 'true', 'yes', 'on'):
        return True
    if override in ('0', 'false', 'no', 'off'):
        return False
    return on_render()


ENABLED = _enabled() and _LIMITER_AVAILABLE


# ---------------------------------------------------------------------------
# Exemptions
# ---------------------------------------------------------------------------

def bypass_ok() -> bool:
    """Requests carrying the owner's bypass token skip all limits."""
    token = os.environ.get('RATE_LIMIT_BYPASS_TOKEN', '')
    return bool(token) and request.headers.get('X-RateLimit-Bypass', '') == token


def _default_exempt() -> bool:
    # Static assets and Render's healthcheck never count against the budget.
    return request.endpoint in ('static', 'health') or bypass_ok()


# ---------------------------------------------------------------------------
# Limit tiers
# ---------------------------------------------------------------------------

LLM_LIMIT = '10 per minute; 100 per day'
AV_LIMIT = '4 per minute; 20 per day'
DEFAULT_LIMIT = '120 per minute'


if _LIMITER_AVAILABLE:
    limiter = Limiter(
        get_remote_address,
        default_limits=[DEFAULT_LIMIT],
        default_limits_exempt_when=_default_exempt,
        storage_uri='memory://',
        headers_enabled=True,   # X-RateLimit-* + Retry-After on responses
        enabled=ENABLED,
        swallow_errors=True,    # a limiter fault must never 500 an endpoint
    )
else:
    class _NoopLimiter:
        """Passthrough stand-in so route decorators stay valid."""

        def limit(self, *_a, **_kw):
            return lambda fn: fn

        def exempt(self, fn, *_a, **_kw):
            return fn

        def init_app(self, _app):
            pass

    limiter = _NoopLimiter()


def init_app(app):
    """Attach the limiter, the Render proxy fix, and a JSON/SSE-aware 429."""
    if on_render():
        # Render terminates TLS at its proxy, so request.remote_addr is the
        # proxy's IP unless we trust one X-Forwarded-For hop. Without this,
        # every visitor shares a single budget and one abuser locks out all.
        from werkzeug.middleware.proxy_fix import ProxyFix
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

    limiter.init_app(app)

    if _LIMITER_AVAILABLE:
        @app.errorhandler(429)
        def _rate_limited(e):
            import json
            desc = str(getattr(e, 'description', '') or 'too many requests')
            msg = f'Rate limit exceeded ({desc}). Please retry shortly.'
            if request.path == '/api/chat':
                # The chat widget parses SSE, not JSON — reply in its format
                # so the error renders in the bubble instead of hanging.
                body = ('data: ' + json.dumps({'error': msg}) + '\n\n'
                        'data: [DONE]\n\n')
                return app.response_class(body, status=429,
                                          mimetype='text/event-stream')
            return jsonify({'success': False, 'error': msg}), 429

    if _enabled() and not _LIMITER_AVAILABLE:
        print('[RATELIMIT] wanted ON but flask-limiter is not installed — '
              'running UNLIMITED (pip install flask-limiter)', flush=True)
    else:
        print(f"[RATELIMIT] {'ENABLED' if ENABLED else 'disabled'} "
              f'(on_render={on_render()})', flush=True)
