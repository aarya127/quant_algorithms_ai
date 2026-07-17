"""
llm_analyst.py — LLM narrative layer over computed numbers.

Division of labor (deliberate): scenario_engine.py computes every number
(targets, bands, probabilities); this module feeds those numbers — plus
fundamentals, peer comparisons, technicals, and sentiment — to an LLM that
writes the words. The LLM never invents or alters a figure (HARD RULES in the
prompt) and the whole layer is additive: on any failure callers degrade to
numbers-only output.

Guardrails:
  * one LLM call per symbol per hour (TTL cache), temperature 0.2
  * provider preference: NVIDIA (free) → router default order
  * every payload + response is appended to analyst_log.jsonl for later audit
"""
import datetime
import json
import os
import sys
import threading

from cachetools import TTLCache

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from services import (
    get_company_statistics, get_company_news, get_insider_sentiment,
)
import scenario_engine

_LOG_PATH = os.environ.get(
    'ANALYST_LOG_PATH',
    os.path.join(os.path.dirname(__file__), 'analyst_log.jsonl'),
)

_brief_cache = TTLCache(maxsize=64, ttl=3600)   # one LLM call / symbol / hour
_peers_cache = TTLCache(maxsize=256, ttl=86400)  # peer sets barely change
_lock = threading.Lock()

_SYSTEM_PROMPT = """You are an equity analyst writing a terse brief. You are given DATA (JSON).

HARD RULES:
1. Use ONLY numbers present in DATA. Never supply a figure from memory — your
   training data is stale; DATA is live.
2. A field that is null or absent is "unavailable" — say so, never estimate it.
3. Compare the company ONLY against the peers listed in DATA.
4. The scenario targets and probabilities in DATA were computed upstream —
   EXPLAIN them, never alter or re-derive them.
5. Every quantitative claim must trace to a DATA field.
6. NEVER derive higher/lower/above/below/better/worse yourself. All peer
   comparison directions are precomputed in DATA.peer_comparisons
   (peers_with_higher_value / peers_with_lower_value). Use those lists
   verbatim; if a metric is not in peer_comparisons, do not compare it.
   Never mention this rule or any verification in the output.

Return STRICT JSON (no markdown fences, no commentary):
{"signal": "long|neutral|short",
 "narrative": "2-3 sentences: the one thing that matters most right now",
 "competitive_position": "1-2 sentences vs the provided peers",
 "bull_factors": ["...", "...", "..."],
 "bear_factors": ["...", "...", "..."],
 "risk_flags": ["only if DATA supports them"],
 "scenario_rationale": {"bull": "...", "base": "...", "bear": "..."}}"""


def _peer_table(symbol):
    """3-4 Finnhub peers with compact comparison metrics (cached 24h)."""
    symbol = symbol.upper()
    with _lock:
        if symbol in _peers_cache:
            return _peers_cache[symbol]
    peers = []
    try:
        from services import get_basic_financials
        from data.finnhub import get_company_peers
        for p in [t for t in (get_company_peers(symbol) or []) if t != symbol][:4]:
            row = {'symbol': p}
            try:
                m = (get_basic_financials(p) or {}).get('metric', {})
                row.update({
                    'pe': m.get('peTTM'),
                    'net_margin': m.get('netProfitMarginTTM'),
                    'rev_growth': m.get('revenueGrowthTTMYoy'),
                    'ret_3m_pct': m.get('13WeekPriceReturnDaily'),
                })
            except Exception:
                pass
            peers.append(row)
    except Exception as exc:
        print(f'[ANALYST] peers unavailable for {symbol}: {exc}', flush=True)
    with _lock:
        _peers_cache[symbol] = peers
    return peers


def build_payload(symbol, timeframe='1M'):
    """Assemble the computed-numbers payload the LLM is allowed to talk about."""
    symbol = symbol.upper()
    payload = {'symbol': symbol,
               'as_of': datetime.date.today().isoformat(),
               'timeframe': timeframe}

    payload['price_technicals'] = scenario_engine.market_snapshot(symbol)

    sc = scenario_engine.compute_scenarios(symbol, timeframe)
    payload['scenarios_computed'] = {
        'engine': sc['engine'], 'model_backed': sc['model_backed'],
        'p_up': sc['p_up'],
        'bull': sc['scenarios']['bull_case'],
        'base': sc['scenarios']['base_case'],
        'bear': sc['scenarios']['bear_case'],
    }

    try:  # fundamentals — nulls stay null (HARD RULE 2 handles them)
        stats = get_company_statistics(symbol) or {}
        payload['fundamentals'] = {
            'valuation': stats.get('valuation_ttm'),
            'margins': stats.get('margins'),
            'growth': stats.get('growth'),
            'financial_health': stats.get('financial_health'),
        }
    except Exception:
        payload['fundamentals'] = None

    payload['peers'] = _peer_table(symbol)

    # Precomputed comparison directions — the LLM is NOT allowed to derive
    # higher/lower itself (small models silently flip them); it reads these.
    try:
        from services import get_basic_financials
        subj = (get_basic_financials(symbol) or {}).get('metric', {})
        subject_metrics = {
            'pe': subj.get('peTTM'),
            'net_margin': subj.get('netProfitMarginTTM'),
            'rev_growth': subj.get('revenueGrowthTTMYoy'),
            'ret_3m_pct': subj.get('13WeekPriceReturnDaily'),
        }
        comparisons = {}
        for metric, own in subject_metrics.items():
            if own is None:
                continue
            higher = [p['symbol'] for p in payload['peers']
                      if p.get(metric) is not None and p[metric] > own]
            lower = [p['symbol'] for p in payload['peers']
                     if p.get(metric) is not None and p[metric] < own]
            comparisons[metric] = {
                f'{symbol}_value': own,
                'peers_with_higher_value': higher,
                'peers_with_lower_value': lower,
            }
        payload['peer_comparisons'] = comparisons
    except Exception:
        payload['peer_comparisons'] = None

    sentiment = {}
    today = datetime.date.today()
    try:
        news = get_company_news(symbol, (today - datetime.timedelta(days=7)).isoformat(),
                                today.isoformat()) or []
        sentiment['headlines'] = [
            {'date': datetime.datetime.fromtimestamp(a.get('datetime', 0)).strftime('%m-%d'),
             'title': a.get('headline', '')[:120]}
            for a in news[:5]
        ]
    except Exception:
        sentiment['headlines'] = []
    try:
        ins = get_insider_sentiment(symbol, (today - datetime.timedelta(days=90)).isoformat(),
                                    today.isoformat())
        mspr = [d.get('mspr') for d in (ins or {}).get('data', []) if d.get('mspr') is not None]
        sentiment['insider_mspr_3m'] = round(sum(mspr) / len(mspr), 1) if mspr else None
    except Exception:
        sentiment['insider_mspr_3m'] = None
    payload['sentiment'] = sentiment

    return payload


def generate_brief(symbol, timeframe='1M'):
    """Payload → LLM → validated dict. None when no provider/parse failure."""
    symbol = symbol.upper()
    with _lock:
        if symbol in _brief_cache:
            return _brief_cache[symbol]

    payload = build_payload(symbol, timeframe)

    from ai_platform.llm_router import chat_completion, active_provider
    provider = 'nvidia' if os.environ.get('NVIDIA_API_KEY') or _keys_has_nvidia() else None
    raw = chat_completion(
        [{'role': 'user', 'content': 'DATA:\n' + json.dumps(payload, default=str)}],
        system=_SYSTEM_PROMPT, provider=provider,
        temperature=0.2, max_tokens=900, timeout=45,
    )
    if raw is None and provider:  # NVIDIA down → router default order
        raw = chat_completion(
            [{'role': 'user', 'content': 'DATA:\n' + json.dumps(payload, default=str)}],
            system=_SYSTEM_PROMPT, temperature=0.2, max_tokens=900, timeout=30,
        )
    if raw is None:
        return None

    try:  # strict-ish parse: strip fences, require the core keys
        text = raw.strip()
        if text.startswith('```'):
            text = text.split('```')[1].lstrip('json').strip()
        brief = json.loads(text)
        assert brief.get('signal') in ('long', 'neutral', 'short')
        assert isinstance(brief.get('narrative'), str)
    except Exception as exc:
        print(f'[ANALYST] unparseable brief for {symbol}: {exc}', flush=True)
        _log(symbol, payload, raw, ok=False)
        return None

    brief['provider'] = provider or active_provider()
    brief['generated_at'] = datetime.datetime.now().isoformat()
    _log(symbol, payload, brief, ok=True)
    with _lock:
        _brief_cache[symbol] = brief
    return brief


def _keys_has_nvidia():
    try:
        from common import keys_txt_value
        return bool(keys_txt_value('nvidia'))
    except Exception:
        return False


def _log(symbol, payload, response, ok):
    """Audit trail: every brief (or failure) as one JSONL row."""
    try:
        with open(_LOG_PATH, 'a') as fh:
            fh.write(json.dumps({
                'ts': datetime.datetime.now().isoformat(),
                'symbol': symbol, 'ok': ok,
                'payload': payload, 'response': response,
            }, default=str) + '\n')
    except OSError:
        pass
