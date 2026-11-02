"""
routes/stock.py — per-symbol detail (stock, ai-overview, statistics, sentiment, scenarios, metrics, recommendations) blueprint, extracted verbatim from app.py.
"""
import os
import json
import datetime

from concurrent.futures import ThreadPoolExecutor

from flask import Blueprint, jsonify, request

from common import get_ticker_for_charts, is_canadian_stock, _int_arg
from rate_limit import limiter, bypass_ok, LLM_LIMIT, AV_LIMIT
from services import (
    yf, av, get_company_overview_llm,
    get_company_profile, get_stock_quote, get_basic_financials, get_company_news,
    get_company_statistics, format_statistics_for_display,
    _cached_get_stock_quote, _cached_get_company_profile, _cached_get_company_peers,
    _cached_get_basic_financials, _cached_get_company_news, _cached_yf_info,
    _safe, graded_fundamentals,
)

bp = Blueprint('stock', __name__)

# backend/ directory (this file lives one level deeper than app.py did; moved
# code that used os.path.dirname(__file__) now uses _BACKEND instead)
_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@bp.route('/api/stock/<symbol>')
def stock_details(symbol):
    """Get detailed stock information"""
    try:
        symbol_upper = symbol.upper()
        today = datetime.date.today()
        from_date = (today - datetime.timedelta(days=7)).strftime('%Y-%m-%d')
        to_date = today.strftime('%Y-%m-%d')

        if is_canadian_stock(symbol_upper):
            tsx_symbol = get_ticker_for_charts(symbol_upper)

            # Fetch yfinance info + history in parallel with news
            def _fetch_yf():
                info = _cached_yf_info(tsx_symbol)
                hist = yf.Ticker(tsx_symbol).history(period='5d')
                return info, hist

            with ThreadPoolExecutor(max_workers=2) as ex:
                future_yf   = ex.submit(_fetch_yf)
                future_news = ex.submit(_cached_get_company_news, symbol_upper, from_date, to_date)
                info, hist  = future_yf.result()
                news        = future_news.result()

            company = {
                'name': info.get('longName', info.get('shortName', symbol_upper)),
                'ticker': symbol_upper,
                'country': 'CA',
                'currency': 'CAD',
                'exchange': info.get('exchange', 'TSX'),
                'ipo': info.get('ipoDate', ''),
                'marketCapitalization': info.get('marketCap', 0),
                'shareOutstanding': info.get('sharesOutstanding', 0),
                'logo': info.get('logo_url', ''),
                'phone': info.get('phone', ''),
                'weburl': info.get('website', ''),
                'finnhubIndustry': info.get('industry', ''),
                'sector': info.get('sector', ''),
                'city': info.get('city', ''),
                'state': info.get('state', ''),
                'country_full': info.get('country', 'Canada'),
                'fullTimeEmployees': info.get('fullTimeEmployees', 0),
                'longBusinessSummary': info.get('longBusinessSummary', ''),
                'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh', 0),
                'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow', 0),
                'dividendYield': info.get('dividendYield', 0),
                'trailingPE': info.get('trailingPE', 0),
                'forwardPE': info.get('forwardPE', 0),
                'priceToBook': info.get('priceToBook', 0),
                'beta': info.get('beta', 0)
            }

            if not hist.empty:
                latest = hist.iloc[-1]
                prev = hist.iloc[-2] if len(hist) > 1 else latest
                quote = {
                    'c': round(latest['Close'], 2),
                    'h': round(latest['High'], 2),
                    'l': round(latest['Low'], 2),
                    'o': round(latest['Open'], 2),
                    'pc': round(prev['Close'], 2),
                    'd': round(latest['Close'] - prev['Close'], 2),
                    'dp': round(((latest['Close'] - prev['Close']) / prev['Close'] * 100), 2)
                }
            else:
                quote = {}

            # Financials fetched after the parallel block (Canadian: no Finnhub financials)
            financials = _cached_get_basic_financials(symbol_upper)

        else:
            # US stocks: fire all 4 independent calls in parallel
            def _fetch_profile():  return _cached_get_company_profile(symbol_upper)
            def _fetch_quote():    return _cached_get_stock_quote(symbol_upper)
            def _fetch_yf_info():  return _cached_yf_info(symbol_upper)
            def _fetch_news():     return _cached_get_company_news(symbol_upper, from_date, to_date)
            def _fetch_fins():     return _cached_get_basic_financials(symbol_upper)

            with ThreadPoolExecutor(max_workers=5) as ex:
                f_profile = ex.submit(_fetch_profile)
                f_quote   = ex.submit(_fetch_quote)
                f_yf_info = ex.submit(_fetch_yf_info)
                f_news    = ex.submit(_fetch_news)
                f_fins    = ex.submit(_fetch_fins)

                company    = f_profile.result()
                quote      = f_quote.result()
                info       = f_yf_info.result()
                news       = f_news.result()
                financials = f_fins.result()

            # Merge yfinance supplemental fields into Finnhub profile
            try:
                # Finnhub's profile gives market cap in millions; the page (and the
                # Canadian branch's yfinance figure) use dollars
                company['marketCapitalization'] = (info.get('marketCap')
                                                   or (company.get('marketCapitalization') or 0) * 1e6)
                company['longBusinessSummary'] = info.get('longBusinessSummary', '')
                company['sector'] = info.get('sector', company.get('finnhubIndustry', ''))
                company['fullTimeEmployees'] = info.get('fullTimeEmployees', 0)
                company['city'] = info.get('city', '')
                company['state'] = info.get('state', '')
                company['country_full'] = info.get('country', '')
                company['fiftyTwoWeekHigh'] = info.get('fiftyTwoWeekHigh', 0)
                company['fiftyTwoWeekLow'] = info.get('fiftyTwoWeekLow', 0)
                company['dividendYield'] = info.get('dividendYield', 0)
                company['trailingPE'] = info.get('trailingPE', 0)
                company['forwardPE'] = info.get('forwardPE', 0)
                company['priceToBook'] = info.get('priceToBook', 0)
                company['beta'] = info.get('beta', 0)
            except Exception as e:
                print(f"Could not merge yfinance data for {symbol_upper}: {e}")

        # yfinance news fallback (if Finnhub returned nothing)
        if not news or len(news) == 0:
            try:
                news_symbol = get_ticker_for_charts(symbol_upper) if is_canadian_stock(symbol_upper) else symbol_upper
                yf_news = yf.Ticker(news_symbol).news
                if yf_news:
                    news = []
                    for item in yf_news[:10]:
                        content      = item.get('content', {})
                        provider     = content.get('provider', {})
                        canonical_url = content.get('canonicalUrl', {})
                        pub_date     = content.get('pubDate', '')
                        timestamp    = 0
                        if pub_date:
                            try:
                                from dateutil import parser
                                timestamp = int(parser.parse(pub_date).timestamp())
                            except Exception:
                                pass
                        news.append({
                            'headline': content.get('title', 'N/A'),
                            'summary': (content.get('summary', '') or '')[:300] or 'No summary available',
                            'url': canonical_url.get('url', ''),
                            'datetime': timestamp,
                            'source': provider.get('displayName', 'Yahoo Finance')
                        })
            except Exception as e:
                print(f"yfinance news fallback error for {symbol}: {e}")

        return jsonify({
            'success': True,
            'symbol': symbol_upper,
            'is_canadian': is_canadian_stock(symbol_upper),
            'currency': 'CAD' if is_canadian_stock(symbol_upper) else 'USD',
            'company': company,
            'quote': quote,
            'news': news[:10] if news else [],
            'financials': financials
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@bp.route('/api/ai-overview/<symbol>')
def get_ai_overview(symbol):
    """Get AI-generated company overview (loads separately for speed)

    TEMPORARILY DISABLED (NVIDIA API too slow/unreliable for the request path).
    Returns immediately — no upstream calls — so the UI never waits on a
    feature that is off. To re-enable: fetch the profile name and call
    get_company_overview_llm(company_name, symbol) here.
    """
    return jsonify({
        'success': True,
        'symbol': symbol.upper(),
        'ai_overview': None,  # Disabled for speed
        'message': 'AI overview is currently disabled.'
    })

@bp.route('/api/statistics/<symbol>')
def get_statistics(symbol):
    """Get comprehensive company statistics"""
    try:
        symbol_upper = symbol.upper()
        print(f"📊 Fetching statistics for {symbol_upper}...")

        # Fetch both upstreams in parallel through the shared TTL caches
        # (yf info: 5 min, Finnhub financials: 10 min) — the Overview tab
        # usually warmed these already, making this a cache hit.
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_info = ex.submit(_safe, _cached_yf_info, symbol_upper)
            f_fins = ex.submit(_safe, _cached_get_basic_financials, symbol_upper)
            info = f_info.result()
            fins = f_fins.result()

        # Get raw statistics
        stats = get_company_statistics(symbol_upper, info=info, financials=fins)

        # Format for display
        formatted_stats = format_statistics_for_display(stats)
        
        return jsonify({
            'success': True,
            'symbol': symbol.upper(),
            'statistics': formatted_stats,
            'raw_statistics': stats  # Include raw data for calculations
        })
        
    except Exception as e:
        print(f"❌ Error fetching statistics for {symbol}: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'statistics': {}
        })

@bp.route('/api/sentiment/news/<symbol>')
@limiter.limit(AV_LIMIT, exempt_when=bypass_ok)
def sentiment_news(symbol):
    """Sentiment tab: Alpha Vantage-scored articles + Finnhub headlines, a
    relevance-weighted overall score and a daily trend (sentiment_feed.py).

    Query: days — lookback in calendar days (default 30, max 365).
    At most one Alpha Vantage and one Finnhub call per symbol/window per 30 min.
    """
    try:
        import sentiment_feed
        days = _int_arg('days', 30, 365)
        payload = sentiment_feed.get_feed(symbol.upper(), days, av, _cached_get_company_news)
        return jsonify({'success': True, **payload})
    except Exception as e:
        print(f"[SENTIMENT] {symbol}: {e!r}", flush=True)
        return jsonify({'success': False, 'error': 'Sentiment is unavailable right now.'}), 500


@bp.route('/api/scenarios/<symbol>')
def scenario_analysis(symbol):
    """Bull/base/bear scenarios — numbers from scenario_engine (vol-scaled
    statistical, or the ML registry when the ticker has trained models);
    rationale text from the cached LLM analyst brief when available."""
    try:
        import scenario_engine
        sc = scenario_engine.compute_scenarios(symbol, request.args.get('timeframe', '1M'))
        timeframe = sc['timeframe']               # unknown values fall back to 1M

        # LLM narrative is additive: use the cached brief if present; a miss
        # falls back to honest generated-from-numbers text (no LLM call here —
        # the Signal Brief panel warms the cache).
        rationale = {'bull': None, 'base': None, 'bear': None}
        factors = []
        try:
            import llm_analyst
            with llm_analyst._lock:
                brief = llm_analyst._brief_cache.get(symbol.upper())
            if brief:
                # the brief is written against the 1-month targets; its prose
                # quotes those numbers, so it only fits the 1M cards
                if timeframe == '1M':
                    rationale = brief.get('scenario_rationale') or rationale
                factors = {'bull': brief.get('bull_factors') or [],
                           'bear': brief.get('bear_factors') or []}
        except Exception:
            pass

        if not isinstance(rationale, dict):
            rationale = {}
        default_txt = {
            'bull': f"1-in-5 chance the price ends above this after {sc['horizon_days']} trading days",
            'base': f"Median outcome after {sc['horizon_days']} trading days",
            'bear': f"1-in-5 chance the price ends below this after {sc['horizon_days']} trading days",
        }

        def case(name, factor_key):
            c = sc['scenarios'][f'{name}_case']
            text = rationale.get(name)
            listed = factors.get(factor_key) if factor_key and isinstance(factors, dict) else None
            return {
                'price_target': c['price_target'],
                'probability': c['probability'],
                'return': c['return'],
                'factors': [f for f in listed if isinstance(f, str)] if isinstance(listed, list) else [],
                'rationale': text if isinstance(text, str) and text else (
                    f"{default_txt[name]} ({sc['engine']}, {sc['annualized_vol']:.0%} "
                    f"annualized vol over the horizon)."),
            }

        return jsonify({
            'success': True,
            'symbol': sc['symbol'],
            'current_price': sc['current_price'],
            'currency': sc['currency'],
            'timeframe': timeframe,
            'engine': sc['engine'],
            'probability_method': sc['probability_method'],
            'prediction_date': sc.get('prediction_date'),
            'model_backed': sc['model_backed'],
            'model_note': sc.get('model_note'),
            'annualized_vol': sc['annualized_vol'],
            'p_up': sc['p_up'],
            'p_loss_10': sc['p_loss_10'],
            'range_90': sc['range_90'],
            'calibration': sc['calibration'],
            'vol_now_annualized': sc['vol_now_annualized'],
            'vol_longrun_annualized': sc['vol_longrun_annualized'],
            'drift_annualized': sc['drift_annualized'],
            'data_sources': ['yfinance history', sc['engine']],
            'bull_case': case('bull', 'bull'),
            'base_case': case('base', None),
            'bear_case': case('bear', 'bear'),
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Finnhub's free tier allows 60 calls/min; a cold request makes 3 + MAX_PEERS.
@bp.route('/api/metrics/<symbol>')
def stock_metrics(symbol):
    """Fundamental grades relative to the company's Finnhub industry peers."""
    try:
        symbol_upper = symbol.upper()
        metrics_data, industry, peers = graded_fundamentals(symbol_upper)
        if not metrics_data['graded_categories']:
            return jsonify({'success': False, 'error':
                            f'Not enough fundamental data to grade {symbol_upper} '
                            '(ETFs and funds are not graded).'})

        return jsonify({
            'success': True,
            'symbol': symbol_upper,
            'industry': industry,
            'peers': peers,
            **metrics_data
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@bp.route('/api/recommendations/<symbol>')
def recommendations(symbol):
    """Per-horizon Buy/Hold/Sell from a weighted vote of trend, momentum, the ML
    model and the quality grade, with each price rule backtested on this ticker
    (recommendation_engine.py). The cached AI brief, when present, is shown once
    as the 1-month view."""
    try:
        import recommendation_engine
        rec = recommendation_engine.recommend(symbol)
        brief = None
        try:
            import llm_analyst
            with llm_analyst._lock:
                brief = llm_analyst._brief_cache.get(symbol.upper())
        except Exception:
            pass
        return jsonify({'success': True, **rec,
                        'ai_brief': {'signal': brief.get('signal'), 'narrative': brief.get('narrative')}
                        if brief else None})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/api/narrate/<symbol>')
@limiter.limit(LLM_LIMIT, exempt_when=bypass_ok)
def narrate(symbol):
    """AI Signal Brief — LLM narrative over computed numbers (llm_analyst).
    The frontend panel has called this endpoint since the UI was built; it now
    exists. First call per symbol/hour does one LLM round-trip; then cached."""
    try:
        import llm_analyst
        brief = llm_analyst.generate_brief(symbol.upper())
        if brief is None:
            return jsonify({'success': False,
                            'error': 'No LLM provider available or brief generation failed'}), 503
        return jsonify({
            'success': True,
            'symbol': symbol.upper(),
            'signal': brief.get('signal', 'neutral'),
            'narrative': brief.get('narrative', ''),
            'competitive_position': brief.get('competitive_position'),
            'bull_factors': brief.get('bull_factors', []),
            'bear_factors': brief.get('bear_factors', []),
            'risk_flags': brief.get('risk_flags', []),
            'provider': brief.get('provider'),
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

