"""
routes/stock.py — per-symbol detail (stock, ai-overview, statistics, sentiment, scenarios, metrics, recommendations) blueprint, extracted verbatim from app.py.
"""
import os
import sys
import json
import datetime

from concurrent.futures import ThreadPoolExecutor

from flask import Blueprint, jsonify, request

from common import get_ticker_for_charts, is_canadian_stock
from services import (
    yf, StockAnalyzer, get_company_overview_llm,
    get_company_profile, get_stock_quote, get_basic_financials, get_company_news,
    get_company_statistics, format_statistics_for_display,
    _cached_get_stock_quote, _cached_get_company_profile,
    _cached_get_basic_financials, _cached_get_company_news, _cached_yf_info,
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
        print(f"📊 Fetching statistics for {symbol.upper()}...")
        
        # Get raw statistics
        stats = get_company_statistics(symbol.upper())
        
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

@bp.route('/api/sentiment/<symbol>')
def sentiment_analysis(symbol):
    """Get comprehensive sentiment analysis from multiple sources"""
    try:
        analyzer = StockAnalyzer(symbol.upper())
        sentiment_data = analyzer.get_comprehensive_sentiment()
        
        # Format for frontend
        sources = sentiment_data.get('sources', {})
        comparison = sentiment_data.get('comparison', {})
        consensus = sentiment_data.get('consensus', {})
        
        # Build response
        response = {
            'success': True,
            'symbol': symbol.upper(),
            'timestamp': sentiment_data.get('timestamp'),
            
            # Overall consensus
            'overall_sentiment': consensus.get('sentiment', 'neutral'),
            'confidence': consensus.get('confidence', 0),
            'consensus_score': consensus.get('score', 50),
            
            # Individual sources
            'sources': [],
            
            # Comparison metrics
            'agreement_level': comparison.get('agreement_level', 'unknown'),
            'sentiment_consensus': comparison.get('sentiment_consensus', 'unknown'),
            'average_score': comparison.get('average_score', 50),
            'score_variance': comparison.get('variance', 0),
            
            # Summary
            'summary': f"Analysis from {len(sources)} sources shows {consensus.get('sentiment', 'neutral')} sentiment "
                      f"with {comparison.get('agreement_level', 'unknown')} agreement across providers."
        }
        
        # Add detailed source information
        for source_name, source_data in sources.items():
            if 'error' not in source_data:
                source_info = {
                    'name': source_name,
                    'provider': source_data.get('provider', source_name),
                    'sentiment': source_data.get('overall_sentiment', 'unknown'),
                    'score': source_data.get('score', 50),
                    'articles_analyzed': source_data.get('articles_analyzed', 0)
                }
                
                # Add source-specific metrics
                if 'confidence' in source_data:
                    source_info['confidence'] = source_data['confidence']
                if 'positive_ratio' in source_data:
                    source_info['positive_ratio'] = source_data['positive_ratio']
                    source_info['negative_ratio'] = source_data['negative_ratio']
                    source_info['neutral_ratio'] = source_data['neutral_ratio']
                if 'sentiment_score' in source_data:
                    source_info['raw_score'] = source_data['sentiment_score']
                if 'mspr' in source_data:
                    source_info['mspr'] = source_data['mspr']
                    source_info['insider_signal'] = 'bullish' if source_data['mspr'] > 0 else 'bearish'
                
                response['sources'].append(source_info)
        
        # Calculate ratios for visualization
        total = len(response['sources'])
        if total > 0:
            pos = sum(1 for s in response['sources'] if s['sentiment'] == 'positive')
            neg = sum(1 for s in response['sources'] if s['sentiment'] == 'negative')
            neu = sum(1 for s in response['sources'] if s['sentiment'] == 'neutral')
            
            response['positive_ratio'] = pos / total
            response['negative_ratio'] = neg / total
            response['neutral_ratio'] = neu / total
            response['articles_analyzed'] = sum(s.get('articles_analyzed', 0) for s in response['sources'])
        else:
            response['positive_ratio'] = 0
            response['negative_ratio'] = 0
            response['neutral_ratio'] = 0
            response['articles_analyzed'] = 0
        
        return jsonify(response)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@bp.route('/api/sentiment/news/<symbol>')
def sentiment_news(symbol):
    """
    Article-level sentiment feed + daily trend for the Sentiment tab.

    Query params:
        days  — lookback window in calendar days (default 30, max 365)

    Returns up to 150 articles (AV + Finnhub) with pre-computed scores,
    plus a daily aggregate series for charting.
    API budget: 1 AV call (25/day) + 1 Finnhub call (60/min).
    """
    try:
        sys.path.insert(0, os.path.join(_BACKEND, '..'))
        from algorithms.machine_learning_algorithms.data_pipelines.extractor import DataExtractor

        sym      = symbol.upper()
        days     = min(int(request.args.get('days', 30)), 365)
        end      = datetime.date.today().isoformat()
        start    = (datetime.date.today() - datetime.timedelta(days=days)).isoformat()

        ex = DataExtractor()

        # Article-level feed (AV scored + Finnhub unscored); cap conservatively to stay within timeout
        max_art = 40 if days > 30 else 25
        art_df = ex.news_with_scores(sym, start=start, end=end, max_articles=max_art)

        # Daily aggregate trend (uses the same sources + weights)
        daily_df = ex.news_sentiment_history(sym, start=start, end=end, use_finbert=False)

        # --- Build articles list ---
        articles = []
        if not art_df.empty:
            for _, r in art_df.iterrows():
                score = r.get("sent_score")
                score_val = float(score) if score == score else None  # NaN → None
                articles.append({
                    "date":      str(r["date"])[:10],
                    "headline":  str(r.get("headline", "")),
                    "source":    str(r.get("source", "")),
                    "provider":  str(r.get("provider", "")),
                    "score":     score_val,
                    "label":     str(r.get("label", "neutral")),
                    "url":       str(r.get("url", "") or ""),
                })

        # --- Build daily trend ---
        daily = []
        if not daily_df.empty:
            for dt, row in daily_df.iterrows():
                sc = row.get("news_sent_score")
                daily.append({
                    "date":  str(dt)[:10],
                    "score": float(sc) if sc == sc else None,
                    "count": int(row.get("news_articles", 0)),
                })

        # --- Overall summary (from AV-scored articles only) ---
        scored = [a for a in articles if a["score"] is not None]
        if scored:
            mean_score = sum(a["score"] for a in scored) / len(scored)
            overall_label = "positive" if mean_score > 0.05 else ("negative" if mean_score < -0.05 else "neutral")
            pos = sum(1 for a in scored if a["label"] == "positive")
            neg = sum(1 for a in scored if a["label"] == "negative")
            neu = sum(1 for a in scored if a["label"] == "neutral")
        else:
            mean_score = 0.0
            overall_label = "neutral"
            pos = neg = neu = 0

        source_counts = {}
        for a in articles:
            source_counts[a["provider"]] = source_counts.get(a["provider"], 0) + 1

        return jsonify({
            "success":        True,
            "symbol":         sym,
            "window_days":    days,
            "overall_score":  round(mean_score, 3),
            "overall_label":  overall_label,
            "articles_total": len(articles),
            "scored_total":   len(scored),
            "positive_count": pos,
            "negative_count": neg,
            "neutral_count":  neu,
            "source_counts":  source_counts,
            "daily":          daily,
            "articles":       articles,
        })
    except Exception as exc:
        import traceback
        return jsonify({"success": False, "error": str(exc), "traceback": traceback.format_exc()})


@bp.route('/api/scenarios/<symbol>')
def scenario_analysis(symbol):
    """Bull/base/bear scenarios — numbers from scenario_engine (vol-scaled
    statistical, or the ML registry when the ticker has trained models);
    rationale text from the cached LLM analyst brief when available."""
    try:
        import scenario_engine
        timeframe = request.args.get('timeframe', '1M')
        sc = scenario_engine.compute_scenarios(symbol, timeframe)

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
                rationale = brief.get('scenario_rationale') or rationale
                factors = {'bull': brief.get('bull_factors') or [],
                           'bear': brief.get('bear_factors') or []}
        except Exception:
            pass

        def case(name, up_factors):
            c = sc['scenarios'][f'{name}_case']
            default_txt = (f"{name.title()} case: ±1σ over {sc['horizon_days']} trading days "
                           f"at {sc['annualized_vol']:.0%} annualized vol "
                           f"({sc['engine']}); probability from this ticker's own "
                           f"return distribution.")
            return {
                'price_target': c['price_target'],
                'probability': c['probability'],
                'return': c['return'],
                'factors': (factors.get(up_factors) if isinstance(factors, dict) else None) or [],
                'rationale': (rationale.get(name) or default_txt),
            }

        return jsonify({
            'success': True,
            'symbol': sc['symbol'],
            'current_price': sc['current_price'],
            'timeframe': timeframe,
            'engine': sc['engine'],
            'model_backed': sc['model_backed'],
            'annualized_vol': sc['annualized_vol'],
            'p_up': sc['p_up'],
            'data_sources': ['yfinance history', sc['engine']],
            'bull_case': case('bull', 'bull'),
            'base_case': case('base', 'bull'),
            'bear_case': case('bear', 'bear'),
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@bp.route('/api/metrics/<symbol>')
def stock_metrics(symbol):
    """Get comprehensive metrics and grading using StockAnalyzer"""
    try:
        analyzer = StockAnalyzer(symbol.upper())
        metrics_data = analyzer.get_detailed_metrics()
        
        return jsonify({
            'success': True,
            **metrics_data
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@bp.route('/api/recommendations/<symbol>')
def recommendations(symbol):
    """Time-based recommendations driven by scenario_engine numbers (empirical
    p_up + expected return), with LLM narrative when the brief is cached.
    Every recommendation is logged so its hit rate can be measured later."""
    try:
        import scenario_engine
        symbol_u = symbol.upper()

        brief = None
        try:
            import llm_analyst
            with llm_analyst._lock:
                brief = llm_analyst._brief_cache.get(symbol_u)
        except Exception:
            pass

        timeframes = [
            ('1W', 'Short-term Trading'),
            ('1M', 'Swing Trading'),
            ('3M', 'Medium-term Investment'),
            ('6M', 'Position Trading'),
            ('1Y', 'Long-term Investment'),
        ]

        recommendations_data = {}
        current_price = None
        for tf, description in timeframes:
            sc = scenario_engine.compute_scenarios(symbol_u, tf)
            current_price = sc['current_price']
            base_return = sc['scenarios']['base_case']['return']
            p_up = sc['p_up']

            # Action from the empirical odds; confidence = distance from a coin
            # flip (calibrated to the ticker's own return history, not vibes).
            if p_up >= 0.62 and base_return > 2:
                action = 'Strong Buy'
            elif p_up >= 0.55 and base_return > 0:
                action = 'Buy'
            elif p_up <= 0.38 or base_return < -5:
                action = 'Strong Sell' if p_up <= 0.33 else 'Sell'
            else:
                action = 'Hold'
            confidence = round(0.5 + abs(p_up - 0.5), 2)

            reasoning = (f"{description}: {p_up:.0%} of comparable {sc['horizon_days']}-day "
                         f"periods finished positive ({sc['engine']}); base case "
                         f"{base_return:+.1f}%.")
            if brief and brief.get('narrative'):
                reasoning += f" Analyst view: {brief['narrative']}"

            recommendations_data[tf] = {
                'action': action,
                'confidence': confidence,
                'reasoning': reasoning,
                'timeframe': description,
                'expected_return': base_return,
                'p_up': p_up,
                'model_backed': sc['model_backed'],
            }

        # Audit trail — measurable hit rate later (price at recommendation time)
        try:
            import datetime as _dt, json as _json
            log_path = os.path.join(_BACKEND, 'recommendation_log.jsonl')
            with open(log_path, 'a') as fh:
                fh.write(_json.dumps({
                    'ts': _dt.datetime.now().isoformat(),
                    'symbol': symbol_u, 'price': current_price,
                    'recommendations': {tf: {k: r[k] for k in ('action', 'p_up', 'expected_return')}
                                        for tf, r in recommendations_data.items()},
                }) + '\n')
        except OSError:
            pass

        return jsonify({
            'success': True,
            'symbol': symbol_u,
            'recommendations': recommendations_data,
            'signal': (brief or {}).get('signal'),
            'engine_note': 'probabilities from empirical return distribution; '
                           'ML-registry drift/vol where a trained model exists',
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/api/narrate/<symbol>')
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

