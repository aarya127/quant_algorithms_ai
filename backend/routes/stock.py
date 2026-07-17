"""
routes/stock.py — per-symbol detail (stock, ai-overview, statistics, sentiment, scenarios, metrics, recommendations) blueprint, extracted verbatim from app.py.
"""
import os
import sys
import json
import datetime

from flask import Blueprint, jsonify, request

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
    """Get AI-generated company overview (loads separately for speed)"""
    try:
        # TEMPORARILY DISABLED: NVIDIA API is too slow/unreliable
        # Uncomment below to re-enable AI overviews
        
        # Get company profile to get the full name
        company = get_company_profile(symbol.upper())
        company_name = company.get('name', symbol.upper()) if company else symbol.upper()
        
        # Generate AI overview (DISABLED - uncomment to enable)
        # ai_overview = get_company_overview_llm(company_name, symbol.upper())
        
        # Return basic company description instead of AI overview
        return jsonify({
            'success': True,
            'symbol': symbol.upper(),
            'ai_overview': None,  # Disabled for speed
            'message': 'AI overview feature temporarily disabled for faster loading. Enable in app.py if needed.'
        })
        
    except Exception as e:
        print(f"❌ Error generating AI overview for {symbol}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
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
    """Generate bull/base/bear scenarios using comprehensive data"""
    try:
        timeframe = request.args.get('timeframe', '1M')
        
        analyzer = StockAnalyzer(symbol.upper())
        scenarios_data = analyzer.get_enhanced_scenarios(timeframe)
        
        # Format for frontend
        scenarios = scenarios_data.get('scenarios', {})
        
        return jsonify({
            'success': True,
            'symbol': symbol.upper(),
            'current_price': scenarios_data.get('current_price', 0),
            'timeframe': timeframe,
            'sentiment_score': scenarios_data.get('sentiment_score', 50),
            'eps_growth': scenarios_data.get('eps_growth', 10),
            'data_sources': scenarios_data.get('data_sources', []),
            'bull_case': {
                'price_target': scenarios['bull_case']['price_target'],
                'probability': scenarios['bull_case']['probability'],
                'return': scenarios['bull_case']['return'],
                'factors': scenarios['bull_case']['factors'],
                'rationale': scenarios['bull_case']['rationale']
            },
            'base_case': {
                'price_target': scenarios['base_case']['price_target'],
                'probability': scenarios['base_case']['probability'],
                'return': scenarios['base_case']['return'],
                'factors': scenarios['base_case']['factors'],
                'rationale': scenarios['base_case']['rationale']
            },
            'bear_case': {
                'price_target': scenarios['bear_case']['price_target'],
                'probability': scenarios['bear_case']['probability'],
                'return': scenarios['bear_case']['return'],
                'factors': scenarios['bear_case']['factors'],
                'rationale': scenarios['bear_case']['rationale']
            }
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
    """Get time-based recommendations"""
    try:
        analyzer = StockAnalyzer(symbol.upper())
        comprehensive_data = analyzer.get_comprehensive_data()
        metrics = analyzer.get_detailed_metrics()
        
        # Get sentiment and metrics for recommendations
        sentiment = comprehensive_data.get('sentiment', {}).get('consensus', {})
        sentiment_score = sentiment.get('score', 50)
        overall_grade = metrics.get('overall_grade', 'C')
        
        # Generate recommendations for each timeframe
        recommendations_data = {}
        
        timeframes = [
            ('1W', 'Short-term Trading'),
            ('1M', 'Swing Trading'),
            ('3M', 'Medium-term Investment'),
            ('6M', 'Position Trading'),
            ('1Y', 'Long-term Investment')
        ]
        
        for tf, description in timeframes:
            scenarios = analyzer.get_enhanced_scenarios(tf)
            base_return = scenarios['scenarios']['base_case']['return']
            
            # Decision logic based on multiple factors
            if sentiment_score > 65 and overall_grade in ['A', 'B'] and base_return > 10:
                action = 'Strong Buy'
                confidence = 0.85
            elif sentiment_score > 55 and overall_grade in ['A', 'B', 'C'] and base_return > 5:
                action = 'Buy'
                confidence = 0.70
            elif sentiment_score > 45 and base_return > 0:
                action = 'Hold'
                confidence = 0.60
            elif sentiment_score < 40 or base_return < -5:
                action = 'Sell'
                confidence = 0.75
            elif sentiment_score < 30:
                action = 'Strong Sell'
                confidence = 0.80
            else:
                action = 'Hold'
                confidence = 0.50
            
            reasoning = f"{description}: Based on {sentiment.get('sentiment', 'neutral')} sentiment (score: {sentiment_score:.1f}/100), " \
                       f"grade {overall_grade}, and projected {base_return:.1f}% return in {tf}. "
            
            if action in ['Strong Buy', 'Buy']:
                reasoning += f"Positive outlook supported by strong fundamentals and favorable sentiment."
            elif action == 'Hold':
                reasoning += f"Mixed signals suggest maintaining current position."
            else:
                reasoning += f"Concerns about fundamentals or market sentiment warrant caution."
            
            recommendations_data[tf] = {
                'action': action,
                'confidence': confidence,
                'reasoning': reasoning,
                'timeframe': description,
                'expected_return': base_return
            }
        
        return jsonify({
            'success': True,
            'symbol': symbol.upper(),
            'recommendations': recommendations_data,
            'sentiment_score': sentiment_score,
            'overall_grade': overall_grade
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

