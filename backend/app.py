"""
Invest.ai Web Application
Comprehensive UI for stock analysis and predictions
"""

import sys
import os
import datetime
import json
import subprocess
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, render_template, jsonify, request, send_file, Response
from cachetools import TTLCache

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Create the Flask app first — this guarantees /health always responds even if
# downstream data-module imports fail, preventing Railway healthcheck failures.
app = Flask(__name__)

# ---------------------------------------------------------------------------
# TTL caches — keyed by symbol, thread-safe for concurrent requests.
# Quotes: 30s  |  Profiles: 1h  |  Financials/news: 10min
# ---------------------------------------------------------------------------
_cache_lock = __import__('threading').Lock()
_quote_cache    = TTLCache(maxsize=256, ttl=30)
_profile_cache  = TTLCache(maxsize=256, ttl=3600)
_fin_cache      = TTLCache(maxsize=256, ttl=600)
_news_cache     = TTLCache(maxsize=256, ttl=600)
_yf_info_cache  = TTLCache(maxsize=256, ttl=300)

def _cached_get_stock_quote(symbol):
    with _cache_lock:
        if symbol in _quote_cache:
            return _quote_cache[symbol]
    result = get_stock_quote(symbol)
    with _cache_lock:
        _quote_cache[symbol] = result
    return result

def _cached_get_company_profile(symbol):
    with _cache_lock:
        if symbol in _profile_cache:
            return _profile_cache[symbol]
    result = get_company_profile(symbol)
    with _cache_lock:
        _profile_cache[symbol] = result
    return result

def _cached_get_basic_financials(symbol):
    with _cache_lock:
        if symbol in _fin_cache:
            return _fin_cache[symbol]
    result = get_basic_financials(symbol)
    with _cache_lock:
        _fin_cache[symbol] = result
    return result

def _cached_get_company_news(symbol, from_date, to_date):
    key = f"{symbol}:{from_date}:{to_date}"
    with _cache_lock:
        if key in _news_cache:
            return _news_cache[key]
    result = get_company_news(symbol, from_date, to_date)
    with _cache_lock:
        _news_cache[key] = result
    return result

def _cached_yf_info(ticker_symbol):
    with _cache_lock:
        if ticker_symbol in _yf_info_cache:
            return _yf_info_cache[ticker_symbol]
    result = yf.Ticker(ticker_symbol).info
    with _cache_lock:
        _yf_info_cache[ticker_symbol] = result
    return result

# ---------------------------------------------------------------------------
# Data module imports — wrapped so a single bad import never crashes the server.
# If something fails the error is printed to Railway's deploy logs, the server
# still starts, the healthcheck passes, and affected API endpoints return 500.
# ---------------------------------------------------------------------------
_DATA_MODULES_LOADED = False
av = None
yf = None

try:
    import yfinance as yf
    from data.finnhub import (
        get_company_news,
        get_basic_financials,
        get_earnings_surprises,
        get_insider_transactions,
        get_insider_sentiment,
        get_earnings_calendar,
        get_company_profile,
        get_stock_quote
    )
    from data.alphavantage import AlphaVantage
    from data.nvidia_llm import get_company_overview_llm
    from data.charts import get_chart_data, get_multiple_timeframes, get_comparison_data, get_technical_indicators
    from data.twitter_feed import get_market_tweets, get_financial_news_feed
    from data.alpaca_news import get_recent_news, start_news_stream, stop_news_stream
    from data.company_statistics import get_company_statistics, format_statistics_for_display
    from stock_analyzer import StockAnalyzer
    av = AlphaVantage()
    _DATA_MODULES_LOADED = True
    print("[STARTUP] All data modules loaded successfully.", flush=True)
except Exception as _import_err:
    import traceback as _tb
    print(f"[STARTUP ERROR] One or more data modules failed to import:\n{_tb.format_exc()}", flush=True)

# Configuration
DEFAULT_STOCKS = ["NVDA", "TD", "ACDVF", "MSFT", "ENB", "RCI", "CVE", "HUBS", "MU", "CNSWF", "AMD"]

# Map US tickers to TSX equivalents for Canadian stocks (to get CAD prices)
CANADIAN_STOCKS_MAP = {
    'TD': 'TD.TO',
    'ACDVF': 'AC.TO',  # Air Canada
    'ENB': 'ENB.TO',
    'RCI': 'RCI-B.TO',  # Rogers Class B
    'CVE': 'CVE.TO',
    'CNSWF': 'CSU.TO'  # Constellation Software
}

TIMEFRAMES = {
    '1W': {'days': 7, 'label': '1 Week'},
    '1M': {'days': 30, 'label': '1 Month'},
    '3M': {'days': 90, 'label': '3 Months'},
    '6M': {'days': 180, 'label': '6 Months'},
    '1Y': {'days': 365, 'label': '1 Year'},
}

def get_ticker_for_charts(symbol):
    """Get the appropriate ticker symbol for charts (TSX for Canadian stocks)"""
    return CANADIAN_STOCKS_MAP.get(symbol, symbol)

def is_canadian_stock(symbol):
    """Check if a stock is Canadian"""
    return symbol in CANADIAN_STOCKS_MAP

@app.route('/health')
def health():
    """Railway healthcheck endpoint — always returns 200 immediately"""
    return jsonify({'status': 'ok'}), 200

@app.route('/')
def index():
    """Main dashboard"""
    resp = app.make_response(render_template('index.html', default_stocks=DEFAULT_STOCKS))
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    return resp

@app.route('/api/search/<query>')
def search_stocks(query):
    """Search for stocks by symbol or company name - Fast version with minimal API calls"""
    try:
        query = query.upper().strip()
        results = []
        
        # Use yfinance's fast search functionality
        # This uses Yahoo's search API which is much faster than fetching individual tickers
        try:
            import yfinance.utils as yf_utils
            
            # Search for tickers matching the query
            search_results = yf.Ticker(query).get_fast_info()
            
            # For the primary symbol, get basic info
            stock = yf.Ticker(query)
            
            # Use fast_info instead of info to avoid slow API calls
            fast_info = stock.fast_info
            
            if fast_info:
                # Get additional info only for the exact match (much faster)
                info = stock.info
                if info and info.get('symbol'):
                    exchange = info.get('exchange', '')
                    # Filter for major exchanges
                    if exchange in ['NYQ', 'NMS', 'NAS', 'NYSE', 'NASDAQ', 'TOR', 'TSX', 'TSE', 'NEO', 'CNQ'] or not exchange:
                        results.append({
                            'symbol': info.get('symbol', query),
                            'name': info.get('longName', info.get('shortName', query)),
                            'exchange': exchange or 'Unknown',
                            'type': info.get('quoteType', 'EQUITY'),
                            'currency': info.get('currency', 'USD'),
                            'sector': info.get('sector', ''),
                            'industry': info.get('industry', '')
                        })
        except:
            # Fallback: just try the exact symbol with minimal info
            try:
                stock = yf.Ticker(query)
                info = stock.info
                
                if info and info.get('symbol'):
                    results.append({
                        'symbol': info.get('symbol', query),
                        'name': info.get('longName', info.get('shortName', query)),
                        'exchange': info.get('exchange', 'Unknown'),
                        'type': info.get('quoteType', 'EQUITY'),
                        'currency': info.get('currency', 'USD'),
                        'sector': info.get('sector', ''),
                        'industry': info.get('industry', '')
                    })
            except:
                pass
        
        # Only try TSX variant if no results and query doesn't already have a suffix
        if not results and '.' not in query:
            try:
                tsx_symbol = f"{query}.TO"
                stock = yf.Ticker(tsx_symbol)
                info = stock.info
                
                if info and info.get('symbol'):
                    results.append({
                        'symbol': info.get('symbol', tsx_symbol),
                        'name': info.get('longName', info.get('shortName', tsx_symbol)),
                        'exchange': info.get('exchange', 'TSX'),
                        'type': info.get('quoteType', 'EQUITY'),
                        'currency': info.get('currency', 'CAD'),
                        'sector': info.get('sector', ''),
                        'industry': info.get('industry', '')
                    })
            except:
                pass
        
        return jsonify({
            'success': True,
            'results': results,
            'query': query
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'results': []})

@app.route('/api/indices')
def market_indices():
    """Get live market indices: S&P 500, NASDAQ, TSX, Dow, Russell 2000, VIX"""
    indices = [
        ('^GSPC',   'S&P 500'),
        ('^IXIC',   'NASDAQ'),
        ('^DJI',    'Dow Jones'),
        ('^GSPTSE', 'TSX'),
        ('^RUT',    'Russell 2000'),
        ('^VIX',    'VIX'),
    ]
    result = []
    for symbol, name in indices:
        try:
            fi = yf.Ticker(symbol).fast_info
            price = fi.last_price
            prev  = fi.previous_close
            change = round(price - prev, 2)
            pct    = round((change / prev) * 100, 2) if prev else 0
            result.append({
                'symbol': symbol,
                'name': name,
                'price': round(price, 2),
                'change': change,
                'pct_change': pct,
            })
        except Exception:
            pass
    return jsonify({'success': True, 'indices': result})

@app.route('/api/indices/history')
def indices_history():
    """Return 30-day daily closes for each index for sparkline charts"""
    symbols = ['^GSPC', '^IXIC', '^DJI', '^GSPTSE', '^RUT', '^VIX']
    result = {}
    for symbol in symbols:
        try:
            df = yf.Ticker(symbol).history(period='1mo', interval='1d')
            if not df.empty:
                result[symbol] = {
                    'labels': [d.strftime('%b %d') for d in df.index],
                    'closes': [round(float(v), 2) for v in df['Close']],
                }
        except Exception:
            pass
    return jsonify({'success': True, 'history': result})

@app.route('/api/dashboard')
def dashboard_data():
    """Get dashboard overview data"""
    try:
        today = datetime.date.today()
        from_date = (today - datetime.timedelta(days=7)).strftime('%Y-%m-%d')
        to_date = today.strftime('%Y-%m-%d')
        
        # Get trending news
        trending = []
        for symbol in DEFAULT_STOCKS[:3]:
            news = get_company_news(symbol, from_date, to_date)
            if news and len(news) > 0:
                article = news[0]
                trending.append({
                    'symbol': symbol,
                    'headline': article.get('headline', ''),
                    'summary': article.get('summary', '')[:200] + '...',
                    'url': article.get('url', ''),
                    'datetime': article.get('datetime', 0)
                })
        
        return jsonify({
            'success': True,
            'trending': trending,
            'market_status': 'Open' if 9 <= datetime.datetime.now().hour < 16 else 'Closed',
            'timestamp': datetime.datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/stock/<symbol>')
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

@app.route('/api/ai-overview/<symbol>')
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

@app.route('/api/backtest', methods=['POST'])
def run_backtest_api():
    """Run a real options volatility backtest via VolatilityStrategyBacktestEngine."""
    try:
        from backtest_runner import run_backtest, STRATEGIES
        body = request.get_json(force=True) or {}

        strategy   = body.get('strategy', '')
        asset      = body.get('asset', '').upper().strip()
        start_date = body.get('start_date', '')
        end_date   = body.get('end_date', '')
        capital    = float(body.get('capital', 250_000))

        if not strategy or strategy not in STRATEGIES:
            return jsonify({'success': False,
                            'error': f"Invalid strategy. Choose from: {list(STRATEGIES.keys())}"}), 400
        if not asset:
            return jsonify({'success': False, 'error': 'Asset symbol is required'}), 400
        if not start_date or not end_date:
            return jsonify({'success': False, 'error': 'start_date and end_date are required'}), 400

        result = run_backtest(strategy=strategy, asset=asset,
                              start_date=start_date, end_date=end_date, capital=capital)
        return jsonify(result)

    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        import traceback
        print(f"[BACKTEST ERROR] {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/backtest/strategies')
def backtest_strategies():
    """Return the list of available backtest strategy names."""
    try:
        from backtest_runner import STRATEGIES
        return jsonify({'success': True, 'strategies': list(STRATEGIES.keys())})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/statistics/<symbol>')
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

@app.route('/api/sentiment/<symbol>')
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
        total = len(sentiments)
        overall = 'neutral'
        if total > 0:
            if positive_count > negative_count and positive_count > neutral_count:
                overall = 'positive'
            elif negative_count > positive_count:
                overall = 'negative'
        
        return jsonify({
            'success': True,
            'symbol': symbol.upper(),
            'overall_sentiment': overall,
            'distribution': {
                'positive': positive_count,
                'negative': negative_count,
                'neutral': neutral_count
            },
            'sentiments': sentiments,
            'total_analyzed': total
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/sentiment/news/<symbol>')
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
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from algorithms.machine_learning_algorithms.data_pipelines.extractor import DataExtractor

        sym      = symbol.upper()
        days     = min(int(request.args.get('days', 30)), 365)
        end      = datetime.date.today().isoformat()
        start    = (datetime.date.today() - datetime.timedelta(days=days)).isoformat()

        ex = DataExtractor()

        # Article-level feed (AV scored + Finnhub unscored); larger cap for longer windows
        max_art = 150 if days > 30 else 100
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


@app.route('/api/scenarios/<symbol>')
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

@app.route('/api/metrics/<symbol>')
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

@app.route('/api/recommendations/<symbol>')
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

@app.route('/api/calendar')
def earnings_calendar():
    """Get comprehensive calendar with earnings, dividends, and macro events"""
    try:
        today = datetime.date.today()
        from_date = today
        to_date = datetime.date(2026, 12, 31)  # Full year 2026
        
        events = []
        
        # 1. Get earnings and dividend dates from yfinance for watchlist stocks
        for symbol in DEFAULT_STOCKS:
            try:
                # Determine correct ticker symbol
                ticker_symbol = get_ticker_for_charts(symbol) if is_canadian_stock(symbol) else symbol
                stock = yf.Ticker(ticker_symbol)
                
                # Get calendar which includes earnings and dividend dates
                calendar = stock.calendar
                
                if calendar:
                    # Earnings date
                    earnings_dates = calendar.get('Earnings Date', [])
                    if earnings_dates:
                        # Handle both list and single date
                        if not isinstance(earnings_dates, list):
                            earnings_dates = [earnings_dates]
                        
                        for earnings_date in earnings_dates:
                            if earnings_date and isinstance(earnings_date, datetime.date):
                                if from_date <= earnings_date <= to_date:
                                    # Get earnings estimates
                                    earnings_avg = calendar.get('Earnings Average', 0)
                                    earnings_low = calendar.get('Earnings Low', 0)
                                    earnings_high = calendar.get('Earnings High', 0)
                                    
                                    estimate_text = ''
                                    if earnings_avg:
                                        estimate_text = f' (Est: ${earnings_avg:.2f}, Range: ${earnings_low:.2f}-${earnings_high:.2f})'
                                    
                                    events.append({
                                        'symbol': symbol,
                                        'date': earnings_date.strftime('%Y-%m-%d'),
                                        'type': 'Earnings',
                                        'description': f'{symbol} Earnings Report{estimate_text}',
                                        'importance': 'high'
                                    })
                    
                    # Dividend ex-date
                    ex_dividend_date = calendar.get('Ex-Dividend Date')
                    if ex_dividend_date and isinstance(ex_dividend_date, datetime.date):
                        if from_date <= ex_dividend_date <= to_date:
                            # Get dividend info
                            dividend_date = calendar.get('Dividend Date')
                            info = stock.info
                            dividend_rate = info.get('dividendRate', info.get('lastDividendValue', 0))
                            
                            div_text = f'${dividend_rate:.2f}/share' if dividend_rate else ''
                            pay_text = f' (Pay: {dividend_date.strftime("%Y-%m-%d")})' if dividend_date else ''
                            
                            events.append({
                                'symbol': symbol,
                                'date': ex_dividend_date.strftime('%Y-%m-%d'),
                                'type': 'Dividend',
                                'description': f'{symbol} Ex-Dividend Date {div_text}{pay_text}',
                                'importance': 'medium'
                            })
                        
            except Exception as e:
                print(f"Error getting calendar data for {symbol}: {e}")
                continue
        
        # 2. Load macro economic events from JSON file
        try:
            economic_events_path = os.path.join(os.path.dirname(__file__), 'economic_events.json')
            with open(economic_events_path, 'r') as f:
                economic_data = json.load(f)
                for event in economic_data.get('events', []):
                    event_date = datetime.datetime.strptime(event['date'], '%Y-%m-%d').date()
                    if from_date <= event_date <= to_date:
                        events.append({
                            'symbol': None,
                            'date': event['date'],
                            'type': event['type'],
                            'description': event['description'],
                            'importance': event['importance']
                        })
        except Exception as e:
            print(f"Error loading economic events: {e}")
        
        # Sort events by date
        events.sort(key=lambda x: x['date'])
        
        return jsonify({
            'success': True,
            'events': events,
            'date_range': {
                'from': from_date.strftime('%Y-%m-%d'),
                'to': to_date.strftime('%Y-%m-%d')
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/recommendations/<symbol>')
def get_recommendations(symbol):
    """Get time-based recommendations"""
    try:
        recommendations = {}
        
        for timeframe, config in TIMEFRAMES.items():
            # Get scenario analysis
            financials = get_basic_financials(symbol.upper())
            quote = av.get_global_quote(symbol.upper())
            
            current_price = float(quote.get('05. price', 0)) if quote else 0
            metrics = financials.get('metric', {}) if financials else {}
            
            eps_growth = float(metrics.get('epsGrowthTTMYoy', 10))
            days = config['days']
            multiplier = days / 365
            
            expected_return = eps_growth * multiplier
            
            # Recommendation logic
            if expected_return > 15:
                action = 'Strong Buy'
                confidence = 'High'
            elif expected_return > 8:
                action = 'Buy'
                confidence = 'Medium'
            elif expected_return > 0:
                action = 'Hold'
                confidence = 'Medium'
            else:
                action = 'Sell'
                confidence = 'Low'
            
            target_price = current_price * (1 + expected_return / 100)
            
            recommendations[timeframe] = {
                'timeframe': config['label'],
                'action': action,
                'confidence': confidence,
                'current_price': current_price,
                'target_price': round(target_price, 2),
                'expected_return': round(expected_return, 2)
            }
        
        return jsonify({
            'success': True,
            'symbol': symbol.upper(),
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/charts/<symbol>')
def get_charts(symbol):
    """Get chart data for a stock
    
    Query Parameters:
    - period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max (default: 1y)
    - interval: 1m, 2m, 5m, 15m, 30m, 60m, 1h, 1d, 5d, 1wk, 1mo (default: 1d)
    """
    period = request.args.get('period', '1y')
    interval = request.args.get('interval', '1d')
    
    symbol_upper = symbol.upper()
    # Use TSX symbol for Canadian stocks to get CAD prices
    chart_symbol = get_ticker_for_charts(symbol_upper)
    
    data = get_chart_data(chart_symbol, period, interval)
    # Keep the original symbol in response
    data['display_symbol'] = symbol_upper
    data['currency'] = 'CAD' if is_canadian_stock(symbol_upper) else 'USD'
    return jsonify(data)

@app.route('/api/charts/<symbol>/all-timeframes')
def get_all_timeframes(symbol):
    """Get chart data for all timeframes"""
    symbol_upper = symbol.upper()
    # Use TSX symbol for Canadian stocks to get CAD prices
    chart_symbol = get_ticker_for_charts(symbol_upper)
    
    data = get_multiple_timeframes(chart_symbol)
    data['display_symbol'] = symbol_upper
    data['currency'] = 'CAD' if is_canadian_stock(symbol_upper) else 'USD'
    return jsonify(data)

@app.route('/api/charts/compare')
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
        })
    
    period = request.args.get('period', '1y')
    interval = request.args.get('interval', '1d')
    
    data = get_comparison_data(symbols, period, interval)
    return jsonify(data)

@app.route('/api/charts/<symbol>/indicators')
def get_indicators(symbol):
    """Get chart data with technical indicators (RSI, MACD, Bollinger Bands, Moving Averages)
    
    Query Parameters:
    - period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max (default: 1y)
    - interval: 1m, 2m, 5m, 15m, 30m, 60m, 1h, 1d, 5d, 1wk, 1mo (default: 1d)
    """
    period = request.args.get('period', '1y')
    interval = request.args.get('interval', '1d')
    
    symbol_upper = symbol.upper()
    # Use TSX symbol for Canadian stocks to get CAD prices
    chart_symbol = get_ticker_for_charts(symbol_upper)
    
    data = get_technical_indicators(chart_symbol, period, interval)
    # Keep the original symbol in response
    data['display_symbol'] = symbol_upper
    data['currency'] = 'CAD' if is_canadian_stock(symbol_upper) else 'USD'
    return jsonify(data)

@app.route('/api/news/twitter')
def twitter_news():
    """Get latest market news from Twitter
    
    Query Parameters:
    - symbol: Optional stock symbol to filter tweets (e.g., AAPL)
    - count: Number of tweets to return (default: 20, max: 100)
    - type: 'market' for general market news, 'financial' for financial news sources
    """
    symbol = request.args.get('symbol', None)
    count = min(int(request.args.get('count', 20)), 100)
    news_type = request.args.get('type', 'market')
    
    try:
        if news_type == 'financial':
            # Get tweets from major financial news accounts
            tweets = get_financial_news_feed(count=count)
        else:
            # Get general market tweets or symbol-specific tweets
            tweets = get_market_tweets(symbol=symbol, count=count)
        
        return jsonify({
            'success': True,
            'count': len(tweets),
            'tweets': tweets,
            'source': 'twitter'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/news/alpaca')
def alpaca_news():
    """Get real-time news from Alpaca stream
    
    Query Parameters:
    - symbol: Optional stock symbol to filter news
    - count: Number of news items to return (default: 20)
    """
    symbol = request.args.get('symbol', None)
    count = int(request.args.get('count', 20))
    
    try:
        news_items = get_recent_news(count=count, symbol=symbol)
        
        return jsonify({
            'success': True,
            'count': len(news_items),
            'news': news_items,
            'source': 'alpaca'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/news/combined')
def combined_news():
    """Get combined news from all sources (Finnhub, Twitter, Alpaca)
    
    Query Parameters:
    - symbol: Optional stock symbol to filter news
    - count: Number of items per source (default: 10)
    """
    symbol = request.args.get('symbol', None)
    count = int(request.args.get('count', 10))
    
    try:
        all_news = []
        
        # Get Finnhub news for popular symbols even if no symbol specified
        symbols_to_check = []
        if symbol:
            symbols_to_check = [symbol.upper()]
        else:
            # Fetch news for popular stocks when no specific symbol
            symbols_to_check = ['NVDA', 'AAPL', 'MSFT', 'TSLA', 'GOOGL'][:2]  # Limit to 2 to respect API limits
        
        for sym in symbols_to_check:
            try:
                today = datetime.date.today()
                from_date = (today - datetime.timedelta(days=7)).strftime('%Y-%m-%d')
                to_date = today.strftime('%Y-%m-%d')
                finnhub_news = get_company_news(sym, from_date, to_date)
                
                # Format Finnhub news (limit per symbol)
                items_per_symbol = count // len(symbols_to_check) if len(symbols_to_check) > 1 else count
                for article in finnhub_news[:items_per_symbol]:
                    all_news.append({
                        'source': 'Finnhub',
                        'headline': article.get('headline', ''),
                        'summary': article.get('summary', ''),
                        'url': article.get('url', ''),
                        'created_at': datetime.datetime.fromtimestamp(article.get('datetime', 0)).isoformat(),
                        'symbols': [sym],
                        'type': 'article'
                    })
            except Exception as e:
                print(f"⚠️  Error fetching Finnhub news for {sym}: {e}")
        
        # Get Twitter news
        twitter_error = None
        try:
            tweets = get_market_tweets(symbol=symbol, count=count)
            if tweets and len(tweets) > 0:
                for tweet in tweets:
                    all_news.append({
                        'source': 'Twitter',
                        'headline': f"@{tweet['author']['username']}: {tweet['text'][:100]}...",
                        'summary': tweet['text'],
                        'url': tweet['url'],
                        'created_at': tweet['created_at'],
                        'symbols': tweet.get('symbols', []),
                        'author': tweet['author'],
                        'metrics': tweet['metrics'],
                        'type': 'tweet'
                    })
            else:
                twitter_error = "Twitter API rate limit reached or no data available"
                print(f"⚠️  {twitter_error}")
        except Exception as e:
            twitter_error = f"Twitter API unavailable: {str(e)}"
            print(f"⚠️  {twitter_error}")
        
        # Get Alpaca news (if available)
        try:
            alpaca_items = get_recent_news(count=count, symbol=symbol)
            for item in alpaca_items:
                all_news.append({
                    'source': item['source'],
                    'headline': item['headline'],
                    'summary': item.get('summary', ''),
                    'url': item.get('url', ''),
                    'created_at': item['created_at'],
                    'symbols': item.get('symbols', []),
                    'author': item.get('author', ''),
                    'type': 'article'
                })
        except Exception as e:
            print(f"⚠️  Alpaca news not available: {e}")
        
        # Sort by created_at (newest first)
        all_news.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        response = {
            'success': True,
            'count': len(all_news),
            'news': all_news
        }
        
        # Add warning if Twitter API failed
        if twitter_error:
            response['warning'] = twitter_error
        
        return jsonify(response)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/research/<paper_name>')
def get_research_paper(paper_name):
    """
    Dynamically compile LaTeX to PDF and serve it.
    This saves space by not storing PDFs and always serves the latest version.
    """
    try:
        # Map paper names to their .tex file paths
        paper_map = {
            'heston': '../quant_research/stochastic_volatility/heston_model/theory.tex',
            'sabr': '../quant_research/stochastic_volatility/sabr_model/theory.tex',
            'state_space': '../quant_research/state_space_models/theory.tex',
            'market_microstructure': '../quant_research/market_microstructure/theory.tex',
            'macd': '../quant_research/macd_rsi/macd_theory.tex',
            'rsi': '../quant_research/macd_rsi/rsi_theory.tex',
            'greeks': '../quant_research/greeks/theory.tex',
            'derivatives_volatility': '../quant_research/derivatives_volatility/theory.tex',
            'advanced_trading': '../quant_research/advanced_trading/theory.tex',
        }
        
        if paper_name not in paper_map:
            return jsonify({
                'success': False,
                'error': 'Research paper not found'
            }), 404
        
        # Get the absolute path to the .tex file
        tex_file = os.path.join(os.path.dirname(__file__), paper_map[paper_name])
        tex_dir = os.path.dirname(tex_file)
        tex_filename = os.path.basename(tex_file)
        
        if not os.path.exists(tex_file):
            return jsonify({
                'success': False,
                'error': 'LaTeX source file not found'
            }), 404
        
        # Create a temporary directory for compilation
        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy the .tex file to temp directory
            import shutil
            temp_tex = os.path.join(tmpdir, tex_filename)
            shutil.copy(tex_file, temp_tex)
            
            # Compile LaTeX to PDF using pdflatex
            # Run twice to resolve references
            for _ in range(2):
                result = subprocess.run(
                    ['pdflatex', '-interaction=nonstopmode', '-output-directory', tmpdir, temp_tex],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
            
            # Check if PDF was generated
            pdf_filename = tex_filename.replace('.tex', '.pdf')
            pdf_path = os.path.join(tmpdir, pdf_filename)
            
            if not os.path.exists(pdf_path):
                return jsonify({
                    'success': False,
                    'error': 'PDF compilation failed',
                    'log': result.stderr
                }), 500
            
            # Read the PDF and return it
            return send_file(
                pdf_path,
                mimetype='application/pdf',
                as_attachment=False,
                download_name=f'{paper_name}_model.pdf'
            )
    
    except subprocess.TimeoutExpired:
        return jsonify({
            'success': False,
            'error': 'LaTeX compilation timed out'
        }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error generating PDF: {str(e)}'
        }), 500

@app.route('/api/research/diagnostics/notebook')
def get_diagnostics_notebook():
    """
    Convert and serve diagnostics notebook as HTML on-demand
    """
    try:
        # Path to the notebook
        notebook_path = os.path.join(
            os.path.dirname(__file__), 
            '../algorithms/volatility_forecasting/research/diagnostics.ipynb'
        )
        
        if not os.path.exists(notebook_path):
            return jsonify({
                'success': False,
                'error': 'Notebook not found'
            }), 404
        
        # Create temporary directory for conversion
        with tempfile.TemporaryDirectory() as tmpdir:
            output_html = os.path.join(tmpdir, 'diagnostics.html')
            
            # Convert notebook to HTML without code cells (--no-input)
            result = subprocess.run(
                ['jupyter', 'nbconvert', '--to', 'html', '--no-input', 
                 '--output', output_html, notebook_path],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                return jsonify({
                    'success': False,
                    'error': 'Notebook conversion failed',
                    'details': result.stderr
                }), 500
            
            # Serve the converted HTML
            return send_file(
                output_html,
                mimetype='text/html',
                as_attachment=False
            )
            
    except subprocess.TimeoutExpired:
        return jsonify({
            'success': False,
            'error': 'Conversion timed out'
        }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/algorithm/<name>')
def get_algorithm_source(name):
    """
    Serve Python algorithm source code as plain text.
    """
    algo_map = {
        'sabr_pricer':           '../algorithms/volatility_forecasting/volatility_models/sabr_pricer.py',
        'sabr_calibration':      '../algorithms/volatility_forecasting/volatility_models/calibration/run_calibration.py',
        'signal_generator':      '../algorithms/volatility_forecasting/volatility_models/signals/signal_generator.py',
        'strategy_signals':      '../algorithms/volatility_forecasting/volatility_models/signals/strategy_signals.py',
        'backtest_engine':       '../algorithms/volatility_forecasting/backtest_engine/engine.py',
        'portfolio_constructor': '../algorithms/volatility_forecasting/portfolio_engine/portfolio_constructor.py',
        'macd_strategy':         '../algorithms/macd_rsi/prototype.py',
        'greeks_calculator':     '../algorithms/greeks/prototype.py',
        'arima':                 '../algorithms/machine_learning_algorithms/time_series_models/arima.py',
        'garch':                 '../algorithms/machine_learning_algorithms/time_series_models/garch.py',
    }

    if name not in algo_map:
        return jsonify({'success': False, 'error': 'Algorithm not found'}), 404

    file_path = os.path.join(os.path.dirname(__file__), algo_map[name])
    if not os.path.exists(file_path):
        return jsonify({'success': False, 'error': 'Source file not found'}), 404

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        return source, 200, {'Content-Type': 'text/plain; charset=utf-8'}
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/research/<paper_name>/markdown')
def get_research_markdown(paper_name):
    """
    Serve markdown research papers directly
    """
    try:
        # Map paper names to their markdown file paths
        markdown_map = {
            'advanced_trading': '../quant_research/advanced_trading/theory.md',
        }
        
        if paper_name not in markdown_map:
            return jsonify({
                'success': False,
                'error': 'Research paper not found'
            }), 404
        
        # Get the absolute path to the markdown file
        md_file = os.path.join(os.path.dirname(__file__), markdown_map[paper_name])
        
        if not os.path.exists(md_file):
            return jsonify({
                'success': False,
                'error': 'Markdown file not found'
            }), 404
        
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return jsonify({
            'success': True,
            'content': content
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/trading/chart', methods=['POST'])
def trading_chart():
    """Proxy chart-img.com TradingView chart image requests.
    
    Tries the v2 POST endpoint first (paid plan); falls back to v1 GET (free plan).
    Auth header: x-api-key.  Set CHART_IMG_KEY env var to override the default key.
    """
    import requests as _req

    CHART_IMG_KEY = os.environ.get('CHART_IMG_KEY', '')

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


if __name__ == '__main__':
    try:
        port = int(os.environ.get('PORT', 5001))
        debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
        app.run(debug=debug, host='0.0.0.0', port=port)
    finally:
        # Clean up: stop the Alpaca news stream when app closes
        stop_news_stream()

