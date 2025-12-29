"""
Stock Predictor Web Application
Comprehensive UI for stock analysis and predictions
"""

from flask import Flask, render_template, jsonify, request
import sys
import os
import datetime
import json

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Data.finnhub import (
    get_company_news,
    get_basic_financials,
    get_earnings_surprises,
    get_insider_transactions,
    get_insider_sentiment,
    get_earnings_calendar,
    get_company_profile,
    get_stock_quote
)
from Data.alphavantage import AlphaVantage
from Data.nvidia_llm import get_company_overview_llm
from Data.charts import get_chart_data, get_multiple_timeframes, get_comparison_data
from Sentiment.finbert import get_sentiment
from stock_analyzer import StockAnalyzer

app = Flask(__name__)
av = AlphaVantage()

# Configuration
DEFAULT_STOCKS = ["NVDA", "AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "META"]
TIMEFRAMES = {
    '1W': {'days': 7, 'label': '1 Week'},
    '1M': {'days': 30, 'label': '1 Month'},
    '3M': {'days': 90, 'label': '3 Months'},
    '6M': {'days': 180, 'label': '6 Months'},
    '1Y': {'days': 365, 'label': '1 Year'},
}

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html', default_stocks=DEFAULT_STOCKS)

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
        # Company profile from Finnhub
        company = get_company_profile(symbol.upper())
        
        # Current quote from Finnhub (more reliable than Alpha Vantage)
        quote = get_stock_quote(symbol.upper())
        
        # Recent news
        today = datetime.date.today()
        from_date = (today - datetime.timedelta(days=7)).strftime('%Y-%m-%d')
        to_date = today.strftime('%Y-%m-%d')
        news = get_company_news(symbol.upper(), from_date, to_date)
        
        # Basic financials
        financials = get_basic_financials(symbol.upper())
        
        # Return immediately without waiting for AI overview (load it separately)
        return jsonify({
            'success': True,
            'symbol': symbol.upper(),
            'company': company,  # Finnhub company profile
            'quote': quote,  # Finnhub real-time quote
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
    """Get earnings calendar"""
    try:
        today = datetime.date.today()
        from_date = today.strftime('%Y-%m-%d')
        to_date = (today + datetime.timedelta(days=30)).strftime('%Y-%m-%d')
        
        events = []
        for symbol in DEFAULT_STOCKS:
            try:
                earnings = get_earnings_surprises(symbol)
                if earnings and len(earnings) > 0:
                    latest = earnings[0]
                    events.append({
                        'symbol': symbol,
                        'date': latest.get('period', ''),
                        'type': 'Earnings',
                        'estimate': latest.get('estimate', 0),
                        'actual': latest.get('actual', None)
                    })
            except:
                continue
        
        return jsonify({
            'success': True,
            'events': events
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

@app.route('/api/search')
def search_stocks():
    """Search for stocks"""
    query = request.args.get('q', '').upper()
    
    # Simple search - in production, you'd use a proper stock search API
    results = []
    all_stocks = DEFAULT_STOCKS + ['AC.TO', 'TD.TO', 'ENB.TO']
    
    for symbol in all_stocks:
        if query in symbol:
            results.append({
                'symbol': symbol,
                'name': symbol  # In production, get actual company name
            })
    
    return jsonify({
        'success': True,
        'results': results
    })

@app.route('/api/charts/<symbol>')
def get_charts(symbol):
    """Get chart data for a stock
    
    Query Parameters:
    - period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max (default: 1y)
    - interval: 1m, 2m, 5m, 15m, 30m, 60m, 1h, 1d, 5d, 1wk, 1mo (default: 1d)
    """
    period = request.args.get('period', '1y')
    interval = request.args.get('interval', '1d')
    
    data = get_chart_data(symbol.upper(), period, interval)
    return jsonify(data)

@app.route('/api/charts/<symbol>/all-timeframes')
def get_all_timeframes(symbol):
    """Get chart data for all timeframes"""
    data = get_multiple_timeframes(symbol.upper())
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

if __name__ == '__main__':
    app.run(debug=True, port=5000)

