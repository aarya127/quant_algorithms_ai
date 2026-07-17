"""
routes/market.py — market overview (search, indices, dashboard, calendar) blueprint, extracted verbatim from app.py.
"""
import os
import sys
import json
import datetime

from flask import Blueprint, jsonify, request

from concurrent.futures import ThreadPoolExecutor
from common import DEFAULT_STOCKS, MARKET_INDICES
from services import yf, get_company_news

bp = Blueprint('market', __name__)

# backend/ directory (this file lives one level deeper than app.py did; moved
# code that used os.path.dirname(__file__) now uses _BACKEND instead)
_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@bp.route('/api/search/<query>')
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

@bp.route('/api/indices')
def market_indices():
    """Get live market indices for the dashboard cards (see MARKET_INDICES)"""
    def _quote(entry):
        symbol, name = entry
        try:
            fi = yf.Ticker(symbol).fast_info
            price = fi.last_price
            prev  = fi.previous_close
            change = round(price - prev, 2)
            pct    = round((change / prev) * 100, 2) if prev else 0
            return {
                'symbol': symbol,
                'name': name,
                'price': round(price, 2),
                'change': change,
                'pct_change': pct,
            }
        except Exception:
            return None

    # Parallel fetch (10 symbols serially would take seconds); map() preserves
    # MARKET_INDICES order, which is the card display order.
    with ThreadPoolExecutor(max_workers=10) as ex:
        result = [q for q in ex.map(_quote, MARKET_INDICES) if q]
    return jsonify({'success': True, 'indices': result})

@bp.route('/api/indices/history')
def indices_history():
    """Return daily closes for each index for sparkline charts"""
    period = request.args.get('period', '1mo')
    interval = request.args.get('interval', '1d')
    # Safety: only allow known yfinance period/interval values
    allowed_periods = {'5d', '1mo', '3mo', '6mo', '1y'}
    allowed_intervals = {'5m', '15m', '30m', '1h', '1d', '1wk'}
    if period not in allowed_periods:
        period = '1mo'
    if interval not in allowed_intervals:
        interval = '1d'
    def _hist(entry):
        symbol, _name = entry
        try:
            df = yf.Ticker(symbol).history(period=period, interval=interval)
            if not df.empty:
                return symbol, {
                    'labels': [d.strftime('%b %d') for d in df.index],
                    'closes': [round(float(v), 2) for v in df['Close']],
                }
        except Exception:
            pass
        return symbol, None

    with ThreadPoolExecutor(max_workers=10) as ex:
        result = {sym: hist for sym, hist in ex.map(_hist, MARKET_INDICES) if hist}
    return jsonify({'success': True, 'history': result})

@bp.route('/api/dashboard')
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
        
        # Market hours in US/Eastern (server clock is UTC on Render) incl. weekends.
        from zoneinfo import ZoneInfo
        now_et = datetime.datetime.now(ZoneInfo('America/New_York'))
        is_open = (now_et.weekday() < 5
                   and (now_et.hour, now_et.minute) >= (9, 30)
                   and now_et.hour < 16)
        return jsonify({
            'success': True,
            'trending': trending,
            'market_status': 'Open' if is_open else 'Closed',
            'timestamp': datetime.datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@bp.route('/api/calendar')
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
            economic_events_path = os.path.join(_BACKEND, 'economic_events.json')
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

# NOTE: a second '/api/recommendations/<symbol>' handler (get_recommendations)
# was removed here — it was unreachable dead code. Flask matched the first
# registration (`recommendations`, above) and this one never ran.

