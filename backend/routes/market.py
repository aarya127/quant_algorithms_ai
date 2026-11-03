"""
routes/market.py — market overview (search, indices, dashboard, calendar) blueprint, extracted verbatim from app.py.
"""
import os
import sys
import json
import datetime

from flask import Blueprint, jsonify, request

from concurrent.futures import ThreadPoolExecutor
from common import DEFAULT_STOCKS, MARKET_INDICES, get_ticker_for_charts, is_canadian_stock
from cachetools import TTLCache
from services import yf, get_company_news, _ttl_cached, _cached_yf_info, _safe

bp = Blueprint('market', __name__)

# backend/ directory (this file lives one level deeper than app.py did; moved
# code that used os.path.dirname(__file__) now uses _BACKEND instead)
_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@bp.route('/api/search/<query>')
def search_stocks(query):
    """Search for stocks by symbol or company name (one Yahoo search call)"""
    try:
        quotes = yf.Search(query.strip(), max_results=10, news_count=0).quotes
        results = [{
            'symbol': q['symbol'],
            'name': q.get('longname') or q.get('shortname') or q['symbol'],
            'exchange': q.get('exchange', ''),
            'type': q.get('quoteType', ''),
            'sector': q.get('sector', ''),
            'industry': q.get('industry', ''),
        } for q in quotes
            if q.get('exchange') in ('NYQ', 'NMS', 'NGM', 'NCM', 'ASE', 'PCX', 'TOR', 'NEO', 'CNQ')]
        return jsonify({'success': True, 'results': results, 'query': query})
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
    allowed_periods = {'1d', '5d', '1mo', '3mo', '6mo', '1y'}
    allowed_intervals = {'5m', '15m', '30m', '1h', '1d', '1wk'}
    if period not in allowed_periods:
        period = '1mo'
    if interval not in allowed_intervals:
        interval = '1d'
    return jsonify({'success': True, 'history': _indices_history(period, interval)})


# Range switches on the home page revisit the same few windows; 10 Yahoo
# history calls per click made every switch take seconds.
@_ttl_cached(TTLCache(maxsize=32, ttl=120))
def _indices_history(period, interval):
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
        return {sym: hist for sym, hist in ex.map(_hist, MARKET_INDICES) if hist}

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

def _symbol_calendar_events(symbol, from_date, to_date):
    """Earnings + ex-dividend events for one watchlist symbol (yfinance)."""
    events = []
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
                    # Get dividend info (shared 5-min cache with the stock pages)
                    dividend_date = calendar.get('Dividend Date')
                    info = _safe(_cached_yf_info, ticker_symbol) or {}
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
    return events


# Earnings/dividend dates move rarely; one parallel sweep per 30 min instead
# of ~15-20 serial Yahoo calls on every calendar load (which also burned the
# rate limit the stock pages need).
@_ttl_cached(TTLCache(maxsize=4, ttl=1800))
def _watchlist_calendar_events(from_date, to_date):
    with ThreadPoolExecutor(max_workers=len(DEFAULT_STOCKS)) as ex:
        per_symbol = ex.map(lambda sym: _symbol_calendar_events(sym, from_date, to_date),
                            DEFAULT_STOCKS)
        return [event for events in per_symbol for event in events]


@bp.route('/api/calendar')
def earnings_calendar():
    """Get comprehensive calendar with earnings, dividends, and macro events"""
    try:
        today = datetime.date.today()
        from_date = today
        to_date = today + datetime.timedelta(days=365)
        
        # 1. Earnings and dividend dates from yfinance for watchlist stocks
        events = list(_watchlist_calendar_events(from_date, to_date))
        
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

