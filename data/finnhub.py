import finnhub
import datetime
import os

# ---------------------------------------------------------------------------
# Module-level singleton — one client for the entire process lifetime.
# This avoids re-reading the API key file and re-instantiating the HTTP
# client on every function call (previously done ~8 times per page load).
# ---------------------------------------------------------------------------

def _load_api_key():
    key = os.environ.get('FINNHUB_API_KEY')
    if key:
        return key
    possible_paths = ['keys.txt', '../keys.txt', os.path.join(os.path.dirname(__file__), '..', 'keys.txt')]
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if 'finnhub' in line.lower() or 'finhub' in line.lower():
                        if i + 1 < len(lines):
                            return lines[i + 1].strip()
            break
    return None

_client = finnhub.Client(api_key=_load_api_key())

def get_company_profile(symbol):
    """Get company profile information"""
    return _client.company_profile2(symbol=symbol)

def get_stock_quote(symbol):
    """Get real-time stock quote"""
    return _client.quote(symbol)

def get_company_news(symbol, from_date, to_date):
    """Get company news for a specific symbol"""
    return _client.company_news(symbol, _from=from_date, to=to_date)

def get_market_news(category='general'):
    """Get general market news. Categories: general, forex, crypto, merger"""
    return _client.general_news(category)

def get_basic_financials(symbol, metric='all'):
    """Get basic financials for a symbol"""
    return _client.company_basic_financials(symbol, metric)

def get_insider_transactions(symbol, from_date=None, to_date=None):
    """Get insider transactions for a symbol"""
    return _client.stock_insider_transactions(symbol, from_date, to_date)

def get_insider_sentiment(symbol, from_date, to_date):
    """Get insider sentiment for a symbol"""
    return _client.stock_insider_sentiment(symbol, from_date, to_date)

def get_earnings_surprises(symbol):
    """Get earnings surprises for a symbol"""
    return _client.company_earnings(symbol)

def get_earnings_calendar(from_date, to_date, symbol=None):
    """Get earnings calendar. If symbol provided, filter for that symbol"""
    return _client.earnings_calendar(_from=from_date, to=to_date, symbol=symbol)

def get_usa_spending(symbol, from_date, to_date):
    """Get USA spending data for a symbol"""
    return _client.stock_usa_spending(symbol, from_date, to_date)

# Example usage
if __name__ == "__main__":
    symbol = "AAPL"
    today = datetime.date.today()
    from_date = (today - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
    to_date = today.strftime('%Y-%m-%d')
    
    # Example calls
    # news = get_company_news(symbol, from_date, to_date)
    # financials = get_basic_financials(symbol)
    # transactions = get_insider_transactions(symbol)