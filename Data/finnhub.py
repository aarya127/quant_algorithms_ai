import finnhub
import datetime

def load_api_key():
    """Load Finnhub API key from keys.txt"""
    # Try to find keys.txt in current directory or parent directory
    import os
    possible_paths = ['keys.txt', '../keys.txt', os.path.join(os.path.dirname(__file__), '..', 'keys.txt')]
    
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    # Check for finnhub or finhub (typo)
                    if 'finnhub' in line.lower() or 'finhub' in line.lower():
                        if i + 1 < len(lines):
                            return lines[i + 1].strip()
            break
    
    raise ValueError("Finnhub API key not found in keys.txt")

def get_company_profile(symbol):
    """Get company profile information"""
    api_key = load_api_key()
    finnhub_client = finnhub.Client(api_key=api_key)
    profile = finnhub_client.company_profile2(symbol=symbol)
    return profile

def get_stock_quote(symbol):
    """Get real-time stock quote"""
    api_key = load_api_key()
    finnhub_client = finnhub.Client(api_key=api_key)
    quote = finnhub_client.quote(symbol)
    return quote

def get_company_news(symbol, from_date, to_date):
    """Get company news for a specific symbol"""
    api_key = load_api_key()
    finnhub_client = finnhub.Client(api_key=api_key)
    news = finnhub_client.company_news(symbol, _from=from_date, to=to_date)
    return news

def get_market_news(category='general'):
    """Get general market news. Categories: general, forex, crypto, merger"""
    api_key = load_api_key()
    finnhub_client = finnhub.Client(api_key=api_key)
    news = finnhub_client.general_news(category)
    return news

def get_basic_financials(symbol, metric='all'):
    """Get basic financials for a symbol"""
    api_key = load_api_key()
    finnhub_client = finnhub.Client(api_key=api_key)
    financials = finnhub_client.company_basic_financials(symbol, metric)
    return financials

def get_insider_transactions(symbol, from_date=None, to_date=None):
    """Get insider transactions for a symbol"""
    api_key = load_api_key()
    finnhub_client = finnhub.Client(api_key=api_key)
    transactions = finnhub_client.stock_insider_transactions(symbol, from_date, to_date)
    return transactions

def get_insider_sentiment(symbol, from_date, to_date):
    """Get insider sentiment for a symbol"""
    api_key = load_api_key()
    finnhub_client = finnhub.Client(api_key=api_key)
    sentiment = finnhub_client.stock_insider_sentiment(symbol, from_date, to_date)
    return sentiment

def get_earnings_surprises(symbol):
    """Get earnings surprises for a symbol"""
    api_key = load_api_key()
    finnhub_client = finnhub.Client(api_key=api_key)
    surprises = finnhub_client.company_earnings(symbol)
    return surprises

def get_earnings_calendar(from_date, to_date, symbol=None):
    """Get earnings calendar. If symbol provided, filter for that symbol"""
    api_key = load_api_key()
    finnhub_client = finnhub.Client(api_key=api_key)
    calendar = finnhub_client.earnings_calendar(_from=from_date, to=to_date, symbol=symbol)
    return calendar

def get_usa_spending(symbol, from_date, to_date):
    """Get USA spending data for a symbol"""
    api_key = load_api_key()
    finnhub_client = finnhub.Client(api_key=api_key)
    spending = finnhub_client.stock_usa_spending(symbol, from_date, to_date)
    return spending

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