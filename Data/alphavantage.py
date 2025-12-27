import requests
import os

class AlphaVantage:
    def __init__(self, api_key=None):
        """Initialize AlphaVantage API client with API key from keys.txt"""
        if api_key is None:
            self.api_key = self._load_api_key()
        else:
            self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
    
    def _load_api_key(self):
        """Load API key from keys.txt file"""
        keys_path = os.path.join(os.path.dirname(__file__), '..', 'keys.txt')
        with open(keys_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if 'alphavantage' in line.lower():
                    # Get the next line which should contain the API key
                    if i + 1 < len(lines):
                        return lines[i + 1].strip()
        raise ValueError("AlphaVantage API key not found in keys.txt")
    
    def _make_request(self, params):
        """Helper method to make API requests"""
        params['apikey'] = self.api_key
        response = requests.get(self.base_url, params=params)
        return response.json()
    
    # News & Sentiment
    def get_market_news_sentiment(self, tickers=None, topics=None, time_from=None, time_to=None, sort='LATEST', limit=50):
        """Get market news and sentiment data"""
        params = {'function': 'NEWS_SENTIMENT', 'sort': sort, 'limit': limit}
        if tickers:
            params['tickers'] = tickers
        if topics:
            params['topics'] = topics
        if time_from:
            params['time_from'] = time_from
        if time_to:
            params['time_to'] = time_to
        return self._make_request(params)
    
    # Earnings & Fundamentals
    def get_earnings_call_transcript(self, symbol):
        """Get earnings call transcript"""
        params = {'function': 'EARNINGS_CALL_TRANSCRIPT', 'symbol': symbol}
        return self._make_request(params)
    
    def get_top_gainers_losers(self):
        """Get top gainers, losers, and most actively traded tickers"""
        params = {'function': 'TOP_GAINERS_LOSERS'}
        return self._make_request(params)
    
    def get_insider_transactions(self, symbol):
        """Get insider transactions for a symbol"""
        params = {'function': 'INSIDER_TRANSACTIONS', 'symbol': symbol}
        return self._make_request(params)
    
    def get_company_overview(self, symbol):
        """Get company overview and fundamental data"""
        params = {'function': 'OVERVIEW', 'symbol': symbol}
        return self._make_request(params)
    
    def get_global_quote(self, symbol):
        """Get real-time quote data for a symbol"""
        params = {'function': 'GLOBAL_QUOTE', 'symbol': symbol}
        result = self._make_request(params)
        return result.get('Global Quote', {}) if result else {}
    
    def get_income_statement(self, symbol):
        """Get income statement"""
        params = {'function': 'INCOME_STATEMENT', 'symbol': symbol}
        return self._make_request(params)
    
    def get_balance_sheet(self, symbol):
        """Get balance sheet"""
        params = {'function': 'BALANCE_SHEET', 'symbol': symbol}
        return self._make_request(params)
    
    def get_cash_flow(self, symbol):
        """Get cash flow statement"""
        params = {'function': 'CASH_FLOW', 'symbol': symbol}
        return self._make_request(params)
    
    def get_shares_outstanding(self, symbol):
        """Get shares outstanding"""
        params = {'function': 'SHARES_OUTSTANDING', 'symbol': symbol}
        return self._make_request(params)
    
    def get_earnings_history(self, symbol):
        """Get earnings history"""
        params = {'function': 'EARNINGS', 'symbol': symbol}
        return self._make_request(params)
    
    def get_earnings_estimates(self, symbol):
        """Get earnings estimates"""
        params = {'function': 'EARNINGS_ESTIMATES', 'symbol': symbol}
        return self._make_request(params)
    
    def get_earnings_calendar(self, symbol=None, horizon='3month'):
        """Get earnings calendar"""
        params = {'function': 'EARNINGS_CALENDAR', 'horizon': horizon}
        if symbol:
            params['symbol'] = symbol
        return self._make_request(params)
    
    # Economic Indicators
    def get_wti_crude_oil(self, interval='monthly'):
        """Get WTI crude oil prices"""
        params = {'function': 'WTI', 'interval': interval}
        return self._make_request(params)
    
    def get_brent_crude_oil(self, interval='monthly'):
        """Get Brent crude oil prices"""
        params = {'function': 'BRENT', 'interval': interval}
        return self._make_request(params)
    
    def get_natural_gas(self, interval='monthly'):
        """Get natural gas prices"""
        params = {'function': 'NATURAL_GAS', 'interval': interval}
        return self._make_request(params)
    
    def get_real_gdp(self, interval='annual'):
        """Get real GDP"""
        params = {'function': 'REAL_GDP', 'interval': interval}
        return self._make_request(params)
    
    def get_treasury_yield(self, interval='monthly', maturity='10year'):
        """Get treasury yield"""
        params = {'function': 'TREASURY_YIELD', 'interval': interval, 'maturity': maturity}
        return self._make_request(params)
    
    def get_federal_funds_rate(self, interval='monthly'):
        """Get federal funds rate"""
        params = {'function': 'FEDERAL_FUNDS_RATE', 'interval': interval}
        return self._make_request(params)
    
    def get_inflation(self, interval='monthly'):
        """Get inflation (CPI)"""
        params = {'function': 'INFLATION', 'interval': interval}
        return self._make_request(params)
    
    # Technical Indicators
    def get_sma(self, symbol, interval='daily', time_period=20, series_type='close'):
        """Get Simple Moving Average"""
        params = {
            'function': 'SMA',
            'symbol': symbol,
            'interval': interval,
            'time_period': time_period,
            'series_type': series_type
        }
        return self._make_request(params)
    
    def get_ema(self, symbol, interval='daily', time_period=20, series_type='close'):
        """Get Exponential Moving Average"""
        params = {
            'function': 'EMA',
            'symbol': symbol,
            'interval': interval,
            'time_period': time_period,
            'series_type': series_type
        }
        return self._make_request(params)
    
    def get_rsi(self, symbol, interval='daily', time_period=14, series_type='close'):
        """Get Relative Strength Index"""
        params = {
            'function': 'RSI',
            'symbol': symbol,
            'interval': interval,
            'time_period': time_period,
            'series_type': series_type
        }
        return self._make_request(params)
    
    def get_adx(self, symbol, interval='daily', time_period=14):
        """Get Average Directional Movement Index"""
        params = {
            'function': 'ADX',
            'symbol': symbol,
            'interval': interval,
            'time_period': time_period
        }
        return self._make_request(params)