"""
Company Statistics Module
Aggregates comprehensive company statistics from multiple sources:
- yfinance (primary source - free, unlimited)
- Alpha Vantage (fallback for missing data)
- Finnhub (additional metrics)
"""

import yfinance as yf
from data.alphavantage import AlphaVantage
from data.finnhub import get_basic_financials


def get_company_statistics(symbol: str) -> dict:
    """
    Get comprehensive company statistics including:
    - Profile (Market Cap, EV, Shares, Revenue, Employees)
    - Margins (Gross, EBITDA, Operating, Pre-Tax, Net, FCF)
    - Returns (ROA, ROTA, ROE, ROCE, ROIC)
    - Valuation (P/E, P/B, EV/Sales, EV/EBITDA, P/FCF)
    - Financial Health (Cash, Debt, Ratios)
    - Growth (CAGR for Revenue and EPS)
    - Dividends
    
    Args:
        symbol: Stock ticker symbol
    
    Returns:
        Dictionary with organized company statistics
    """
    
    stats = {
        'profile': {},
        'margins': {},
        'returns': {},
        'valuation_ttm': {},
        'valuation_forward': {},
        'financial_health': {},
        'growth': {},
        'dividends': {}
    }
    
    try:
        # Primary source: yfinance (most comprehensive and free)
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # ===== PROFILE =====
        stats['profile'] = {
            'market_cap': info.get('marketCap'),
            'enterprise_value': info.get('enterpriseValue'),
            'shares_outstanding': info.get('sharesOutstanding'),
            'revenue_ttm': info.get('totalRevenue'),
            'employees': info.get('fullTimeEmployees'),
        }
        
        # ===== MARGINS =====
        stats['margins'] = {
            'gross_margin': info.get('grossMargins'),
            'ebitda_margin': info.get('ebitdaMargins'),
            'operating_margin': info.get('operatingMargins'),
            'pretax_margin': info.get('profitMargins'),  # Actually profit margin
            'net_margin': info.get('profitMargins'),
            'fcf_margin': None  # Calculate if needed
        }
        
        # Calculate FCF margin if we have data
        if info.get('freeCashflow') and info.get('totalRevenue'):
            stats['margins']['fcf_margin'] = info.get('freeCashflow') / info.get('totalRevenue')
        
        # ===== RETURNS =====
        stats['returns'] = {
            'roa': info.get('returnOnAssets'),
            'rota': None,  # Not directly available in yfinance
            'roe': info.get('returnOnEquity'),
            'roce': None,  # Not directly available
            'roic': None   # Not directly available
        }
        
        # ===== VALUATION (TTM) =====
        stats['valuation_ttm'] = {
            'pe_ratio': info.get('trailingPE'),
            'pb_ratio': info.get('priceToBook'),
            'ps_ratio': info.get('priceToSalesTrailing12Months'),
            'ev_to_sales': info.get('enterpriseToRevenue'),
            'ev_to_ebitda': info.get('enterpriseToEbitda'),
            'p_fcf': None,  # Calculate if needed
            'ev_to_gross_profit': None  # Calculate if needed
        }
        
        # Calculate P/FCF if we have data
        if info.get('freeCashflow') and info.get('marketCap'):
            stats['valuation_ttm']['p_fcf'] = info.get('marketCap') / info.get('freeCashflow')
        
        # ===== VALUATION (FORWARD) =====
        stats['valuation_forward'] = {
            'price_target': info.get('targetMeanPrice'),
            'forward_pe': info.get('forwardPE'),
            'peg_ratio': info.get('pegRatio'),
            'forward_ev_sales': None,  # Not directly available
            'forward_ev_ebitda': None,
            'forward_p_fcf': None
        }
        
        # ===== FINANCIAL HEALTH =====
        stats['financial_health'] = {
            'cash': info.get('totalCash'),
            'total_debt': info.get('totalDebt'),
            'net_debt': None,  # Calculate
            'debt_to_equity': info.get('debtToEquity'),
            'current_ratio': info.get('currentRatio'),
            'quick_ratio': info.get('quickRatio'),
            'ebit_to_interest': None  # Not directly available
        }
        
        # Calculate net debt
        if info.get('totalDebt') and info.get('totalCash'):
            stats['financial_health']['net_debt'] = info.get('totalDebt') - info.get('totalCash')
        
        # ===== GROWTH =====
        stats['growth'] = {
            'revenue_growth_3y': info.get('revenueGrowth'),  # This is typically YoY
            'revenue_growth_5y': None,  # Would need historical data
            'revenue_growth_10y': None,
            'eps_growth_3y': None,
            'eps_growth_5y': None,
            'eps_growth_10y': None,
            'revenue_growth_fwd_2y': None,
            'ebitda_growth_fwd_2y': None,
            'eps_growth_fwd_2y': None,
            'eps_lt_growth_est': info.get('earningsGrowth')
        }
        
        # ===== DIVIDENDS =====
        stats['dividends'] = {
            'dividend_yield': info.get('dividendYield'),
            'payout_ratio': info.get('payoutRatio'),
            'dividend_per_share': info.get('dividendRate'),
            'ex_dividend_date': info.get('exDividendDate'),
            'dps_growth_3y': None,  # Would need historical data
            'dps_growth_5y': None,
            'dps_growth_10y': None,
            'dps_growth_fwd_2y': None
        }
        
        # ===== ENHANCE WITH FINNHUB DATA =====
        try:
            finnhub_data = get_basic_financials(symbol)
            if finnhub_data and 'metric' in finnhub_data:
                metrics = finnhub_data['metric']
                
                # Fill in missing returns
                if not stats['returns']['roic']:
                    stats['returns']['roic'] = metrics.get('roicTTM')
                if not stats['returns']['roce']:
                    stats['returns']['roce'] = metrics.get('roceTTM')
                
                # Additional valuation metrics
                if not stats['valuation_ttm']['ev_to_ebitda']:
                    stats['valuation_ttm']['ev_to_ebitda'] = metrics.get('evToEbitdaTTM')
                
                # Additional margins
                if not stats['margins']['ebitda_margin']:
                    stats['margins']['ebitda_margin'] = metrics.get('ebitdaMarginTTM')
                
        except Exception as e:
            print(f"⚠️  Could not fetch Finnhub data: {e}")
        
        # ===== ENHANCE WITH ALPHA VANTAGE DATA (use sparingly - 25/day limit) =====
        # Only use if critical data is missing
        # Commented out to preserve API calls
        # try:
        #     av = AlphaVantage()
        #     av_data = av.get_company_overview(symbol)
        #     
        #     if not stats['profile']['market_cap']:
        #         stats['profile']['market_cap'] = av_data.get('MarketCapitalization')
        #     if not stats['returns']['roe']:
        #         stats['returns']['roe'] = av_data.get('ReturnOnEquityTTM')
        #     if not stats['returns']['roa']:
        #         stats['returns']['roa'] = av_data.get('ReturnOnAssetsTTM')
        #         
        # except Exception as e:
        #     print(f"⚠️  Could not fetch Alpha Vantage data: {e}")
        
    except Exception as e:
        print(f"❌ Error fetching statistics for {symbol}: {e}")
    
    return stats


def format_statistics_for_display(stats: dict) -> dict:
    """
    Format statistics for frontend display with proper units and formatting
    
    Args:
        stats: Raw statistics dictionary
    
    Returns:
        Formatted statistics ready for display
    """
    
    def format_currency(value, short=True):
        """Format large numbers as B, M, K"""
        if value is None:
            return "—"
        try:
            value = float(value)
            if short:
                if abs(value) >= 1e9:
                    return f"${value/1e9:.2f}B"
                elif abs(value) >= 1e6:
                    return f"${value/1e6:.2f}M"
                elif abs(value) >= 1e3:
                    return f"${value/1e3:.2f}K"
            return f"${value:,.0f}"
        except:
            return "—"
    
    def format_number(value, suffix=""):
        """Format number with suffix"""
        if value is None:
            return "—"
        try:
            value = float(value)
            if abs(value) >= 1e9:
                return f"{value/1e9:.2f}B{suffix}"
            elif abs(value) >= 1e6:
                return f"{value/1e6:.2f}M{suffix}"
            elif abs(value) >= 1e3:
                return f"{value/1e3:.2f}K{suffix}"
            return f"{value:.0f}{suffix}"
        except:
            return "—"
    
    def format_percent(value, decimals=1):
        """Format as percentage"""
        if value is None:
            return "—"
        try:
            value = float(value) * 100 if value < 1 else float(value)
            return f"{value:.{decimals}f}%"
        except:
            return "—"
    
    def format_ratio(value, decimals=1):
        """Format as ratio"""
        if value is None:
            return "—"
        try:
            return f"{float(value):.{decimals}f}"
        except:
            return "—"
    
    formatted = {
        'profile': {
            'Market Cap': format_currency(stats['profile'].get('market_cap')),
            'Enterprise Value': format_currency(stats['profile'].get('enterprise_value')),
            'Shares Outstanding': format_number(stats['profile'].get('shares_outstanding')),
            'Revenue (TTM)': format_currency(stats['profile'].get('revenue_ttm')),
            'Employees': format_number(stats['profile'].get('employees'), ''),
        },
        'margins': {
            'Gross': format_percent(stats['margins'].get('gross_margin')),
            'EBITDA': format_percent(stats['margins'].get('ebitda_margin')),
            'Operating': format_percent(stats['margins'].get('operating_margin')),
            'Pre-Tax': format_percent(stats['margins'].get('pretax_margin')),
            'Net': format_percent(stats['margins'].get('net_margin')),
            'FCF': format_percent(stats['margins'].get('fcf_margin')),
        },
        'returns': {
            'ROA': format_percent(stats['returns'].get('roa')),
            'ROTA': format_percent(stats['returns'].get('rota')),
            'ROE': format_percent(stats['returns'].get('roe')),
            'ROCE': format_percent(stats['returns'].get('roce')),
            'ROIC': format_percent(stats['returns'].get('roic')),
        },
        'valuation_ttm': {
            'P/E': format_ratio(stats['valuation_ttm'].get('pe_ratio')),
            'P/B': format_ratio(stats['valuation_ttm'].get('pb_ratio')),
            'P/S': format_ratio(stats['valuation_ttm'].get('ps_ratio')),
            'EV/Sales': format_ratio(stats['valuation_ttm'].get('ev_to_sales')),
            'EV/EBITDA': format_ratio(stats['valuation_ttm'].get('ev_to_ebitda')),
            'P/FCF': format_ratio(stats['valuation_ttm'].get('p_fcf')),
            'EV/Gross Profit': format_ratio(stats['valuation_ttm'].get('ev_to_gross_profit')),
        },
        'valuation_forward': {
            'Price Target': format_currency(stats['valuation_forward'].get('price_target'), short=False),
            'Forward P/E': format_ratio(stats['valuation_forward'].get('forward_pe')),
            'PEG': format_ratio(stats['valuation_forward'].get('peg_ratio')),
            'Forward EV/Sales': format_ratio(stats['valuation_forward'].get('forward_ev_sales')),
            'Forward EV/EBITDA': format_ratio(stats['valuation_forward'].get('forward_ev_ebitda')),
            'Forward P/FCF': format_ratio(stats['valuation_forward'].get('forward_p_fcf')),
        },
        'financial_health': {
            'Cash': format_currency(stats['financial_health'].get('cash')),
            'Total Debt': format_currency(stats['financial_health'].get('total_debt')),
            'Net Debt': format_currency(stats['financial_health'].get('net_debt')),
            'Debt/Equity': format_ratio(stats['financial_health'].get('debt_to_equity')),
            'Current Ratio': format_ratio(stats['financial_health'].get('current_ratio')),
            'Quick Ratio': format_ratio(stats['financial_health'].get('quick_ratio')),
            'EBIT/Interest': format_ratio(stats['financial_health'].get('ebit_to_interest')),
        },
        'growth': {
            'Revenue 3Yr': format_percent(stats['growth'].get('revenue_growth_3y')),
            'Revenue 5Yr': format_percent(stats['growth'].get('revenue_growth_5y')),
            'Revenue 10Yr': format_percent(stats['growth'].get('revenue_growth_10y')),
            'EPS 3Yr': format_percent(stats['growth'].get('eps_growth_3y')),
            'EPS 5Yr': format_percent(stats['growth'].get('eps_growth_5y')),
            'EPS 10Yr': format_percent(stats['growth'].get('eps_growth_10y')),
            'Revenue Fwd 2Yr': format_percent(stats['growth'].get('revenue_growth_fwd_2y')),
            'EBITDA Fwd 2Yr': format_percent(stats['growth'].get('ebitda_growth_fwd_2y')),
            'EPS Fwd 2Yr': format_percent(stats['growth'].get('eps_growth_fwd_2y')),
            'EPS LT Growth Est': format_percent(stats['growth'].get('eps_lt_growth_est')),
        },
        'dividends': {
            'Yield': format_percent(stats['dividends'].get('dividend_yield')),
            'Payout Ratio': format_percent(stats['dividends'].get('payout_ratio')),
            'DPS': format_currency(stats['dividends'].get('dividend_per_share'), short=False) if stats['dividends'].get('dividend_per_share') else "—",
            'Ex-Dividend Date': stats['dividends'].get('ex_dividend_date') or "—",
            'DPS Growth 3Yr': format_percent(stats['dividends'].get('dps_growth_3y')),
            'DPS Growth 5Yr': format_percent(stats['dividends'].get('dps_growth_5y')),
            'DPS Growth 10Yr': format_percent(stats['dividends'].get('dps_growth_10y')),
            'DPS Growth Fwd 2Yr': format_percent(stats['dividends'].get('dps_growth_fwd_2y')),
        }
    }
    
    return formatted


# Example usage
if __name__ == "__main__":
    symbol = "NVDA"
    print(f"\n{'='*60}")
    print(f"Company Statistics for {symbol}")
    print(f"{'='*60}\n")
    
    stats = get_company_statistics(symbol)
    formatted = format_statistics_for_display(stats)
    
    for category, data in formatted.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        print("-" * 40)
        for key, value in data.items():
            print(f"{key:20s}: {value}")
