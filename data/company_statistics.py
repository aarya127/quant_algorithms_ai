"""
Company Statistics Module
Aggregates comprehensive company statistics from multiple sources:
- yfinance (primary source - free, unlimited)
- Alpha Vantage (fallback for missing data)
- Finnhub (additional metrics)
"""

import datetime

import yfinance as yf
from data.alphavantage import AlphaVantage
from data.finnhub import get_basic_financials


def _epoch_to_date(value):
    """yfinance gives exDividendDate as a Unix timestamp; show it as YYYY-MM-DD."""
    if isinstance(value, (int, float)):
        return datetime.datetime.fromtimestamp(value, datetime.timezone.utc).strftime('%Y-%m-%d')
    return value


def _times(a, b):
    """a * b, or None when either is missing (Finnhub nulls, no share count)."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a * b
    return None


def _ratio(a, b):
    """a / b, or None when either is missing or b is zero."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b:
        return a / b
    return None


def get_company_statistics(symbol: str, info: dict = None, financials: dict = None,
                           profile: dict = None, cash_flow: dict = None) -> dict:
    """
    Get comprehensive company statistics including:
    - Profile (Market Cap, EV, Shares, Revenue, Employees)
    - Margins (Gross, EBITDA, Operating, Pre-Tax, Net, FCF)
    - Returns (ROA, ROTA, ROE, ROCE, ROIC)
    - Valuation (P/E, P/B, EV/Sales, EV/EBITDA, P/FCF)
    - Financial Health (Cash, Debt, Ratios)
    - Growth (CAGR for Revenue and EPS)
    - Dividends
    - Cash Flow (Operating CF, CapEx, FCF, SBC, D&A, buybacks, dividends paid)

    Args:
        symbol: Stock ticker symbol
        info: pre-fetched yfinance Ticker.info dict (fetched here if None)
        financials: pre-fetched Finnhub basic financials (fetched here if None)
        profile: Finnhub company profile (share count); not fetched when None
        cash_flow: financial_statements.latest_ttm() from SEC filings; not fetched when None

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
        'dividends': {},
        'cash_flow': {},
    }

    def put(section, field, value):
        """Fill a stat only if no earlier (preferred) source set it."""
        if stats[section].get(field) is None and value is not None:
            stats[section][field] = value
    
    # Primary source: yfinance (most comprehensive and free). An empty dict
    # means the caller already tried and Yahoo failed — don't fetch again.
    if info is None:
        try:
            info = yf.Ticker(symbol).info
        except Exception as e:
            print(f"⚠️  yfinance info failed for {symbol}: {e}")
    info = info or {}

    # Unit convention: every percent-type stat is stored as a FRACTION
    # (0.25 = 25%). yfinance mostly returns fractions; Finnhub returns
    # percentages, so its values are divided by 100 where they are filled in.
    try:
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
            'pretax_margin': None,  # yfinance has none; filled from Finnhub below
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
            # yfinance reports debtToEquity in percent (150 = 1.5x); stored as a ratio
            'debt_to_equity': (info['debtToEquity'] / 100
                               if info.get('debtToEquity') is not None else None),
            'current_ratio': info.get('currentRatio'),
            'quick_ratio': info.get('quickRatio'),
            'ebit_to_interest': None  # Not directly available
        }
        
        
        # ===== GROWTH =====
        stats['growth'] = {
            'revenue_growth_3y': None,  # yfinance revenueGrowth is quarterly YoY, not 3Y
            'revenue_growth_5y': None,  # Would need historical data
            'revenue_growth_10y': None,
            'eps_growth_3y': None,
            'eps_growth_5y': None,
            'eps_growth_10y': None,
            'revenue_growth_fwd_2y': None,
            'ebitda_growth_fwd_2y': None,
            'eps_growth_fwd_2y': None,
            'eps_lt_growth_est': None,  # yfinance earningsGrowth is quarterly YoY, not LT
        }
        
        # ===== DIVIDENDS =====
        stats['dividends'] = {
            # yfinance reports dividendYield in percent (0.49 = 0.49%)
            'dividend_yield': (info['dividendYield'] / 100
                               if info.get('dividendYield') is not None else None),
            'payout_ratio': info.get('payoutRatio'),
            'dividend_per_share': info.get('dividendRate'),
            'ex_dividend_date': _epoch_to_date(info.get('exDividendDate')),
            'dps_growth_3y': None,  # Would need historical data
            'dps_growth_5y': None,
            'dps_growth_10y': None,
            'dps_growth_fwd_2y': None
        }
        
        # ===== ENHANCE WITH FINNHUB DATA =====
        try:
            finnhub_data = financials if financials is not None else get_basic_financials(symbol)
            if finnhub_data and 'metric' in finnhub_data:
                metrics = finnhub_data['metric']

                def fill(section, field, metric_key, percent=True):
                    """Fill a missing stat from Finnhub (percent values -> fraction)."""
                    if stats[section].get(field) is None and metrics.get(metric_key) is not None:
                        value = metrics[metric_key]
                        stats[section][field] = value / 100 if percent else value

                # Returns / margins — also the fallback when yfinance is down
                fill('returns', 'roa', 'roaTTM')
                fill('returns', 'roe', 'roeTTM')
                fill('returns', 'roic', 'roiTTM')
                fill('margins', 'gross_margin', 'grossMarginTTM')
                fill('margins', 'operating_margin', 'operatingMarginTTM')
                fill('margins', 'pretax_margin', 'pretaxMarginTTM')
                fill('margins', 'net_margin', 'netProfitMarginTTM')

                # Valuation / health ratios (plain multiples, no unit change)
                fill('valuation_ttm', 'pe_ratio', 'peTTM', percent=False)
                fill('valuation_ttm', 'pb_ratio', 'pb', percent=False)
                fill('valuation_ttm', 'ps_ratio', 'psTTM', percent=False)
                fill('valuation_ttm', 'ev_to_sales', 'evRevenueTTM', percent=False)
                fill('valuation_ttm', 'ev_to_ebitda', 'evEbitdaTTM', percent=False)
                fill('financial_health', 'current_ratio', 'currentRatioQuarterly', percent=False)
                fill('financial_health', 'quick_ratio', 'quickRatioQuarterly', percent=False)
                fill('financial_health', 'ebit_to_interest', 'netInterestCoverageTTM', percent=False)

                # Multi-year growth (yfinance only has quarterly YoY)
                fill('growth', 'revenue_growth_3y', 'revenueGrowth3Y')
                fill('growth', 'revenue_growth_5y', 'revenueGrowth5Y')
                fill('growth', 'eps_growth_3y', 'epsGrowth3Y')
                fill('growth', 'eps_growth_5y', 'epsGrowth5Y')
                fill('dividends', 'dps_growth_5y', 'dividendGrowthRate5Y')

                # Profile / valuation the yfinance fields above cover when Yahoo answers
                fill('valuation_ttm', 'p_fcf', 'pfcfShareTTM', percent=False)
                fill('valuation_forward', 'forward_pe', 'forwardPE', percent=False)
                fill('valuation_forward', 'peg_ratio', 'pegTTM', percent=False)
                fill('financial_health', 'debt_to_equity', 'totalDebt/totalEquityQuarterly', percent=False)
                fill('dividends', 'dividend_yield', 'currentDividendYieldTTM')
                fill('dividends', 'payout_ratio', 'payoutRatioTTM')
                fill('dividends', 'dividend_per_share', 'dividendPerShareTTM', percent=False)
                # Finnhub market cap / EV are in millions
                put('profile', 'market_cap', _times(metrics.get('marketCapitalization'), 1e6))
                put('profile', 'enterprise_value', _times(metrics.get('enterpriseValue'), 1e6))

                # Per-share metrics -> totals, with the profile's share count (millions)
                shares = _times((profile or {}).get('shareOutstanding'), 1e6)
                put('profile', 'shares_outstanding', shares)
                put('profile', 'revenue_ttm', _times(metrics.get('revenuePerShareTTM'), shares))
                put('financial_health', 'cash', _times(metrics.get('cashPerSharePerShareQuarterly'), shares))
                equity = _times(metrics.get('bookValuePerShareQuarterly'), shares)
                put('financial_health', 'total_debt',
                    _times(metrics.get('totalDebt/totalEquityQuarterly'), equity))
                # revenue per employee is in $ millions
                rev_per_emp = _times(metrics.get('revenueEmployeeTTM'), 1e6)
                if stats['profile'].get('revenue_ttm') and rev_per_emp:
                    put('profile', 'employees', round(stats['profile']['revenue_ttm'] / rev_per_emp))

                # Margins as ratios of per-share TTM figures
                put('margins', 'ebitda_margin',
                    _ratio(metrics.get('ebitdPerShareTTM'), metrics.get('revenuePerShareTTM')))
                # P/S over P/FCF = FCF / sales
                put('margins', 'fcf_margin', _ratio(metrics.get('psTTM'), metrics.get('pfcfShareTTM')))

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

        # ===== CASH FLOW + CAPITAL RETURNS (SEC filings, data/financial_statements.py) =====
        cf = cash_flow or {}
        ocf, fcf, sbc = cf.get('operating_cash_flow'), cf.get('free_cash_flow'), cf.get('stock_comp')
        stats['cash_flow'] = {
            'operating_cash_flow': ocf,
            'capex': cf.get('capex'),
            'free_cash_flow': fcf,
            'capex_to_revenue': _ratio(cf.get('capex'), cf.get('revenue')),
            'stock_comp': sbc,
            'fcf_after_sbc': fcf - sbc if fcf is not None and sbc is not None else None,
            'depreciation_amortization': cf.get('depreciation_amortization'),
            'buybacks': cf.get('buybacks'),
            'dividends_paid': cf.get('dividends_paid'),
            'basis': cf.get('basis'),
            'period_end': cf.get('period_end'),
        }
        put('profile', 'revenue_ttm', cf.get('revenue'))
        put('margins', 'fcf_margin', _ratio(fcf, cf.get('revenue')))
        mcap = stats['profile'].get('market_cap')
        if fcf and fcf > 0:
            put('valuation_ttm', 'p_fcf', _ratio(mcap, fcf))
        # EBIT ~ operating income; ROTA on total assets, ROCE on assets - current liabilities
        assets, cur_liab = cf.get('total_assets'), cf.get('current_liabilities')
        put('returns', 'rota', _ratio(cf.get('operating_income'), assets))
        if assets is not None and cur_liab is not None:
            put('returns', 'roce', _ratio(cf.get('operating_income'), assets - cur_liab))

        # ===== DERIVED FROM WHATEVER SOURCE FILLED THE INPUTS =====
        gross = stats['margins'].get('gross_margin')
        if gross and gross > 0:
            put('valuation_ttm', 'ev_to_gross_profit',
                _ratio(stats['valuation_ttm'].get('ev_to_sales'), gross))
        health = stats['financial_health']
        if health.get('total_debt') is not None and health.get('cash') is not None:
            put('financial_health', 'net_debt', health['total_debt'] - health['cash'])

    except Exception as e:
        print(f"❌ Error building statistics for {symbol}: {e}")
    
    return stats


def has_any_statistic(stats: dict) -> bool:
    """True when at least one stat was populated (False = every upstream failed)."""
    return any(v is not None for section in stats.values() for v in section.values())


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
            sign, value = ("-" if value < 0 else ""), abs(value)
            if short:
                if value >= 1e9:
                    return f"{sign}${value/1e9:.2f}B"
                elif value >= 1e6:
                    return f"{sign}${value/1e6:.2f}M"
                elif value >= 1e3:
                    return f"{sign}${value/1e3:.2f}K"
            return f"{sign}${value:,.2f}"
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
        """Format a fraction as a percentage (0.25 -> "25.0%"; 1.17 -> "117.0%")"""
        if value is None:
            return "—"
        try:
            return f"{float(value) * 100:.{decimals}f}%"
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
    
    cash_flow = stats.get('cash_flow') or {}
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
            'Debt/Equity': format_ratio(stats['financial_health'].get('debt_to_equity'), decimals=2),
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
        },
        'cash_flow': {
            'Operating Cash Flow': format_currency(cash_flow.get('operating_cash_flow')),
            'CapEx': format_currency(cash_flow.get('capex')),
            'Free Cash Flow': format_currency(cash_flow.get('free_cash_flow')),
            'CapEx / Revenue': format_percent(cash_flow.get('capex_to_revenue')),
            'Stock-Based Comp': format_currency(cash_flow.get('stock_comp')),
            'FCF after SBC': format_currency(cash_flow.get('fcf_after_sbc')),
            'D&A': format_currency(cash_flow.get('depreciation_amortization')),
            'Share Buybacks': format_currency(cash_flow.get('buybacks')),
            'Dividends Paid': format_currency(cash_flow.get('dividends_paid')),
            # header note: "TTM to 2026-09-26" / "FY ending 2026-09-26" (None = no SEC filings)
            'Period': ((('TTM to ' if cash_flow.get('basis') == 'ttm' else 'FY ending ')
                        + cash_flow['period_end']) if cash_flow.get('period_end') else None),
        },
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
