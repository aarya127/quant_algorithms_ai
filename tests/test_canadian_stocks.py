"""
Test script to check if APIs work with Canadian stocks
Canadian stocks typically use .TO suffix for Toronto Stock Exchange
Example: TD.TO (TD Bank), RY.TO (Royal Bank), SHOP.TO (Shopify)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Data import alphavantage, finnhub
import datetime

def test_canadian_stocks():
    """Test various Canadian stock symbols with different APIs"""
    
    # Common Canadian stock symbols
    canadian_symbols = [
        'TD.TO',      # TD Bank
        'RY.TO',      # Royal Bank of Canada
        'SHOP.TO',    # Shopify (also trades as SHOP on NYSE)
        'ENB.TO',     # Enbridge
        'CNQ.TO',     # Canadian Natural Resources
    ]
    
    # Also test the US-listed Canadian companies
    us_listed_canadian = [
        'SHOP',       # Shopify on NYSE
        'CNQ',        # Canadian Natural on NYSE
    ]
    
    print("="*80)
    print("TESTING CANADIAN STOCKS WITH APIs")
    print("="*80)
    
    # Test AlphaVantage
    print("\n" + "="*80)
    print("1. TESTING ALPHAVANTAGE API")
    print("="*80)
    
    try:
        av = alphavantage.AlphaVantage()
        
        for symbol in canadian_symbols[:2]:  # Test first 2 to avoid rate limits
            print(f"\n--- Testing {symbol} ---")
            
            # Test Company Overview
            print("\nCompany Overview:")
            overview = av.get_company_overview(symbol)
            if overview and 'Symbol' in overview:
                print(f"✓ Success! Company: {overview.get('Name', 'N/A')}")
                print(f"  Exchange: {overview.get('Exchange', 'N/A')}")
                print(f"  Currency: {overview.get('Currency', 'N/A')}")
                print(f"  Country: {overview.get('Country', 'N/A')}")
            else:
                print(f"✗ Failed or no data returned")
                print(f"  Response: {overview}")
            
            # Test Global Quote
            print("\nGlobal Quote:")
            quote = av.get_global_quote(symbol)
            if quote and quote.get('01. symbol'):
                print(f"✓ Success! Price: ${quote.get('05. price', 'N/A')}")
                print(f"  Volume: {quote.get('06. volume', 'N/A')}")
            else:
                print(f"✗ Failed or no data returned")
                print(f"  Response: {quote}")
                
    except Exception as e:
        print(f"✗ AlphaVantage Error: {str(e)}")
    
    # Test Finnhub
    print("\n" + "="*80)
    print("2. TESTING FINNHUB API")
    print("="*80)
    
    try:
        for symbol in canadian_symbols[:2]:  # Test first 2 to avoid rate limits
            print(f"\n--- Testing {symbol} ---")
            
            # Test Company Profile
            print("\nCompany Profile:")
            profile = finnhub.get_company_profile(symbol)
            if profile and profile.get('ticker'):
                print(f"✓ Success! Company: {profile.get('name', 'N/A')}")
                print(f"  Exchange: {profile.get('exchange', 'N/A')}")
                print(f"  Country: {profile.get('country', 'N/A')}")
                print(f"  Currency: {profile.get('currency', 'N/A')}")
            else:
                print(f"✗ Failed or no data returned")
                print(f"  Response: {profile}")
            
            # Test Stock Quote
            print("\nStock Quote:")
            quote = finnhub.get_stock_quote(symbol)
            if quote and quote.get('c'):  # 'c' is current price
                print(f"✓ Success! Current Price: ${quote.get('c', 'N/A')}")
                print(f"  High: ${quote.get('h', 'N/A')}")
                print(f"  Low: ${quote.get('l', 'N/A')}")
            else:
                print(f"✗ Failed or no data returned")
                print(f"  Response: {quote}")
            
            # Test Company News
            print("\nCompany News:")
            today = datetime.date.today()
            from_date = (today - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
            to_date = today.strftime('%Y-%m-%d')
            news = finnhub.get_company_news(symbol, from_date, to_date)
            if news and len(news) > 0:
                print(f"✓ Success! Found {len(news)} news articles")
                print(f"  Latest: {news[0].get('headline', 'N/A')[:60]}...")
            else:
                print(f"✗ No news found or failed")
                print(f"  Response: {news}")
                
    except Exception as e:
        print(f"✗ Finnhub Error: {str(e)}")
    
    # Test US-listed Canadian companies
    print("\n" + "="*80)
    print("3. TESTING US-LISTED CANADIAN COMPANIES")
    print("="*80)
    
    try:
        av = alphavantage.AlphaVantage()
        
        for symbol in us_listed_canadian[:1]:  # Test first one
            print(f"\n--- Testing {symbol} (US-listed) ---")
            
            overview = av.get_company_overview(symbol)
            if overview and 'Symbol' in overview:
                print(f"✓ Success! Company: {overview.get('Name', 'N/A')}")
                print(f"  Country: {overview.get('Country', 'N/A')}")
                print(f"  Exchange: {overview.get('Exchange', 'N/A')}")
            else:
                print(f"✗ Failed")
                
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
Canadian Stock Symbol Formats:
1. TSX symbols use .TO suffix (e.g., TD.TO, RY.TO)
2. Some Canadian companies also trade on US exchanges (e.g., SHOP on NYSE)

API Compatibility:
- AlphaVantage: Check results above
- Finnhub: Check results above

Note: Some APIs may have better coverage for US-listed stocks vs TSX-listed stocks.
For best results with Canadian stocks, you may want to:
1. Use US ticker when available (e.g., SHOP instead of SHOP.TO)
2. Check if your API plan includes international/Canadian market data
3. Consider using TSX-specific data providers for complete coverage
    """)

if __name__ == "__main__":
    test_canadian_stocks()
