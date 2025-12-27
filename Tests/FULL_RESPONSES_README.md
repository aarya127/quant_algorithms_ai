# Full API Response Test Results

## Summary

✅ **All tests passed: 17/17 (100%)**
- Finnhub API: 7/7 endpoints ✓
- Alpha Vantage API: 10/10 endpoints ✓

## Location

All full responses saved to:
```
Tests/results/full_responses/
```

## What You Have

### Finnhub API Responses (7 files)

1. **01_get_company_news_NVDA_*.json**
   - Last 30 days of NVDA news articles
   - Complete with headlines, summaries, URLs, sources

2. **02_get_basic_financials_NVDA_*.json**
   - Comprehensive financial metrics
   - P/E ratio, beta, market cap, profit margins, etc.

3. **03_get_earnings_surprises_NVDA_*.json**
   - Historical earnings vs estimates
   - Actual vs expected EPS with surprise percentage

4. **04_get_earnings_calendar_NVDA_*.json**
   - Upcoming earnings dates (may be empty if none scheduled)

5. **05_get_insider_transactions_NVDA_*.json**
   - Executive stock transactions
   - Buys, sells, shares, prices, dates

6. **06_get_insider_sentiment_NVDA_*.json**
   - Aggregated insider trading sentiment
   - MSPR (Monthly Shares Purchased Ratio)

7. **07_get_usa_spending_NVDA_*.json**
   - Government contracts (if any)

### Alpha Vantage API Responses (10 files)

1. **01_get_market_news_sentiment_NVDA_*.json**
   - 50 recent news articles with sentiment analysis
   - Overall sentiment scores + per-ticker sentiment
   - Relevance scores for each article
   - Example: "AI wobble casts shadow" (Somewhat-Bullish 0.16)

2. **02_get_company_overview_NVDA_*.json**
   - Complete company profile
   - Industry, sector, description, address
   - Market cap, P/E, dividend info, fiscal year

3. **03_get_global_quote_NVDA_*.json**
   - Real-time quote data
   - Open, high, low, close, volume
   - Price change and percent change

4. **04_get_income_statement_NVDA_*.json**
   - Annual and quarterly income statements
   - Revenue, gross profit, operating income
   - Net income, EPS, EBITDA

5. **05_get_balance_sheet_NVDA_*.json**
   - Assets, liabilities, shareholder equity
   - Current assets, long-term debt
   - Retained earnings

6. **06_get_cash_flow_NVDA_*.json**
   - Operating, investing, financing cash flows
   - Free cash flow, capital expenditures
   - Dividend payments

7. **07_get_earnings_history_NVDA_*.json**
   - Historical earnings reports
   - Reported vs estimated EPS
   - Surprise amounts and percentages

8. **08_get_insider_transactions_NVDA_*.json**
   - Detailed insider trading records
   - Names, positions, transaction types
   - Shares and acquisition/disposal codes

9. **09_get_sma_NVDA_*.json**
   - Simple Moving Average (20-day)
   - Daily time series data
   - Historical SMA values

10. **10_get_rsi_NVDA_*.json**
    - Relative Strength Index (14-day)
    - Momentum indicator
    - Overbought/oversold signals

## File Format

Each JSON file contains:
```json
{
  "test_number": 1,
  "timestamp": "2025-11-09T00:45:46.013228",
  "endpoint": "get_market_news_sentiment",
  "symbol": "NVDA",
  "status": "PASS",
  "error": null,
  "response": {
    // COMPLETE API RESPONSE HERE
    // All data, no truncation
  }
}
```

## Example: News Sentiment Data

From `01_get_market_news_sentiment_NVDA_*.json`:
- **50 news articles** with full text summaries
- Each article includes:
  - Title, URL, publication time
  - Overall sentiment score (-1 to +1)
  - Sentiment label (Bearish, Neutral, Bullish, etc.)
  - Per-ticker sentiment for NVDA and related stocks
  - Relevance scores
  - Topics (Technology, Manufacturing, Finance, etc.)

**Sample headline**: "Global week ahead: AI wobble casts shadow over 'Davos for geeks'"
- Overall: Somewhat-Bullish (0.16)
- NVDA sentiment: Neutral (0.09)

## How to Use These Files

### View in VS Code
Just open any `.json` file in the `full_responses/` folder

### Load in Python
```python
import json

# Load any response
with open('Tests/results/full_responses/01_get_market_news_sentiment_NVDA_*.json') as f:
    data = json.load(f)
    
# Access the API response
response = data['response']

# For news sentiment
news_articles = response['feed']
for article in news_articles[:5]:  # First 5 articles
    print(f"Title: {article['title']}")
    print(f"Sentiment: {article['overall_sentiment_label']}")
    print(f"NVDA sentiment: {article['ticker_sentiment'][0]['ticker_sentiment_score']}")
    print()
```

### Analyze All Data
```python
import os
import json

# Load all Alpha Vantage responses
av_files = [f for f in os.listdir('Tests/results/full_responses/') 
            if 'NVDA_20251109_004545' in f]

for file in av_files:
    with open(f'Tests/results/full_responses/{file}') as f:
        data = json.load(f)
        print(f"{data['endpoint']}: {data['status']}")
```

## File Sizes

The JSON files range from small (a few KB) to large:
- News sentiment: ~200KB (50 articles with full text)
- Insider transactions: Can be 500KB+ (extensive transaction data)
- Technical indicators: ~50KB (time series data)
- Company overview: ~5KB (concise profile)

## Next Steps

You now have **complete, untruncated access** to all API endpoints for NVDA. You can:

1. **Analyze the data** - Open any JSON file to see full responses
2. **Build features** - Use this data for sentiment analysis, trend detection
3. **Compare APIs** - See which provides better data for your needs
4. **Integrate** - Load these responses in your analysis scripts

All data is real, current (as of Nov 9, 2025), and fully captured!
