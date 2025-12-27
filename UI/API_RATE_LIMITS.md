# API Rate Limits & Usage Guide

## Overview
This stock analysis platform uses multiple FREE-tier APIs. Understanding rate limits is crucial to avoid hitting quotas.

## API Rate Limits (from keys.txt)

### 1. **Finnhub** - Primary Data Source
- **Limit**: 60 API calls per minute
- **Resets**: Every minute
- **Used For**: 
  - Company news (1 call)
  - Stock quotes (1 call)
  - Company profile (1 call)
  - Basic financials (1 call)
  - Earnings data (1 call)
  - Insider transactions (1 call)
  - Insider sentiment (1 call)

**Per Stock Analysis**: ~5-7 Finnhub calls

### 2. **Alpha Vantage** - News Sentiment
- **Limit**: 25 requests per day 🚨 **VERY LIMITED**
- **Resets**: Daily (midnight UTC)
- **Used For**:
  - News sentiment analysis (1 call per stock)
  - Company overview (1 call per stock)
  - Income statements (1 call per stock)

**Per Stock Analysis**: ~1-3 Alpha Vantage calls

⚠️ **WARNING**: With only 25/day, you can analyze ~8-10 stocks per day if using all features!

### 3. **Tweepy** (Twitter/X API)
- **Status**: Not yet implemented
- **Planned For**: Social sentiment analysis

### 4. **Fiscal.ai**
- **Limit**: 250 calls per day
- **Status**: Not currently used in code
- **Potential Use**: Additional financial data

## Function Call Breakdown

### `get_comprehensive_sentiment(symbol)` 
API calls per stock:
- ✅ Finnhub news: 1 call (via `analyze_stock_sentiment`)
- ✅ Finnhub earnings: 1 call (via `analyze_stock_sentiment`)
- ✅ Alpha Vantage sentiment: 1 call 🚨
- ✅ Finnhub insider sentiment: 1 call
- **Total**: ~4 calls (1 Alpha Vantage, 3 Finnhub)

### `get_comprehensive_data(symbol)`
API calls per stock:
- ✅ Company profile: 1 Finnhub call
- ✅ Stock quote: 1 Finnhub call
- ✅ Basic financials: 1 Finnhub call
- ✅ Company news: 1 Finnhub call
- ✅ Earnings surprises: 1 Finnhub call
- ✅ Insider transactions: 1 Finnhub call
- ✅ Comprehensive sentiment: 4 calls (includes AV)
- ✅ Alpha Vantage overview: 1 call 🚨
- ✅ Alpha Vantage financials: 1 call 🚨
- **Total**: ~12 calls (3 Alpha Vantage, 9 Finnhub)

## Usage Scenarios

### Scenario 1: Analyzing 1 Stock (All Features)
```python
analyzer = StockAnalyzer('AAPL')
data = analyzer.get_comprehensive_data()
```
- **Finnhub**: 9 calls (well within 60/min)
- **Alpha Vantage**: 3 calls (12% of daily quota)
- **Can analyze**: 8 stocks/day before hitting AV limit

### Scenario 2: Sentiment Only (Multiple Stocks)
```python
for symbol in ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA']:
    analyzer = StockAnalyzer(symbol)
    sentiment = analyzer.get_comprehensive_sentiment()
```
- **Finnhub**: 15 calls (25% of minute quota)
- **Alpha Vantage**: 5 calls (20% of daily quota)
- **Can analyze**: 25 stocks/day (sentiment only)

### Scenario 3: Watchlist Price Updates (No Sentiment)
```python
for symbol in watchlist:
    # Just get quote - no sentiment
    quote = get_stock_quote(symbol)
```
- **Finnhub**: 1 call per stock
- **Alpha Vantage**: 0 calls
- **Can analyze**: 60 stocks/minute

## Best Practices

### 1. **Cache Aggressively**
```python
# Cache data for 5-15 minutes
stockDataCache[symbol] = {
    'timestamp': Date.now(),
    'data': data,
    'ttl': 5 * 60 * 1000  # 5 minutes
}
```

### 2. **Prioritize Data Sources**
- Use **Finnhub** for real-time quotes, news, earnings (plenty of calls)
- Use **Alpha Vantage** sparingly for sentiment analysis only
- Skip Alpha Vantage financial data if Finnhub provides it

### 3. **Batch Requests Carefully**
```javascript
// DON'T: Load all stocks at once
watchlist.forEach(stock => loadSentiment(stock));  // Uses 5 AV calls instantly!

// DO: Load prices first, sentiment on demand
watchlist.forEach(stock => loadQuote(stock));  // Uses only Finnhub
// User clicks stock → THEN load sentiment
```

### 4. **Monitor Usage**
Add console warnings when approaching limits:
```javascript
let avCallsToday = 0;

async function callAlphaVantage() {
    avCallsToday++;
    if (avCallsToday > 20) {
        console.warn(`⚠️  Alpha Vantage: ${avCallsToday}/25 calls used today!`);
    }
    // ... make call
}
```

### 5. **Implement Fallbacks**
```python
try:
    av_sentiment = av.get_market_news_sentiment(symbol)
except Exception as e:
    if "rate limit" in str(e).lower():
        print("⚠️  Alpha Vantage rate limit hit - using Finnhub only")
        # Continue with just FinBERT + Finnhub
```

## Error Messages to Watch For

### Finnhub
- **"API rate limit reached"**: Wait 1 minute
- **Status 429**: Too many requests

### Alpha Vantage
- **"Thank you for using Alpha Vantage! Our standard API call frequency is 25 requests per day"**
- **Status 429**: Wait until next day

## Optimization Strategies

### Current Issue: 0 Articles Analyzed
Your error `"0 articles analyzed"` suggests:
1. ✅ **FIXED**: Added `articles_analyzed` field to finbert.py return value
2. Check if Finnhub news is actually returning articles
3. Verify date range (last 7 days might have no news for some stocks)

### Reduce Alpha Vantage Usage
**Option A**: Disable AV sentiment for watchlist updates
```python
def get_lightweight_sentiment(self):
    """Only use FinBERT + Finnhub (no Alpha Vantage)"""
    # Skip Alpha Vantage to save quota
```

**Option B**: Use AV only on user demand
```javascript
// Only load AV when user explicitly requests full analysis
if (userClickedDetailedAnalysis) {
    loadSentiment(symbol);  // Includes AV
} else {
    loadBasicSentiment(symbol);  // FinBERT + Finnhub only
}
```

## Daily Usage Capacity

With current architecture:

| Feature | Stocks/Day | Bottleneck |
|---------|-----------|------------|
| Full Analysis (all features) | **8 stocks** | Alpha Vantage (25/day) |
| Sentiment Only | **25 stocks** | Alpha Vantage (25/day) |
| Sentiment (no AV) | **Unlimited*** | Finnhub (60/min)* |
| Price Updates | **Unlimited*** | Finnhub (60/min)* |

\* Finnhub allows 60/min = 3,600/hour = 86,400/day theoretically

## Recommendations

1. **For Watchlist** (multiple stocks):
   - Load prices only (Finnhub quotes)
   - Show cached sentiment from last analysis
   - Update sentiment only when user clicks

2. **For Individual Stock**:
   - Use full comprehensive analysis
   - Show user "Analyzing from 3 sources..." message
   - Cache results for 15 minutes

3. **For Production**:
   - Consider upgrading to paid tier:
     - Finnhub Pro: $60/month → 300 calls/min
     - Alpha Vantage: $50/month → 75 calls/min
   - Or switch to alternatives for sentiment (Twitter API, Reddit API)

## Testing Today's Fix

The update to `finbert.py` should now show proper article counts:

```bash
# Test sentiment analysis
cd UI
python stock_analyzer.py AAPL

# Should now show:
# "✓ Analyzed 5 articles" (instead of 0)
```

If still showing 0 articles, check:
1. Finnhub API key is valid
2. News articles exist for date range
3. Finnhub hasn't changed API response format
