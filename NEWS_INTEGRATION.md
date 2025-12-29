# Stock Predictor - Market News Integration

## 🚀 New Features

### Real-time Multi-Source News Feed
The Market News section now integrates **three powerful data sources** to provide comprehensive market coverage:

1. **Twitter Feed** (via Tweepy)
   - Live tweets from major financial news accounts (WSJ, Bloomberg, Reuters, CNBC, etc.)
   - Symbol-specific tweets with $cashtag tracking
   - Engagement metrics (likes, retweets, replies)
   - Verified account indicators

2. **Alpaca Real-time News Stream** (WebSocket)
   - Live streaming news from professional sources
   - Real-time updates as news breaks
   - Multi-symbol coverage
   - Professional journalism sources (Benzinga, etc.)

3. **Finnhub News API**
   - Company-specific news articles
   - Historical news archives
   - Rich article summaries

## 📊 Features

### News Section
- **Combined Feed**: All three sources displayed together
- **Smart Filtering**:
  - Filter by source (Twitter, Alpaca, Finnhub, or All)
  - Filter by stock symbol (e.g., AAPL, TSLA)
  - Adjustable count (15, 30, 50, or 100 items)
- **Real-time Updates**: Live streaming badge indicator
- **Rich Display**:
  - Twitter cards with profile pictures and verification badges
  - Article cards with source badges and timestamps
  - Symbol tags for quick reference
  - Engagement metrics for tweets

### Navigation
- **Dashboard**: Main stock analysis hub
- **News**: Dedicated market news section
- **Calendar**: Earnings calendar
- **Quant**: Quantitative analysis tools (coming soon)

## 🔑 API Keys Required

### Twitter API (Tweepy)
```
API Key: ***REMOVED***
API Secret: ***REMOVED***
Bearer Token: AAAAAAAAAAAAAAAAAAAAAFJd5gEA...
```

### Alpaca API
```
API Key: ***REMOVED***
Secret Key: ***REMOVED***
```

### Finnhub API
```
API Key: ***REMOVED***
Rate Limit: 60 calls/minute
```

## 🛠️ Installation

### Prerequisites
```bash
# Ensure Python 3.12+ is installed
python --version

# Activate virtual environment
cd /Users/aaryas127/Documents/GitHub/stock_predictor
source .venv/bin/activate
```

### Install Dependencies
```bash
pip install tweepy==4.16.0 websockets==15.0.1
```

Or install from requirements.txt:
```bash
cd UI
pip install -r requirements.txt
```

## 🚦 Running the Application

### Start the Server
```bash
cd /Users/aaryas127/Documents/GitHub/stock_predictor
source .venv/bin/activate
cd UI
python app.py
```

The application will:
1. Load the FinBERT sentiment model
2. Start the Alpaca WebSocket news stream
3. Launch Flask on http://127.0.0.1:5000

### Access the Application
Open your browser and navigate to:
```
http://localhost:5000
```

## 📡 API Endpoints

### News Endpoints

#### `/api/news/twitter`
Get latest market news from Twitter

**Query Parameters:**
- `symbol` (optional): Stock symbol to filter (e.g., AAPL)
- `count` (default: 20): Number of tweets (max: 100)
- `type`: 'market' or 'financial'

**Example:**
```bash
curl "http://localhost:5000/api/news/twitter?symbol=AAPL&count=10"
```

#### `/api/news/alpaca`
Get real-time news from Alpaca stream

**Query Parameters:**
- `symbol` (optional): Stock symbol to filter
- `count` (default: 20): Number of items

**Example:**
```bash
curl "http://localhost:5000/api/news/alpaca?count=20"
```

#### `/api/news/combined`
Get combined news from all sources

**Query Parameters:**
- `symbol` (optional): Stock symbol to filter
- `count` (default: 10): Items per source

**Example:**
```bash
curl "http://localhost:5000/api/news/combined?symbol=TSLA&count=15"
```

## 🏗️ Architecture

### Backend Components

#### `Data/twitter_feed.py`
- Tweepy integration
- Tweet fetching and formatting
- User timeline retrieval
- Financial news account aggregation

#### `Data/alpaca_news.py`
- WebSocket client for Alpaca
- Real-time news streaming
- News caching in memory
- Async/await pattern

#### `UI/app.py`
- Flask routes for news endpoints
- Integration layer between data sources
- Error handling and graceful degradation

### Frontend Components

#### `UI/templates/index.html`
- News section with filter controls
- Source/symbol/count selectors
- Real-time streaming indicator
- Responsive grid layout

#### `UI/static/js/main.js`
- News loading and display logic
- Client-side filtering
- Dynamic card generation
- Twitter vs. article differentiation

## 🎨 UI Features

### News Cards

**Twitter Cards:**
- Profile picture
- Username and display name
- Verification badge
- Tweet text
- Symbol tags
- Engagement metrics (likes, retweets)
- Link to original tweet

**Article Cards:**
- Source badge (Finnhub/Alpaca)
- Timestamp
- Author name
- Headline
- Summary
- Symbol tags
- Read more link

### Filters
- **Source Filter**: All, Twitter, Alpaca, Finnhub
- **Symbol Filter**: Text input for stock symbols
- **Count Filter**: 15, 30, 50, 100 items

## ⚠️ Known Issues & Solutions

### Alpaca Authentication Timeout
**Issue:** `ERROR:Data.alpaca_news:Alpaca error: auth timeout`

**Cause:** Free tier Alpaca API has limited WebSocket access

**Solution:** The app gracefully handles this - news from Twitter and Finnhub will still work

### Twitter Rate Limits
**Limit:** Twitter API has rate limits for searches

**Solution:** App uses `wait_on_rate_limit=True` to automatically wait when rate limited

### Finnhub Rate Limits
**Limit:** 60 API calls per minute

**Solution:** Caching and smart request batching

## 📈 Rate Limits Summary

| Source | Limit | Notes |
|--------|-------|-------|
| Finnhub | 60/min | Free tier |
| Twitter | Variable | Depends on endpoint |
| Alpaca | 200/min | WebSocket may need paid tier |
| Alpha Vantage | 25/day | Used for sentiment only |

## 🔮 Future Enhancements

### Planned Features
- [ ] Real-time news notifications
- [ ] Sentiment analysis on tweets
- [ ] News aggregation by topic
- [ ] Historical news search
- [ ] Custom RSS feed integration
- [ ] News impact on stock prices
- [ ] AI-powered news summarization

### Potential Integrations
- [ ] Reddit r/wallstreetbets sentiment
- [ ] News API for broader coverage
- [ ] Discord market channels
- [ ] Telegram financial channels

## 🐛 Troubleshooting

### Server Won't Start
```bash
# Check if port 5000 is already in use
lsof -ti:5000

# Kill existing process
pkill -f "python app.py"

# Restart
python app.py
```

### Missing Dependencies
```bash
# Reinstall all requirements
pip install -r requirements.txt --force-reinstall
```

### API Key Issues
- Check `keys.txt` file has correct credentials
- Verify API keys are active in respective platforms
- Check rate limits haven't been exceeded

## 📝 File Structure

```
stock_predictor/
├── Data/
│   ├── twitter_feed.py       # Twitter integration
│   ├── alpaca_news.py         # Alpaca WebSocket client
│   ├── finnhub.py             # Finnhub API wrapper
│   └── charts.py              # yfinance charts
├── UI/
│   ├── app.py                 # Flask application
│   ├── templates/
│   │   └── index.html         # Main template
│   ├── static/
│   │   ├── js/
│   │   │   └── main.js        # Frontend logic
│   │   └── css/
│   │       └── style.css      # Styling
│   └── requirements.txt       # Python dependencies
└── keys.txt                   # API credentials
```

## 💡 Tips

1. **Performance**: News loading is asynchronous - the page stays responsive
2. **Filtering**: Use filters to focus on specific stocks or sources
3. **Real-time**: Alpaca news streams continuously when working (paid tier)
4. **Refresh**: Click the refresh button to get latest news
5. **Links**: All news items link to original sources

## 📚 Documentation

### Twitter API Docs
https://developer.twitter.com/en/docs/twitter-api

### Alpaca News API Docs
https://alpaca.markets/docs/api-references/market-data-api/news-data/

### Finnhub API Docs
https://finnhub.io/docs/api

## 🤝 Contributing

To add new news sources:
1. Create a new module in `Data/` folder
2. Implement fetching logic
3. Add endpoint in `UI/app.py`
4. Update frontend to display new source
5. Add to combined news endpoint

## 📄 License

This project is for educational and personal use only. API keys and data sources have their own terms of service.

---

**Last Updated:** December 29, 2025  
**Version:** 2.0.0  
**Author:** Stock Predictor Team
