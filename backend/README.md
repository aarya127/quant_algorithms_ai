# Stock Predictor Web UI

A comprehensive web application for AI-powered stock analysis and predictions.

## Features

### 📊 Daily Read - Trending News
- Real-time trending news from top stocks
- Sentiment-tagged articles (Positive/Neutral/Negative)
- Beautiful card-based layout with images

### 📈 Stock Details
- Company overview and basic information
- Real-time price quotes with change indicators
- Recent news feed

### 🧠 AI Sentiment Analysis
- FinBERT-powered sentiment analysis
- Analysis of 200+ news articles per stock
- Visual breakdown with charts
- Confidence scores and detailed summaries

### 🎯 Bull/Base/Bear Scenarios
- Three probability-weighted scenarios for each timeframe
- Price targets based on financial metrics
- Key factors and rationale for each scenario
- Multiple timeframes: 1W, 1M, 3M, 6M, 1Y

### ⭐ Metrics & Grading System
- A-F grading across 4 categories:
  - Valuation (P/E ratio)
  - Profitability (ROE, profit margin)
  - Growth (revenue, EPS)
  - Financial Health (debt/equity)
- Overall grade and score
- Visual progress bars

### 💡 Time-Based Recommendations
- Investment suggestions for 5 timeframes
- Buy/Hold/Sell recommendations
- Confidence levels
- Detailed reasoning

### 📅 Earnings Calendar
- Upcoming earnings events
- EPS estimates
- Company details

### 🔍 Search
- Quick stock search functionality
- Instant results

### 🔔 Notifications (Coming Soon)
- Real-time price alerts
- Sentiment changes
- Earnings reminders

## Installation

1. **Install Python dependencies:**
```bash
cd UI
pip install -r requirements.txt
```

2. **Ensure API keys are configured:**
Make sure you have `keys.txt` in the parent directory with:
```
#finnhub
your_finnhub_api_key

#alphavantage
your_alphavantage_api_key
```

3. **Install FinBERT model:**
The app will automatically download the FinBERT model on first use.

## Running the Application

1. **Start the Flask server:**
```bash
python app.py
```

2. **Open your browser:**
Navigate to `http://localhost:5000`

3. **Start exploring:**
- View trending news on the dashboard
- Click on any stock from the watchlist
- Explore different tabs for detailed analysis
- Try the search feature to find other stocks

## Architecture

### Backend (Flask)
- **app.py**: Main Flask application with API routes
- Integrates Finnhub, Alpha Vantage, and FinBERT
- RESTful API endpoints for all features

### Frontend
- **templates/index.html**: Main HTML template
- **static/css/style.css**: Custom styling
- **static/js/main.js**: JavaScript for interactivity

### API Endpoints

- `GET /` - Main dashboard page
- `GET /api/dashboard` - Trending news data
- `GET /api/stock/<symbol>` - Stock details
- `GET /api/sentiment/<symbol>` - Sentiment analysis
- `GET /api/scenarios/<symbol>` - Bull/Base/Bear scenarios
- `GET /api/metrics/<symbol>` - Metrics and grading
- `GET /api/recommendations/<symbol>` - Time-based suggestions
- `GET /api/calendar` - Earnings calendar
- `GET /api/search` - Stock search

## Technologies Used

### Backend
- **Flask**: Web framework
- **FinBERT**: AI sentiment analysis
- **Finnhub API**: US stock data
- **Alpha Vantage API**: Additional market data

### Frontend
- **Bootstrap 5**: Responsive UI framework
- **Font Awesome**: Icons
- **Chart.js**: Data visualization
- **Vanilla JavaScript**: Interactivity

## Features in Detail

### Sentiment Analysis
Uses the ProsusAI/finbert model to analyze financial news articles. The analysis:
- Processes up to 20 recent news articles
- Classifies each as positive, negative, or neutral
- Provides overall sentiment with confidence scores
- Includes visual charts showing sentiment distribution

### Scenario Analysis
Generates three scenarios based on:
- Current financial metrics (P/E, EPS growth, revenue growth)
- Market sentiment
- Historical volatility
- Time horizon

Probabilities are weighted: Bull (25%), Base (50%), Bear (25%)

### Grading System
Each stock receives grades in 4 categories:
- **Valuation**: Based on P/E ratio vs industry average
- **Profitability**: ROE and profit margins
- **Growth**: Revenue and EPS growth rates
- **Financial Health**: Debt-to-equity ratio

Overall grade is calculated from average scores.

### Time-Based Recommendations
Provides actionable recommendations for different investment horizons:
- **1 Week**: Short-term trading
- **1 Month**: Swing trading
- **3 Months**: Medium-term investment
- **6 Months**: Position trading
- **1 Year**: Long-term investment

## API Rate Limits

- **Finnhub**: 60 calls/minute (free tier)
- **Alpha Vantage**: 25 calls/day (free tier)

The app implements caching and rate limiting to stay within these limits.

## Customization

### Adding Stocks to Watchlist
Edit `DEFAULT_STOCKS` in `app.py`:
```python
DEFAULT_STOCKS = ["NVDA", "AAPL", "MSFT", "YOUR_STOCK"]
```

### Changing Timeframes
Modify `TIMEFRAMES` in `app.py` to add or remove timeframes.

### Styling
Edit `static/css/style.css` to customize colors, fonts, and layout.

## Troubleshooting

### FinBERT Model Not Loading
- Ensure you have internet connection for first download
- Check transformers and torch are installed correctly

### API Errors
- Verify API keys are correct in `keys.txt`
- Check you haven't exceeded rate limits
- Ensure parent directories (Data/, Sentiment/) are accessible

### No Data Showing
- Check browser console for JavaScript errors
- Verify Flask server is running
- Ensure all dependencies are installed

## Future Enhancements

- [ ] Real-time WebSocket notifications
- [ ] LLM-powered detailed analysis page
- [ ] Portfolio tracking
- [ ] Comparison tool for multiple stocks
- [ ] Historical data visualization
- [ ] Export reports to PDF
- [ ] User authentication and saved preferences
- [ ] Mobile app version

## License

This project is for educational purposes. Please ensure you comply with API terms of service.

## Support

For issues or questions, please check:
1. API documentation (Finnhub, Alpha Vantage)
2. FinBERT model documentation
3. Flask documentation
