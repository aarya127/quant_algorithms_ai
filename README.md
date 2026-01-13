# 🤖 Quant Algorithms AI (Invest.ai)

**An AI-powered quantitative analysis and stock prediction platform** combining real-time market data, sentiment analysis, technical indicators, and financial news to provide comprehensive investment insights.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Technologies](#technologies)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Integrations](#api-integrations)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

**Quant Algorithms AI** is a sophisticated financial analytics platform that leverages artificial intelligence and quantitative methods to analyze stocks across NYSE, NASDAQ, and TSX exchanges. The platform integrates multiple data sources, applies machine learning-based sentiment analysis, and presents actionable insights through an intuitive web interface.

### What Makes This Different?

- **AI-Powered Sentiment Analysis**: Uses FinBERT (Financial BERT) for accurate market sentiment extraction from news
- **Multi-Exchange Support**: Seamlessly handles US (NYSE, NASDAQ) and Canadian (TSX) stocks with proper currency conversion
- **Real-Time Data**: Integrates yfinance, Finnhub, and Alpha Vantage for comprehensive market coverage
- **Quantitative Analysis**: Technical indicators (SMA, RSI, MACD) with visual charting
- **Economic Calendar**: Tracks FOMC meetings, CPI reports, earnings dates, and market holidays
- **Influential Monitoring**: Tracks tweets from market-moving figures (Elon Musk, Warren Buffett, Trump, etc.)

---

## ✨ Key Features

### 📊 **Stock Analysis Dashboard**
- Real-time price quotes and historical charts
- Company profiles with financial metrics
- Technical indicators (Moving Averages, RSI, Bollinger Bands)
- Volume analysis and price momentum
- Support for watchlist customization

### 🧠 **AI-Powered Insights**
- **FinBERT Sentiment Analysis**: Analyzes news sentiment (Bullish/Bearish/Neutral)
- **NVIDIA LLM Integration**: Generates comprehensive company overviews
- Earnings predictions and surprise analysis
- Insider trading sentiment tracking

### 📰 **Multi-Source News Aggregation**
- **Finnhub**: Company-specific news and market updates
- **Twitter/X Integration**: Monitors influential investors and financial media
  - Market movers: Elon Musk, Warren Buffett, Carl Icahn, Bill Ackman
  - News outlets: WSJ, Bloomberg, Reuters, CNBC, Financial Times
- **Alpha Vantage**: Market sentiment data and analysis
- Real-time filtering by source, symbol, and importance

### 📅 **Economic Events Calendar**
- 67+ scheduled events for 2026 including:
  - 8 FOMC interest rate decisions
  - 12 monthly CPI reports
  - 12 Non-Farm Payrolls reports
  - 4 quarterly GDP estimates
  - Market holidays and tax deadlines
- Filter by event type and importance level
- Month-by-month organization with visual indicators

### 🔍 **Advanced Stock Search**
- Fast autocomplete search across NYSE, NASDAQ, TSX, NEO, TSX Venture
- View detailed company information instantly
- Add/remove stocks from custom watchlist
- Persistent storage with localStorage

### 📈 **Technical Analysis**
- Multiple timeframe charts (1D, 5D, 1M, 3M, 6M, YTD, 1Y, 5Y)
- Candlestick patterns and volume analysis
- Moving averages (SMA 50, SMA 200)
- Relative Strength Index (RSI)
- Support and resistance levels

### 🏦 **Financial Metrics**
- Income statements, balance sheets, cash flow
- P/E ratios, market cap, dividend yields
- Earnings history and calendar
- Insider transactions and sentiment
- Basic financials from Finnhub

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Web Interface                      │
│          (Bootstrap 5 + Chart.js + Vanilla JS)     │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│              Flask Backend (Python)                 │
│  ┌───────────────────────────────────────────────┐ │
│  │  API Routes: /api/search, /api/stock,        │ │
│  │  /api/news, /api/calendar, /api/dashboard    │ │
│  └───────────────────────────────────────────────┘ │
└─────────────────┬───────────────────────────────────┘
                  │
     ┌────────────┼────────────┐
     │            │            │
┌────▼────┐  ┌───▼────┐  ┌───▼─────┐
│ yfinance│  │Finnhub │  │  Alpha  │
│  (Free) │  │(60/min)│  │ Vantage │
│Unlimited│  │  API   │  │ (25/day)│
└─────────┘  └────────┘  └─────────┘
                  │
         ┌────────┴─────────┐
         │                  │
    ┌────▼─────┐      ┌────▼─────┐
    │ FinBERT  │      │ Twitter  │
    │   AI     │      │   API    │
    │Sentiment │      │(Targeted)│
    └──────────┘      └──────────┘
```

---

## 🛠️ Technologies

### Backend
- **Python 3.12**: Core language
- **Flask 3.0**: Web framework
- **yfinance**: Free, unlimited stock data
- **Finnhub API**: Company news and financials (60 calls/min)
- **Alpha Vantage API**: Sentiment analysis (25 requests/day)
- **Tweepy**: Twitter/X API integration
- **Transformers + PyTorch**: FinBERT sentiment model
- **Pandas**: Data manipulation

### Frontend
- **Bootstrap 5**: Responsive UI framework
- **Chart.js**: Interactive financial charts
- **Font Awesome**: Icons
- **Vanilla JavaScript**: Client-side logic
- **LocalStorage**: Persistent watchlist

### AI/ML
- **FinBERT**: Financial sentiment analysis (Hugging Face)
- **NVIDIA NIM**: LLM-powered company insights

---

## 📦 Installation

### Prerequisites
- Python 3.12+
- pip (Python package manager)
- Virtual environment (recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/aarya127/quant_algorithms_ai.git
cd quant_algorithms_ai
```

### Step 2: Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure API Keys
Create `keys.txt` or `api_key.txt` in the root directory:
```
FINNHUB_API_KEY=your_finnhub_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
TWITTER_BEARER_TOKEN=your_twitter_bearer_token
NVIDIA_API_KEY=your_nvidia_api_key (optional)
```

### Step 5: Download FinBERT Model
```bash
python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; AutoTokenizer.from_pretrained('ProsusAI/finbert'); AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')"
```

---

## ⚙️ Configuration

### API Rate Limits
| Service | Free Tier | Rate Limit | Usage |
|---------|-----------|------------|-------|
| yfinance | ✅ Unlimited | None | Stock prices, company info, charts |
| Finnhub | ✅ Free | 60/min | News, earnings, insider data |
| Alpha Vantage | ✅ Free | 25/day | Sentiment analysis (on-demand) |
| Twitter/X | ⚠️ Limited | 10,000/month | Influential accounts only |

### Canadian Stock Mapping
The system automatically converts US tickers to TSX equivalents for CAD pricing:
```python
CANADIAN_STOCKS_MAP = {
    'TD': 'TD.TO',       # Toronto-Dominion Bank
    'ACDVF': 'AC.TO',    # Air Canada
    'ENB': 'ENB.TO',     # Enbridge
    'RCI': 'RCI-B.TO',   # Rogers Communications
    'CVE': 'CVE.TO',     # Cenovus Energy
}
```

### Default Watchlist
Edit `UI/app.py` to customize:
```python
DEFAULT_STOCKS = ["NVDA", "TD", "ACDVF", "MSFT", "ENB", "RCI", "CVE", "HUBS", "MU", "CNSWF", "AMD"]
```

---

## 🚀 Usage

### Starting the Application
```bash
cd UI
source ../.venv/bin/activate
python app.py
```

Access the application at: **http://localhost:5000**

### Key Workflows

#### 1. **Search for a Stock**
- Type symbol in search bar (e.g., "AAPL", "TSLA", "SHOP.TO")
- Select from dropdown results
- View comprehensive analysis

#### 2. **Add to Watchlist**
- Search for stock
- Click "+ Add to Watchlist" button
- Stock appears in sidebar
- Persists across sessions

#### 3. **View Economic Calendar**
- Navigate to Calendar tab
- Filter by event type (FOMC, CPI, Earnings, etc.)
- Filter by importance (High/Medium/Low)
- See upcoming market-moving events

#### 4. **Monitor Market News**
- Click News tab
- Filter by source (Twitter, Finnhub, Alpaca)
- Search by symbol
- Adjust news count (15/30/50/100 items)

#### 5. **Analyze Sentiment**
- Select stock from watchlist
- View AI Overview tab
- See FinBERT sentiment scores
- Read NVIDIA LLM-generated insights

---

## 🔌 API Integrations

### Getting API Keys

1. **Finnhub** (Free - 60 calls/min)
   - Sign up: https://finnhub.io/register
   - Dashboard → API Keys

2. **Alpha Vantage** (Free - 25 calls/day)
   - Sign up: https://www.alphavantage.co/support/#api-key
   - Free key delivered via email

3. **Twitter/X API** (Limited Free Tier)
   - Developer Portal: https://developer.twitter.com/
   - Create project → Generate Bearer Token
   - Note: Optimized to minimize API calls

4. **NVIDIA NIM** (Optional)
   - Sign up: https://build.nvidia.com/
   - Use for LLM-powered insights

### API Endpoint Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/search/<query>` | GET | Search stocks by symbol |
| `/api/stock/<symbol>` | GET | Get stock details |
| `/api/dashboard` | GET | Dashboard overview data |
| `/api/news/combined` | GET | Aggregated news from all sources |
| `/api/calendar` | GET | Economic events calendar |
| `/api/ai-overview/<symbol>` | GET | AI-generated insights |

---

## 📁 Project Structure

```
quant_algorithms_ai/
├── Data/                          # Data fetching modules
│   ├── alpaca_news.py            # Alpaca news stream
│   ├── alphavantage.py           # Alpha Vantage API
│   ├── charts.py                 # Chart data generation
│   ├── finnhub.py                # Finnhub API integration
│   ├── nvidia_llm.py             # NVIDIA LLM integration
│   ├── prices.py                 # Price data utilities
│   └── twitter_feed.py           # Twitter/X API (optimized)
│
├── Sentiment/                     # Sentiment analysis
│   ├── finbert.py                # FinBERT model integration
│   └── finbert_canadian.py       # Canadian market adaptation
│
├── UI/                            # Web application
│   ├── app.py                    # Flask backend
│   ├── stock_analyzer.py         # Stock analysis engine
│   ├── economic_events.json      # Calendar data (67 events)
│   ├── templates/
│   │   └── index.html            # Main HTML template
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css         # Custom styling
│   │   └── js/
│   │       └── main.js           # Frontend JavaScript (1930 lines)
│
├── Tests/                         # API testing suite
│   ├── test_finnhub.py
│   ├── test_alphavantage.py
│   └── results/                  # Test output logs
│
├── Time Series Forecast/          # ML forecasting (WIP)
│   └── time_series.ipynb         # Jupyter notebook
│
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── keys.txt                       # API keys (gitignored)
```

---

## 🧪 Testing

Run comprehensive API tests:
```bash
cd Tests
python run_all_tests.py
```

Individual tests:
```bash
python test_finnhub.py NVDA
python test_alphavantage.py NVDA
```

Results saved to `Tests/results/` with timestamps.

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Guidelines
- Follow PEP 8 style guide
- Add docstrings to functions
- Test API integrations before committing
- Update README for major changes

---

## 📊 Performance Considerations

### Optimization Strategies
- **Twitter API**: Only queries 17 influential accounts (90% reduction in calls)
- **Caching**: Results cached for 5 minutes to reduce redundant API calls
- **Debouncing**: Search input debounced to 300ms
- **Lazy Loading**: Charts and news load on-demand
- **LocalStorage**: Watchlist persists client-side

### Recommended API Usage
```
Daily API Calls (Typical Usage):
├── yfinance: Unlimited ✅
├── Finnhub: ~200/day (well under 60/min limit)
├── Alpha Vantage: 5-10/day (on-demand sentiment)
└── Twitter: 50-100/day (targeted accounts)
```

---

## 🔒 Security & Privacy

- **API keys stored locally** in gitignored files
- **No user authentication** (single-user application)
- **Client-side watchlist** (no server storage)
- **Rate limiting** respects API provider terms

**⚠️ Important**: Never commit `keys.txt` or `api_key.txt` to version control.

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: Twitter API not returning data
- **Solution**: Check bearer token validity and rate limits. Application shows warning banner when API fails.

**Issue**: Finnhub 429 error (rate limit)
- **Solution**: Wait 1 minute. Reduce refresh frequency. Free tier: 60 calls/min.

**Issue**: FinBERT model not loading
- **Solution**: Run model download command. Requires ~2GB disk space.

**Issue**: Canadian stocks show USD prices
- **Solution**: Verify ticker mapping in `CANADIAN_STOCKS_MAP`. Use `.TO` suffix.

**Issue**: Charts not displaying
- **Solution**: Check browser console. Ensure Chart.js CDN is accessible.

---

## 📝 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2026 Aarya Sinha

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

- **yfinance**: Ran Aroussi for the excellent Yahoo Finance API wrapper
- **FinBERT**: ProsusAI for the financial sentiment model
- **Finnhub**: For comprehensive financial data API
- **Alpha Vantage**: For market intelligence and sentiment data
- **Bootstrap Team**: For the responsive UI framework
- **Chart.js**: For beautiful, interactive charts

---

## 📧 Contact

**Aarya Shah**
- GitHub: [@aarya127](https://github.com/aarya127)
- Repository: [quant_algorithms_ai](https://github.com/aarya127/quant_algorithms_ai)

---

## 🗺️ Roadmap

### Upcoming Features
- [ ] Machine learning price prediction models
- [ ] Portfolio management and tracking
- [ ] Options chain analysis
- [ ] Backtesting framework for strategies
- [ ] Email/SMS alerts for price targets
- [ ] Mobile-responsive improvements
- [ ] Dark mode toggle
- [ ] Export reports to PDF
- [ ] Multi-user authentication
- [ ] WebSocket for real-time updates

### In Progress
- [x] Economic calendar with 2026 events
- [x] Twitter influential account monitoring
- [x] Custom watchlist with localStorage
- [x] Multi-exchange stock search
- [x] FinBERT sentiment analysis

---

**⭐ If you find this project useful, please consider giving it a star on GitHub!**
