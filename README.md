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
- **Research-Driven Development**: Built on peer-reviewed quantitative finance research with comprehensive documentation

### 🧩 Quant Algorithm Architecture

This repository follows professional quantitative finance standards. Algorithms under development follow this comprehensive lifecycle:

```
Idea → Model → Signal → Portfolio & Sizing → Execution → Risk Controls
  ↓      ↓       ↓           ↓                    ↓            ↓
Define  Validate Convert   Sizing &        Order Placement  Hard Stops
Objective Empirically Output   Correlation   & Slippage      & Limits
         to Signal    Control
           ↓
Backtesting → Stress Testing → Performance Attribution → Monitoring & Guardrails
   ↓              ↓                    ↓                        ↓
Causal    Regime Shifts,        Where does PnL         Logging, Metrics,
Testing   Volatility Spikes     come from?              Alerts, Reconciliation
```

#### **The 10 Stages of Algorithm Development** (Memorize This!)

1. **Define the Trading Objective** - Asset class, holding period, market regime, success metrics
2. **Choose / Build the Model** - ARIMA, GARCH, factor models with empirical validation
3. **Convert Model → Signal** - Deterministic entry/exit logic with stable thresholds
4. **Position Sizing & Portfolio Logic** - Max position size, volatility targeting, leverage limits
5. **Execution Logic** - Market vs limit orders, TWAP/VWAP, slippage assumptions
6. **Risk Controls (Hard Stops)** - Max drawdown, daily loss limits, kill switches
7. **Backtesting (With Realism)** - No lookahead bias, include transaction costs, latency
8. **Stress & Failure Testing** - Regime shifts, volatility spikes, liquidity crashes, parameter instability
9. **Performance Attribution** - Understand where PnL comes from and when losses occur
10. **Monitoring & Guardrails** - Logging, metrics, alerts, reconciliation checks

#### **Models & Algorithms In Development**

This repository contains implementations and research across quantitative finance domains:

| Category | Focus | Status |
|----------|-------|--------|
| **Time Series Models** | ARIMA, GARCH, Cointegration, Vector Autoregression | 🔄 In Progress |
| **Volatility Models** | Heston, SABR, Local Volatility, Jump-Diffusion | 🔄 Building |
| **Factor Strategies** | Value, Momentum, Size, Quality, Mean Reversion | 🔄 Research |
| **Options Pricing** | Greeks, Implied Volatility, Smile Dynamics | 📚 Theory Complete |
| **Execution Algorithms** | TWAP, VWAP, Optimal Execution, Market Making | 🔄 Implementing |
| **Portfolio Optimization** | Mean-Variance, Risk Parity, Smart Beta | 🔄 Building |
| **Backtesting Framework** | Walk-forward testing, parameter stability | 📝 Planning |

Each algorithm under development includes:
- Comprehensive mathematical documentation
- Empirical validation on historical data  
- Risk measurement and stress testing
- Performance monitoring templates
- Real-world implementation considerations

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

### 📚 **Quantitative Research Hub**
Comprehensive research library with **9+ peer-reviewed papers** on quantitative finance:

#### **Stochastic Volatility Models**
- Heston Model: Mathematical foundations, calibration, and implementation
- SABR Model: CEV backbone and volatility smile modeling
- Derivatives Volatility: Complete guide to implied volatility, surfaces, and trading

#### **Technical Indicators**
- MACD (Moving Average Convergence Divergence): Momentum detection and trend following
- RSI (Relative Strength Index): Mean reversion and overbought/oversold signals

#### **Options & Risk Measures**
- The Greeks (Delta, Gamma, Theta, Vega, Rho): Comprehensive hedging and risk framework
- Alpha & Beta: Portfolio performance attribution and systematic risk

#### **Time Series & Econometric Models**
- State Space Models: Kalman filters and hidden variable estimation
- Market Microstructure: Price formation, spreads, and execution dynamics

#### **Advanced Trading Strategies**
- Modern portfolio theory and factor investing
- Statistical arbitrage and mean reversion
- Algorithmic execution and high-frequency trading
- Risk management frameworks (VaR, CVaR, stress testing)

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
├── Algorithms/                    # Trading algorithm implementations
│   ├── Machine Learning Algorithms/
│   ├── Monte Carlo Simulations/
│   ├── Position Sizing/
│   └── Volatility Forecasting/
│
├── Models/                        # Quantitative finance models
│   ├── Asset Allocation Models/
│   ├── Credit Risk Models/
│   ├── Equity Options Pricing/
│   ├── Execution Models/ (TWAP, VWAP)
│   ├── Interest Rate Models/ (FMM, HJM, Hull-White, LLM-BGM)
│   └── Time Series Models/
│       ├── ARIMA, GARCH, Volatility Clustering
│       └── Machine Learning (Vector Autoregression)
│
├── Quant Research/                # 9+ Research Papers
│   ├── Stochastic Volatility Models/
│   │   ├── heston_model/ → Heston Theory (PDF)
│   │   ├── sabr_model/ → SABR Theory (PDF)
│   │   └── diagnostics/ → Model validation
│   │
│   ├── Technical Indicators/
│   │   └── macd_rsi/ → MACD & RSI Theory (PDFs)
│   │
│   ├── Options & Greeks/
│   │   └── greeks/ → Delta, Gamma, Theta, Vega, Rho (PDF)
│   │
│   ├── Time Series & Econometrics/
│   │   ├── state_space_models/ → Kalman Filters (PDF)
│   │   └── market_microstructure/ → Price Formation, Execution (PDF)
│   │
│   ├── Volatility Derivatives/
│   │   └── derivatives_volatility/ → Volatility Surfaces, Trading (PDF)
│   │
│   ├── Advanced Trading/
│   │   ├── prototype.py → Algorithm prototypes
│   │   ├── cointegration/ → Pairs trading research
│   │   └── theory.tex → Portfolio Theory & Strategies (PDF)
│   │
│   ├── Risk Management/
│   │   ├── var_calculations/ → Value at Risk
│   │   └── cvar/ → Conditional VaR
│   │
│   └── Other Research/
│       ├── Derivatives Volatility/
│       ├── FRM/CQF Modules/
│       ├── Greeks & Risk Measures/
│       ├── MACD & RSI/
│       ├── Stochastic Volatility/
│       └── VaR Calculations/
│
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
├── Backend/                       # Web application
│   ├── app.py                    # Flask backend with research PDF serving
│   ├── stock_analyzer.py         # Stock analysis engine
│   ├── economic_events.json      # Calendar data (67 events)
│   ├── templates/
│   │   └── index.html            # Main HTML template
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css         # Custom styling
│   │   ├── js/
│   │   │   └── main.js           # Frontend JavaScript (2800+ lines)
│   │   └── research/
│   │       ├── PDFs/              # Compiled research papers
│   │       └── markdown/          # Legacy markdown docs
│
├── Tests/                         # API testing suite
│   ├── test_finnhub.py
│   ├── test_alphavantage.py
│   └── results/                  # Test output logs
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

### Quantitative Research & Algorithm Development

#### **Phase 1: Research Foundation** ✅ Complete
- [x] Market Microstructure theory and paper
- [x] Stochastic Volatility Models (Heston, SABR)
- [x] Technical Indicators (MACD, RSI)
- [x] Option Greeks and Risk Measures
- [x] State Space Models and Kalman Filters
- [x] Derivatives Volatility research
- [x] Advanced Trading Strategies framework

#### **Phase 2: Algorithm Implementation** 🔄 In Progress
- [ ] Time Series Models (ARIMA, GARCH, Cointegration)
- [ ] Factor-based Portfolio Strategies (Value, Momentum, Mean Reversion)
- [ ] Volatility Forecasting Models
- [ ] Risk-Adjusted Position Sizing
- [ ] Pairs Trading Framework
- [ ] Machine Learning Enhancements

#### **Phase 3: Backtesting & Validation** 📝 Planning
- [ ] Walk-forward backtesting framework
- [ ] Stress testing and parameter stability analysis
- [ ] Cross-validation for out-of-sample testing
- [ ] Performance attribution analysis
- [ ] Monte Carlo simulation

#### **Phase 4: Production Deployment** 🚀 Future
- [ ] Live paper trading
- [ ] Risk monitoring and guardrails
- [ ] Execution optimization
- [ ] Multi-strategy orchestration
- [ ] Real-time metrics and alerting

### UI & Platform Features

#### **Upcoming Features**
- [ ] Interactive research paper viewer in Quant section
- [ ] Algorithm parameter optimization dashboard
- [ ] Backtest results visualization
- [ ] Strategy performance comparison
- [ ] Factor analysis and attribution
- [ ] Portfolio management and tracking
- [ ] Options chain analysis with Greeks
- [ ] Email/SMS alerts for price targets
- [ ] Dark mode toggle
- [ ] Export reports to PDF
- [ ] Multi-user authentication
- [ ] WebSocket for real-time updates

#### **In Progress**
- [x] Economic calendar with 2026 events
- [x] Twitter influential account monitoring
- [x] Custom watchlist with localStorage
- [x] Multi-exchange stock search
- [x] FinBERT sentiment analysis
- [x] Quantitative Research Hub (9+ papers)
- [x] Dynamic LaTeX PDF compilation and serving
- [x] Comprehensive research paper library

---

**⭐ If you find this project useful, please consider giving it a star on GitHub!**
