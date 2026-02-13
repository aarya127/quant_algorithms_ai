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

### Multi-Language High-Performance System

This platform uses a **polyglot architecture** optimized for each component's requirements:

- **🐍 Python (70-75%)**: Strategy logic, ML models, data science, backtesting
- **⚡ C++ (15-20%)**: Ultra-low-latency execution engine, order book simulation, market making
- **🔷 Go (10-15%)**: Risk engine, control plane, data ingestion, monitoring

```
┌─────────────────────────────────────────────────────┐
│                  Web Interface                      │
│          (Bootstrap 5 + Chart.js + Vanilla JS)     │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│            Flask Backend (Python)                   │
│  ┌───────────────────────────────────────────────┐ │
│  │  Strategy Engine │ ML Models │ Backtesting   │ │
│  │  Research Tools  │ Analytics │ Visualization │ │
│  └───────────────────────────────────────────────┘ │
└──┬──────────────────┬──────────────────┬───────────┘
   │                  │                  │
   │ Python API       │ gRPC            │ pybind11
   │                  │                  │
┌──▼────────┐  ┌─────▼──────┐  ┌───────▼─────────┐
│ Market    │  │ Go Risk    │  │ C++ Execution   │
│ Data APIs │  │ Engine     │  │ Engine          │
│           │  │            │  │                 │
│ yfinance  │  │ • Real-time│  │ • Order routing │
│ Finnhub   │  │   risk     │  │ • Order book    │
│ Alpha     │  │ • Limits   │  │ • Market making │
│ Vantage   │  │ • Alerts   │  │ • Hedging       │
│           │  │ • Controls │  │ < 10μs latency  │
└───────────┘  └────────────┘  └─────────────────┘
       │
  ┌────┴────┐
  │ FinBERT │ 
  │   AI    │
  │Sentiment│
  └─────────┘
```

**Performance Targets:**
- Order execution: **< 10 microseconds**
- Order book updates: **< 1 microsecond**
- Risk checks: **< 100 microseconds**
- Strategy signals: **< 1 second** (Python is fine here)

**Why This Split?**
- **C++**: When every microsecond matters (execution loop)
- **Go**: System reliability, concurrency, graceful degradation
- **Python**: Rapid development, ML ecosystem, research iteration

---

## 🧩 Quant Algorithm Architecture

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

### **Algorithm Development Structure**

All algorithms in this repository follow this systematic approach:

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

### **Models & Algorithms In Development**

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

## 🛠️ Technologies

### Python Stack (70-75%)
- **Python 3.12**: Core language for strategies and research
- **Flask 3.0**: Web framework and API server
- **NumPy/Pandas**: Numerical computing and data manipulation
- **PyTorch/Transformers**: ML models and FinBERT sentiment
- **yfinance**: Market data
- **Finnhub/Alpha Vantage**: News and fundamentals

### C++ Stack (15-20%)
- **C++20**: Modern C++ with performance optimization
- **CMake**: Build system with `-O3 -march=native -flto`
- **pybind11**: Seamless Python bindings
- **Lock-free algorithms**: For ultra-low-latency operations
- **Cache-optimized data structures**: < 1μs order book updates
- **Google Test/Benchmark**: Testing and performance profiling

**C++ Components:**
- Execution engine (< 10μs order submission)
- Order book simulation (< 1μs updates)
- Market making algorithms
- Latency-sensitive delta hedging

### Go Stack (10-15%)
- **Go 1.21+**: Concurrency and reliability
- **gRPC + Protocol Buffers**: Inter-service communication
- **Goroutines/Channels**: Parallel risk calculations
- **Prometheus**: Metrics and monitoring
- **Zap**: Structured logging

**Go Services:**
- Risk engine (real-time limit checks)
- Control plane (strategy orchestration)
- Data ingestion (high-throughput pipelines)
- Monitoring and alerting

### Frontend
- **Bootstrap 5**: Responsive UI framework
- **Chart.js**: Interactive financial charts
- **Vanilla JavaScript**: Client-side logic
- **LocalStorage**: Persistent state

---

## 📦 Installation

### Prerequisites
- **Python 3.12+**
- **C++ Compiler**: GCC 11+ or Clang 14+ (for C++20 support)
- **Go 1.21+**: For risk engine services
- **CMake 3.20+**: Build system for C++
- **Protocol Buffers**: For gRPC communication

### Quick Start (All Components)

```bash
# Clone repository
git clone https://github.com/aarya127/quant_algorithms_ai.git
cd quant_algorithms_ai

# Build everything (C++, Go, Python)
make all

# Or build individually:
make cpp      # Build C++ components
make go       # Build Go services
make python   # Install Python dependencies
```

### Detailed Installation

#### 1. Install System Dependencies (macOS)
```bash
# C++ toolchain
xcode-select --install

# CMake
brew install cmake

# Go
brew install go

# Protocol Buffers
brew install protobuf
```

#### 2. Build C++ Components
```bash
cd cpp
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(sysctl -n hw.ncpu)
cd ../..

# Python bindings will be available as cpp_bindings.so
```

#### 3. Build Go Services
```bash
cd go
go mod download
go build -o bin/risk_engine ./risk_engine/main.go
cd ..
```

#### 4. Install Python Dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pybind11  # For C++ bindings
```

#### 5. Configure API Keys
Create `keys.txt` in root:
```
FINNHUB_API_KEY=your_finnhub_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
TWITTER_BEARER_TOKEN=your_twitter_bearer_token
NVIDIA_API_KEY=your_nvidia_api_key (optional)
```

#### 6. Download FinBERT Model
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

### Running the Full System

#### 1. Start Go Risk Engine
```bash
# Terminal 1
cd go
go run risk_engine/main.go --port 50051

# Or using Makefile
make run-risk
```

#### 2. Start Flask Backend
```bash
# Terminal 2
cd backend
source ../.venv/bin/activate
python app.py

# Or using Makefile
make run-backend
```

#### 3. Access Web Interface
Open browser to: **http://localhost:5000**

### Using C++ Components from Python

```python
# Import C++ bindings
from cpp_bindings import ExecutionEngine, OrderBook, OrderSide, OrderType

# Create execution engine
engine = ExecutionEngine()

# Submit ultra-fast order (< 10μs)
order_id = engine.submit_order(
    symbol="AAPL",
    side=OrderSide.BUY,
    type=OrderType.LIMIT,
    price=150.0,
    quantity=100
)

# Check order status
order = engine.get_order(order_id)
print(f"Status: {order.status}, Filled: {order.filled_quantity}")

# Get performance metrics
metrics = engine.get_metrics()
print(f"Avg latency: {metrics.avg_latency_us}μs")
```

### Using Go Risk Engine from Python

```python
from go.risk_engine.client import RiskEngineClient
from decimal import Decimal

# Connect to risk engine
risk = RiskEngineClient(host='localhost', port=50051)

# Check position before execution
result = risk.check_position(
    symbol='AAPL',
    quantity=Decimal('1000'),
    price=Decimal('150.50'),
    side='BUY'
)

if result['approved']:
    # Execute via C++ engine
    order_id = engine.submit_order(...)
else:
    print(f"Risk rejected: {result['message']}")

# Get risk metrics
metrics = risk.get_metrics()
print(f"Total exposure: ${metrics['total_exposure']}")
```

### Running Demo

```bash
# Full integration demo
python examples/multi_language_demo.py
```

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
├── performance/                   # High-Performance Components (25-35%)
│   │
│   ├── cpp_execution/            # C++ Ultra-Low-Latency Layer (15-20%)
│   │   ├── execution_engine/     # Order execution (< 10μs)
│   │   │   ├── execution_engine.hpp
│   │   │   ├── execution_engine.cpp
│   │   │   └── CMakeLists.txt
│   │   ├── order_book/           # Order book simulation (< 1μs)
│   │   │   ├── order_book.hpp
│   │   │   ├── order_book.cpp
│   │   │   └── CMakeLists.txt
│   │   ├── market_making/        # Market maker strategies
│   │   ├── hedging/              # Latency-sensitive hedging
│   │   ├── common/               # Shared utilities
│   │   │   └── utils.hpp         # Cache-aligned structures
│   │   ├── bindings/             # Python integration
│   │   │   ├── bindings.cpp      # pybind11 bindings
│   │   │   └── CMakeLists.txt
│   │   ├── tests/                # Google Test suite
│   │   ├── benchmarks/           # Performance benchmarks
│   │   ├── CMakeLists.txt        # Master build config
│   │   └── README.md             # C++ documentation
│   │
│   └── go_services/              # Go System Services (10-15%)
│       ├── risk_engine/          # Real-time risk management
│       │   ├── engine.go         # Risk calculations
│       │   ├── server.go         # gRPC server
│       │   ├── main.go           # Service entry point
│       │   └── client.py         # Python client
│       ├── control_plane/        # Strategy orchestration
│       ├── data_ingestion/       # Market data pipelines
│       ├── monitoring/           # Observability
│       ├── proto/                # Protocol Buffers
│       │   └── risk.proto        # gRPC definitions
│       ├── common/               # Shared Go utilities
│       ├── go.mod                # Go dependencies
│       └── README.md             # Go documentation
│
├── algorithms/                    # Trading Algorithms (Python 70-75%)
│   ├── machine_learning_algorithms/
│   │   ├── deep_learning/
│   │   ├── factor_discovery/
│   │   └── fine_tune/
│   ├── monte_carlo/
│   ├── position_sizing/
│   │   └── reinforcement_learning/
│   └── volatility_forecasting/
│
├── models/                        # Quantitative Finance Models
│   ├── asset_allocation_models/
│   │   └── portfolio_optimization/
│   ├── credit_risk_models/
│   ├── equity_options_pricing_models/
│   ├── execution_models/
│   │   ├── TWAP/                 # Time-Weighted Average Price
│   │   └── VWAP/                 # Volume-Weighted Average Price
│   ├── interest_rate_models/
│   │   ├── fmm/ (Forward Market Model)
│   │   ├── hjm/ (Heath-Jarrow-Morton)
│   │   ├── hull_white/
│   │   └── llm_bgm/ (Libor Market Model)
│   ├── time_series_models/
│   │   ├── arima.py
│   │   ├── garch.py
│   │   ├── cointegrations.py
│   │   ├── volatility_clustering.py
│   │   └── machine_learning/
│   │       └── vector_autoregression.py
│   └── volatility_models/
│
├── quant_research/                # Research Papers (9+ LaTeX Documents)
│   ├── stochastic_volatility/
│   │   ├── heston_model/theory.tex
│   │   ├── sabr_model/theory.tex
│   │   └── diagnostics.ipynb
│   ├── macd_rsi/
│   │   ├── macd_theory.tex
│   │   └── rsi_theory.tex
│   ├── greeks/theory.tex
│   ├── derivatives_volatility/theory.tex
│   ├── advanced_trading/
│   │   ├── theory.tex
│   │   ├── prototype.py
│   │   └── cointegration/
│   ├── market_microstructure/theory.tex
│   ├── state_space_models/theory.tex
│   ├── var_calculations/
│   │   ├── prototype.py
│   │   ├── var/
│   │   └── cvar/
│   └── frm_cqf/prototype.py
│
├── data/                          # Data Fetching & Processing
│   ├── alpaca_news.py
│   ├── alphavantage.py
│   ├── charts.py
│   ├── finnhub.py
│   ├── nvidia_llm.py
│   ├── prices.py
│   └── twitter_feed.py
│
├── sentiment/                     # AI Sentiment Analysis
│   ├── finbert.py                # FinBERT model
│   └── finbert_canadian.py       # Canadian market adaptation
│
├── backtesting/                   # Backtesting Framework
│   ├── backtest.py
│   └── stress_testing/
│       └── stress_test.py
│
├── backend/                       # Flask Web Application
│   ├── app.py                    # Main Flask app + research PDFs
│   ├── stock_analyzer.py
│   ├── economic_events.json
│   ├── templates/
│   │   └── index.html
│   ├── static/
│   │   ├── css/style.css
│   │   ├── js/main.js
│   │   └── research/
│
├── tests/                         # Testing Suite
│   ├── test_finnhub.py
│   ├── test_alphavantage.py
│   └── results/
│
├── scripts/                       # Build Scripts
│   ├── build_cpp.sh              # C++ compilation
│   ├── build_go.sh               # Go services
│   └── build_all.sh              # Master build
│
├── examples/                      # Integration Examples
│   └── multi_language_demo.py    # Python + C++ + Go demo
│
├── Makefile                       # Build automation
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

Copyright (c) 2026 Aarya Shah

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
