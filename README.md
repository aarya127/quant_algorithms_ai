# 🤖 Quant Algorithms AI

AI-powered quantitative analysis platform combining real-time market data, sentiment analysis, and volatility modeling.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


---

## ✨ Key Features

### Stock Analysis
- Real-time quotes and historical charts (NYSE, NASDAQ, TSX)
- Technical indicators (SMA, RSI, Bollinger Bands, MACD)
- Company profiles and financial metrics
- Custom watchlist with persistent storage

### AI & Sentiment
- **FinBERT**: Financial sentiment analysis (Bullish/Bearish/Neutral)
- **NVIDIA LLM**: Company overviews and insights
- Multi-source news aggregation (Finnhub, Twitter/X, Alpha Vantage)
- Earnings predictions and insider trading sentiment

### Quantitative Models
- **Volatility Models**: SABR and Heston calibration with market validation
- **Parameter Diagnostics**: Autocorrelation, regime detection, stress analysis
- **Greeks Validation**: Delta, Gamma, Vega with put-call parity checks
- **Rolling Window Testing**: Out-of-sample forecast validation
- Time series models (ARIMA, GARCH, cointegration)
- Portfolio optimization and risk management

### Economic Calendar
- 67+ events for 2026 (FOMC, CPI, NFP, GDP, earnings)
- Filter by type and importance
- Market holidays and tax deadlines

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- API keys for Finnhub, Alpha Vantage (free tiers available)

### Installation

```bash
# Clone repository
git clone https://github.com/aarya127/quant_algorithms_ai.git
cd quant_algorithms_ai

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
echo "FINNHUB_API_KEY=your_key_here" > keys.txt
echo "ALPHA_VANTAGE_API_KEY=your_key_here" >> keys.txt

# Download FinBERT model (one-time setup)
python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
AutoTokenizer.from_pretrained('ProsusAI/finbert'); \
AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')"
```

### Run Application

```bash
cd backend
python app.py
```

Open browser to: **http://localhost:5000**

---

## 🔧 Configuration

### API Keys (Free Tiers)

1. **Finnhub** (60 calls/min)
   - Sign up: https://finnhub.io/register
   
2. **Alpha Vantage** (25 calls/day)
   - Get key: https://www.alphavantage.co/support/#api-key

3. **Twitter/X** (Optional, 10k/month)
   - Developer Portal: https://developer.twitter.com/

### Canadian Stocks
System auto-converts US tickers to TSX:
```python
'TD' → 'TD.TO'      # Toronto-Dominion Bank
'ENB' → 'ENB.TO'    # Enbridge
'RCI' → 'RCI-B.TO'  # Rogers Communications
```

---

## 📊 Volatility Calibration

Production-grade SABR/Heston calibration with market validation:

```bash
cd models/volatility_models/calibration

# Run calibration
python run_calibration.py --ticker SPY --model sabr

# Validate Greeks
cd ../market_validation
python greeks_validation.py

# Full validation suite
python run_all_validations.py --ticker SPY --model sabr
```

**Features:**
- Surface density gates (production desk standards)
- Parameter path diagnostics with regime detection
- Greeks validation with put-call parity checks
- Rolling window out-of-sample testing

**Recent Results (SPY SABR):**
```
Parameters: α=1.15, ν=0.64, ρ=0.56
Calibration: 24 IV points, 4.7s execution
Greeks MAE: Δ=0.041, Γ=0.00002
Put-call parity: PASS ✓
```

---

## 📁 Project Structure

```
quant_algorithms_ai/
├── backend/              # Flask app
│   ├── app.py
│   ├── templates/
│   └── static/
├── models/
│   └── volatility_models/
│       ├── calibration/  # SABR/Heston calibration
│       │   ├── run_calibration.py
│       │   ├── data_aquisition.py
│       │   ├── objective_function.py
│       │   └── constraints_handling.py
│       ├── market_validation/  # Production validation suite
│       │   ├── parameter_diagnostics.py
│       │   ├── greeks_validation.py
│       │   ├── rolling_validation.py
│       │   └── run_all_validations.py
│       └── sabr_pricer.py
├── data/                 # Market data APIs
├── sentiment/            # FinBERT sentiment
├── quant_research/       # Research papers (9+ LaTeX docs)
└── tests/
```

---

## 🧪 Testing

```bash
# API tests
cd tests
python run_all_tests.py

# Volatility calibration
cd models/volatility_models/calibration
python run_calibration.py --ticker SPY --model sabr

# Greeks validation
cd ../market_validation
python greeks_validation.py
```

---

## 🗺️ Roadmap

### In Progress
- [ ] Time series models (ARIMA, GARCH)
- [ ] Factor strategies (momentum, mean reversion)
- [ ] Backtesting framework with stress testing

### Completed ✅
- [x] SABR/Heston calibration pipeline
- [x] Market validation suite (Greeks, OOS testing)
- [x] Real-time sentiment analysis
- [x] Economic calendar (67+ events)
- [x] Multi-exchange stock search

---

## 📄 License

MIT License - see LICENSE file

---

## 📧 Contact

**Aarya Shah** - [@aarya127](https://github.com/aarya127)

**⭐ Star this repo if you find it useful!**