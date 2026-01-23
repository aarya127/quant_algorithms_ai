# Stock Sentiment Analysis - Setup Guide

## Overview
You now have **two separate sentiment analysis tools**:

### 1. **`finbert.py`** - US Stocks (Finnhub)
- **API**: Finnhub (60 calls/minute)
- **Stocks**: US stocks (NYSE, NASDAQ, etc.)
- **Data Sources**: 
  - Real-time news articles
  - Earnings data
  - Company information
- **Stocks Analyzed**: AAPL, MSFT, TSLA, NVDA, GOOGL, AMZN, META

### 2. **`finbert_canadian.py`** - Canadian Stocks (Alpha Vantage)
- **API**: Alpha Vantage (25 requests/day - FREE tier)
- **Stocks**: TSX stocks (`.TO` suffix)
- **Data Sources**:
  - News sentiment (limited on free tier)
  - Company overview
  - Real-time quote data with price movement analysis
- **Stocks Available**: AC.TO, TD.TO, ENB.TO, RCI-B.TO, SHOP.TO, BMO.TO, RY.TO

---

## Why This Separation?

**The Problem**: Finnhub's free tier doesn't support Canadian/international stocks (403 Forbidden error)

**The Solution**: Use Alpha Vantage for Canadian stocks, Finnhub for US stocks

---

## Usage

### US Stocks
```bash
python Sentiment/finbert.py
```

### Canadian Stocks
```bash
python Sentiment/finbert_canadian.py
```

---

## Results Example

### US Stocks (Finnhub):
```
Symbol       Company              Sentiment    Pos      Neg      Sources 
----------------------------------------------------------------------
AAPL         Apple Inc.           POSITIVE     60.8%    33.5%    11      
MSFT         Microsoft Corp.      POSITIVE     46.1%    35.0%    11      
TSLA         Tesla Inc.           POSITIVE     44.5%    27.6%    11      
NVDA         NVIDIA Corp.         POSITIVE     51.4%    19.7%    11      
GOOGL        Alphabet Inc.        POSITIVE     46.0%    30.0%    11      
```

### Canadian Stocks (Alpha Vantage):
```
Symbol       Company              Sentiment    Pos      Neg      Sources 
----------------------------------------------------------------------
AC.TO        Air Canada           POSITIVE     88.1%    3.0%     1       
```

---

## Key Features

### Both Tools:
✅ FinBERT sentiment analysis (state-of-the-art financial NLP)  
✅ Multiple data sources aggregated  
✅ Confidence scoring  
✅ Distribution tracking (positive/negative/neutral)  
✅ Summary tables

### US Tool (Finnhub):
- 200+ news articles per stock
- Earnings surprises analysis
- High-frequency data (60 calls/min)

### Canadian Tool (Alpha Vantage):
- Real-time quote tracking
- Price movement sentiment
- Company fundamentals
- Limited by API tier (25 calls/day)

---

## API Keys Location

Both APIs read from: `/keys.txt`

```
#finhub - 60 API calls/minute
<your_finnhub_key>

#alphavantage - 25 requests per day
<your_alpha_vantage_key>
```

---

## Troubleshooting

### "403 Forbidden" on Canadian stocks with Finnhub
✅ **Fixed**: Use `finbert_canadian.py` instead

### "No news articles found" for Canadian stocks
⚠️ **Known limitation**: Alpha Vantage free tier has limited news coverage for Canadian stocks
✅ **Still works**: Quote data and price sentiment analysis are functional

### "API limit reached"
- Finnhub: Wait 1 minute (60 calls/min limit)
- Alpha Vantage: Wait 24 hours (25 calls/day limit)

---

## Dependencies

All installed in your virtual environment (`.venv`):
- `transformers` - FinBERT model
- `torch` - PyTorch for deep learning
- `finnhub-python` - Finnhub API client
- `pandas` - Data manipulation
- `yfinance` - Yahoo Finance (for `prices.py`)
- `requests` - HTTP client (for Alpha Vantage)

---

## File Structure

```
stock_predictor/
├── Sentiment/
│   ├── finbert.py              # US stocks (Finnhub)
│   ├── finbert_canadian.py     # Canadian stocks (Alpha Vantage)
│   └── FinBert/
│       └── fin.py              # Original test file
├── Data/
│   ├── finnhub.py              # Finnhub API wrapper
│   └── alphavantage.py         # Alpha Vantage API wrapper
├── .venv/                      # Virtual environment
├── keys.txt                    # API keys
└── requirements.txt            # Dependencies
```

---

## Next Steps

### To analyze all Canadian stocks:
Edit `finbert_canadian.py` line 248:
```python
# Change this:
test_stocks = {"AC.TO": "Air Canada"}

# To this:
for symbol, company_name in canadian_stocks.items():
```

**Warning**: This will use multiple API calls. With 7 Canadian stocks × ~3 API calls each = ~21 calls (close to your 25/day limit).

---

## Performance

- **FinBERT Model Loading**: ~5-10 seconds (one-time per run)
- **US Stock Analysis**: ~2-3 seconds per stock
- **Canadian Stock Analysis**: ~3-5 seconds per stock
- **Total US Run**: ~30-45 seconds for 7 stocks
- **Total Canadian Run**: ~25-40 seconds for 7 stocks
