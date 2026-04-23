# 🤖 Quant Algorithms AI

An end-to-end quantitative research and trading platform. The system combines a real-time web dashboard, a multi-source market data pipeline, FinBERT AI sentiment analysis, NVIDIA LLM overviews, volatility model calibration (SABR/Heston), an ML feature-matrix builder, a high-performance C++ execution engine, and a Go-based risk service — all deployed on Railway via Docker.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Deployment](#2-deployment)
3. [Frontend UI](#3-frontend-ui)
4. [API Layer (Flask Backend)](#4-api-layer-flask-backend)
5. [Data Pipeline](#5-data-pipeline)
6. [AI Summaries — NVIDIA LLM](#6-ai-summaries--nvidia-llm)
7. [Sentiment Analysis — FinBERT](#7-sentiment-analysis--finbert)
8. [ML Feature Matrix](#8-ml-feature-matrix)
9. [Quant Algorithms](#9-quant-algorithms)
10. [Volatility Models & Trading Signals](#10-volatility-models--trading-signals)
11. [Interest Rate & Pricing Models](#11-interest-rate--pricing-models)
12. [Backtesting & Portfolio Engine](#12-backtesting--portfolio-engine)
13. [High-Performance Components](#13-high-performance-components)
14. [Economic Calendar](#14-economic-calendar)
15. [Security & Key Management](#15-security--key-management)
16. [Quick Start](#16-quick-start)
17. [Project Structure](#17-project-structure)
18. [Roadmap](#18-roadmap)

---

## 1. System Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│  Browser  (Bootstrap 5 + Chart.js)                                   │
│  Dashboard · News · Calendar · Quant · Trading                       │
└───────────────────────────┬──────────────────────────────────────────┘
                            │ HTTP/JSON
┌───────────────────────────▼──────────────────────────────────────────┐
│  Flask Backend  (29 routes, TTL cache, Gunicorn)                     │
│  backend/app.py                                                      │
└──┬──────────────┬──────────────┬──────────────┬──────────────────────┘
   │              │              │              │
   ▼              ▼              ▼              ▼
Data Layer    AI / NLP       Quant Models   Perf Services
(7 APIs)      (FinBERT,      (SABR, Heston, (C++ engine,
              NVIDIA LLM)    time series)   Go risk svc)
```

**Stack:**
| Layer | Technology |
|---|---|
| Web framework | Flask 3.0 + Gunicorn |
| Frontend | Bootstrap 5, Chart.js, vanilla JS |
| Primary data | yfinance, Finnhub, Polygon, Alpaca SDK, Fiscal.ai |
| AI / NLP | FinBERT (ProsusAI), NVIDIA Mistral Large 3 |
| Quant | QuantLib, scipy, numpy, pandas |
| High-perf | C++ (pybind11), Go (gRPC) |
| Deployment | Docker, Railway |
| Python | 3.12, `.venv` |

---

## 2. Deployment

**Docker** (`Dockerfile`):
- Base: `python:3.11-slim`
- CPU-only PyTorch (~180 MB vs 2.5 GB CUDA build)
- `entrypoint.sh` runs a pre-flight import test then launches Gunicorn on port 8080

**Railway** (`railway.toml`):
```toml
[build]
builder = "DOCKERFILE"

[deploy]
startCommand     = "./entrypoint.sh"
healthcheckPath  = "/health"     # GET /health → 200 OK
healthcheckTimeout = 300
restartPolicyType  = "ON_FAILURE"
```

Railway auto-builds from the Dockerfile on every push, health-checks via `/health`, and restarts on failure. All secrets are injected as Railway environment variables — nothing is committed to the repo.

---

## 3. Frontend UI

Single-page application served from `backend/templates/index.html`.

**Main navigation tabs:**

| Tab | Description |
|---|---|
| **Dashboard** | Live market indices (SPX, RUT, INDU, VIX) + 30-day history + stock watchlist |
| **News** | Aggregated feed from Finnhub, Alpaca, Twitter/X, and a merged combined view |
| **Calendar** | 40+ economic events for 2026 (FOMC, CPI, NFP, GDP, holidays), filterable by type/importance |
| **Quant** | Quantitative analysis tools and research paper viewer |
| **Trading** | Trading chart viewer with custom TradingView-style charts via chart-img.com |

**Stock detail tabs** (appear when a ticker is selected):

| Tab | Content |
|---|---|
| **Overview** | Company profile, real-time price, recent news |
| **Statistics** | P/E, P/B, margins, debt ratios, growth rates, 52-week range |
| **Charts** | Candlestick + Volume + RSI + MACD — timeframes 1D → MAX |
| **Sentiment** | FinBERT scores (Bullish/Bearish/Neutral) across last 7 days of news |
| **Scenarios** | Bull / Base / Bear price targets at 1W, 1M, 3M, 6M, 1Y horizons |
| **Metrics & Grades** | Composite evaluation scores and letter grades |
| **Recommendations** | LLM-assisted and rule-based investment suggestions |

---

## 4. API Layer (Flask Backend)

`backend/app.py` — 29 REST endpoints with a thread-safe TTL cache.

**Cache TTLs:** quotes 30 s · profile 1 h · financials/news 10 min

| Route | Method | Purpose |
|---|---|---|
| `/health` | GET | Deployment health check |
| `/` | GET | SPA entry point |
| `/api/search/<query>` | GET | Symbol/name search |
| `/api/indices` | GET | Live index levels (SPX, RUT, INDU, VIX) |
| `/api/indices/history` | GET | 30-day index performance |
| `/api/dashboard` | GET | Aggregated overview for a stock |
| `/api/stock/<symbol>` | GET | Price + profile + news bundle |
| `/api/ai-overview/<symbol>` | GET | NVIDIA Mistral Large 3 company overview |
| `/api/backtest` | POST | Run a backtest |
| `/api/backtest/strategies` | GET | List available backtest strategies |
| `/api/statistics/<symbol>` | GET | Ratios, margins, growth metrics |
| `/api/sentiment/<symbol>` | GET | FinBERT sentiment scores |
| `/api/scenarios/<symbol>` | GET | Bull/Base/Bear scenario analysis |
| `/api/metrics/<symbol>` | GET | Evaluation grades |
| `/api/recommendations/<symbol>` | GET | Investment recommendations |
| `/api/calendar` | GET | Economic events calendar |
| `/api/charts/<symbol>` | GET | OHLCV chart data + indicators |
| `/api/charts/<symbol>/all-timeframes` | GET | 1D / 5D / 1M / 3M / 6M / 1Y / 5Y / MAX |
| `/api/charts/compare` | POST | Multi-ticker chart comparison |
| `/api/charts/<symbol>/indicators` | GET | SMA, EMA, RSI, MACD, ATR, OBV |
| `/api/news/twitter` | GET | Twitter/X feed (market influencers) |
| `/api/news/alpaca` | GET | Alpaca real-time news stream |
| `/api/news/combined` | GET | Merged multi-source news |
| `/api/research/<paper>` | GET | Research paper content (LaTeX/markdown) |
| `/api/research/diagnostics/notebook` | GET | Jupyter diagnostic notebook |
| `/api/algorithm/<name>` | GET | Algorithm source + docs |
| `/api/trading/chart` | POST | Custom TradingView-style chart render |

---

## 5. Data Pipeline

`algorithms/machine_learning_algorithms/data_pipelines/`

### DataExtractor (`extractor.py`)

Single-file extraction layer. Every method returns a pandas DataFrame. Provider fallback chains ensure data is always returned even if a source is down.

```python
from algorithms.machine_learning_algorithms.data_pipelines import DataExtractor

ex = DataExtractor()
ex.ohlcv("NVDA", period="1y")          # adjusted daily bars
ex.profile("NVDA")                      # name, sector, market cap, …
ex.fundamentals("NVDA")                 # 30+ ratios from 3 sources
ex.news("NVDA", start="2026-01-01")     # headlines + source + URL
ex.sentiment("NVDA")                    # insider MSPR monthly
ex.technicals("NVDA")                   # OHLCV + SMA/RSI/MACD/BB/ATR/OBV
ex.options("NVDA")                      # options chain + IV
ex.earnings_calendar("NVDA")            # EPS dates + estimates
ex.calendar()                           # NYSE trading days
ex.test_apis("AAPL")                    # live connectivity check
```

**Provider fallback chains:**

| Data type | Primary | Fallback 1 | Fallback 2 |
|---|---|---|---|
| OHLCV | yfinance | Polygon SIP tape | Alpaca SDK |
| Profile | yfinance | Finnhub | — |
| Fundamentals | yfinance | Finnhub | Fiscal.ai |
| News | Finnhub | yfinance | Alpaca SDK |
| Sentiment | Finnhub | — | — |
| Options | yfinance | Polygon (greeks, Starter+) | — |
| Calendar | Alpaca SDK | pandas_market_calendars | bdate_range |

**Provider limits:**

| Provider | Rate limit | Notes |
|---|---|---|
| yfinance | None | Primary everywhere |
| Finnhub | 60 req/min | News, sentiment, profile, fundamentals, earnings |
| Polygon | 5 req/min | SIP-tape OHLCV (free); options greeks (Starter+) |
| Alpaca | 200 req/min | `alpaca-py` SDK — OHLCV, news, market calendar |
| Fiscal.ai | 250 calls/day | Structured fundamentals — 26 free-tier tickers |
| AlphaVantage | 25 calls/day | Reserved for macro data only (excluded from pipeline) |

### DataTransformer (`transform.py`)

Merges all sources into a single daily-indexed ML feature matrix.

```python
from algorithms.machine_learning_algorithms.data_pipelines import DataTransformer

dt  = DataTransformer()
df  = dt.build_feature_matrix("NVDA", period="2y")   # ~500 rows × 84 cols
dt.describe_columns(df)                               # column catalogue + null %
```

**84-column feature groups:**

| Group | Columns | Merge strategy |
|---|---|---|
| OHLCV | Open, High, Low, Close, Volume | Daily spine |
| Technicals | SMA20/50/200, EMA20, RSI14, MACD + signal + hist, BB upper/mid/lower, ATR14, OBV, daily_return, log_return | Computed from spine |
| Derived ratios | `price_to_sma20/50/200`, `bb_pct`, `volume_zscore`, `overnight_gap`, `high_low_range` | Computed in-place |
| Fundamentals | 30+ `fund_*` (P/E, ROE, margins, EV/EBITDA, ROIC, 52w high/low, …) | Broadcast as constants |
| News | `news_count` | Aggregated to daily count, 0-filled |
| Insider sentiment | `insdr_change`, `insdr_mspr` | Monthly → daily forward-fill |
| Options surface | `opt_atm_iv`, `opt_put_call_oi`, `opt_total_oi`, `opt_max_pain` | Snapshot broadcast |
| Earnings flags | `earn_days_to_next`, `earn_days_since_last`, `earn_is_week` | Per-row calendar proximity |

---

## 6. AI Summaries — NVIDIA LLM

`data/nvidia_llm.py` + `/api/ai-overview/<symbol>`

Calls **NVIDIA's hosted Mistral Large 3** API to generate a natural-language overview of any company. The stock detail **Overview** tab renders this summary alongside the real-time price and news feed. API key is read from `NVIDIA_API_KEY` environment variable.

---

## 7. Sentiment Analysis — FinBERT

`sentiment/finbert.py` + `/api/sentiment/<symbol>`

Uses **ProsusAI/finbert** — a BERT model fine-tuned on financial text — to score news articles as Bullish, Bearish, or Neutral.

**Flow:**
1. Fetch last N days of news headlines + summaries via Finnhub
2. Score each article: FinBERT → `(label, {positive, negative, neutral})`
3. Aggregate scores across all articles
4. Return dominant sentiment + breakdown to the Sentiment tab

**Two variants:**
- `finbert.py` — US stocks (Finnhub data source)
- `finbert_canadian.py` — Canadian stocks (Alpha Vantage data source)

Model loads lazily on first request to avoid blocking Gunicorn workers at startup.

---

## 8. ML Feature Matrix

See [§5 DataTransformer](#datatransformer-transformpy) for the full column breakdown.

**Usage in a model training loop:**
```python
from algorithms.machine_learning_algorithms.data_pipelines import DataTransformer
import pandas as pd

dt = DataTransformer()

# Build training set across multiple tickers
frames = [dt.build_feature_matrix(sym, period="5y") for sym in ["NVDA","MSFT","AMZN"]]
X = pd.concat(frames).dropna(subset=["RSI_14", "fund_pe_ratio"])

y = (X["Close"].shift(-5) / X["Close"] - 1).rename("fwd_5d_return")
```

**Time series models** (`algorithms/machine_learning_algorithms/time_series_models/`):

| Module | Model |
|---|---|
| `arima.py` | ARIMA(p,d,q) order selection + fitting |
| `garch.py` | GARCH volatility modelling |
| `arch.py` | ARCH heteroskedasticity |
| `volatility_clustering.py` | Regime detection via volatility clustering |
| `var.py` | Vector Autoregression |
| `cointegrations.py` | Johansen cointegration testing |
| `fine_tune/lora.py` | LoRA adapter fine-tuning for transformer LLMs |

---

## 9. Quant Algorithms

`algorithms/`

| Module | Description |
|---|---|
| `greeks/prototype.py` | Options pricing and Greeks (Δ, Γ, Θ, V) — backed by `theory.tex` |
| `macd_rsi/prototype.py` | MACD + RSI strategy prototype |
| `market_making_algorithms/var_calculations/prototype.py` | VAR-based market-making prototype |
| `execution_algorithms/` | Order execution algorithms (TWAP, VWAP — in `models/execution_models/`) |
| `monte_carlo/` | Monte Carlo simulation (placeholder) |
| `volatility_forecasting/` | Full production volatility stack (see §10) |

---

## 10. Volatility Models & Trading Signals

`algorithms/volatility_forecasting/` (the deepest production module)

### Volatility Models
- **SABR** (`sabr_pricer.py`) — stochastic vol surface calibrated to market IV
- **Heston** — mean-reverting variance model
- **Calibration** (`calibration/`): objective function minimisation, constraint handling, caplet stripping, data acquisition

### Market Validation (`market_validation/`)
- `greeks_validation.py` — Δ, Γ, V accuracy; put-call parity check
- `parameter_diagnostics.py` — autocorrelation, regime detection, path analysis
- `rolling_validation.py` — rolling out-of-sample forecast error
- `run_all_validations.py` — single-command full validation pass

### Signal Generation (`signals/`) — ⚠️ Research only
- `signal_generator.py` — IV-vs-model-IV dispersion signals
- `strategy_signals.py` — multi-leg strategies (straddles, spreads, calendars)
- `run_signals.py` — CLI runner with CSV export
- Transaction cost modelling, bid-ask reality checks, stress tests (2×/3× costs)

```bash
# Full calibration + signal + validation pipeline
cd algorithms/volatility_forecasting
python volatility_models/calibration/run_calibration.py --ticker SPY --model sabr
python volatility_models/signals/run_signals.py        --ticker SPY --model sabr --export-csv
python volatility_models/market_validation/run_all_validations.py --ticker SPY --model sabr
```

**Example output (SPY SABR):**
```
α=1.15  ν=0.64  ρ=0.56  |  Surface: 24 pts  |  Greeks MAE: Δ=0.041
Signal: LONG_STRADDLE  |  Edge: 300 bps raw → 54 bps net after costs
```

### Portfolio Engine (`portfolio_engine/`)
- `portfolio_constructor.py` — weights allocation
- `position_sizer.py` — Kelly / risk-parity sizing
- `execution_simulator.py` — fill simulation with slippage
- `pipeline.py` — end-to-end portfolio construction pipeline

### Backtest Engine (`backtest_engine/`)
- `engine.py`, `attribution.py`, `stress.py`, `schemas.py`
- Rolling P&L attribution, drawdown analysis, stress scenarios

---

## 11. Interest Rate & Pricing Models

`models/`

| Model | Location | Description |
|---|---|---|
| Hull-White | `interest_rate_models/hull_white/` | One-factor short-rate model |
| HJM | `interest_rate_models/hjm/` | Heath-Jarrow-Morton forward-rate model |
| FMM | `interest_rate_models/fmm/` | Finite-difference method term structure |
| LMM/BGM | `interest_rate_models/llm_bgm/` | LIBOR market model (multi-factor) |
| TWAP | `execution_models/TWAP/` | Time-Weighted Average Price execution |
| VWAP | `execution_models/VWAP/` | Volume-Weighted Average Price execution |
| Asset Allocation | `asset_allocation_models/` | Portfolio optimisation (placeholder) |
| Credit Risk | `credit_risk_models/` | Credit model (placeholder) |
| Equity Options | `equity_options_pricing_models/` | Options pricing (placeholder) |

---

## 12. Backtesting & Portfolio Engine

`backtesting/` — top-level entry points (delegate to `algorithms/volatility_forecasting/backtest_engine/`)

- `/api/backtest` (POST) — accepts strategy name, symbol, date range, parameters
- `/api/backtest/strategies` (GET) — lists available strategies

`algorithms/volatility_forecasting/backtest_engine/`:
- `engine.py` — core P&L loop
- `attribution.py` — factor attribution
- `stress.py` — tail risk / scenario stress tests
- `schemas.py` — typed dataclasses for results

---

## 13. High-Performance Components

### C++ Execution Engine (`performance/cpp_execution/`)

Low-latency order routing built in C++20 with CMake (`-O3 -march=native -flto`). Python integration via **pybind11**:
```python
from cpp_bindings import ExecutionEngine, OrderBook
engine   = ExecutionEngine()
order_id = engine.submit_order(symbol="NVDA", quantity=100, price=137.5)
```

**Design targets (not yet benchmarked — no benchmark suite in repo):**

| Component | Target |
|---|---|
| Order book update | < 1 μs |
| Order submission | < 10 μs |
| Market-making quote | < 5 μs |

> ⚠️ Current implementation uses `std::map` (red-black tree, O(log n)) for the order book price levels and calls `match_orders()` on every insert. These structures will not hit sub-microsecond targets under real load — lock-free flat arrays or hash maps are needed. The `benchmarks/` directory referenced in the C++ README does not yet exist.

### Go Risk Service (`performance/go_services/`)

Concurrent risk enforcement layer exposed over **gRPC**.

| Capability | Detail |
|---|---|
| Concurrency | Goroutines + channels + context propagation |
| Reliability | Circuit breakers, retry logic, graceful shutdown |
| Observability | Prometheus metrics, OpenTelemetry tracing, structured logs |
| API | gRPC (`.proto` definitions in `proto/`) |

**Design targets (not yet benchmarked):** throughput > 100k msg/s, risk check latency < 100 μs.

> ⚠️ Current implementation uses `sync.RWMutex` with `shopspring/decimal` arbitrary-precision arithmetic on every position check. This is correct for numerical accuracy but adds overhead relative to integer/fixed-point approaches. No benchmark files exist in the repo yet.

---

## 14. Economic Calendar

`backend/economic_events.json` — 40+ events for 2026.

**Categories:**
- **Market Holidays** — NYSE closures (New Year, MLK Day, Presidents' Day, Good Friday, Memorial Day, Independence Day, Labor Day, Thanksgiving, Christmas)
- **Economic Data** — Non-Farm Payrolls, CPI, PPI, GDP, Retail Sales, Durable Goods, Unemployment Claims
- **FOMC Meetings** — Federal Reserve rate decisions (Jan 29, Mar 18, May 6, Jun 17, Jul 29, Sep 16, Oct 28, Dec 9)

Served via `/api/calendar` and rendered in the Calendar tab with importance badges and type filters.

---

## 15. Security & Key Management

All API credentials are loaded in priority order: **environment variable → `keys.txt`**. `keys.txt` is listed in `.gitignore` and never committed. No secret values exist anywhere in the source code.

| Provider | Environment variable |
|---|---|
| Finnhub | `FINNHUB_API_KEY` |
| Polygon | `POLYGON_API_KEY` |
| Fiscal.ai | `FISCAL_AI_API_KEY` |
| Alpaca key ID | `ALPACA_API_KEY` |
| Alpaca secret | `ALPACA_SECRET_KEY` |
| NVIDIA LLM | `NVIDIA_API_KEY` |
| Twitter/X | `TWITTER_API_KEY`, `TWITTER_API_SECRET`, `TWITTER_BEARER_TOKEN` |
| chart-img.com | `CHART_IMG_KEY` |

On Railway, all of the above are injected as project environment variables. Locally, export them in your shell or write them to `keys.txt`.

---

## 16. Quick Start

```bash
# 1 — Clone & install
git clone https://github.com/aarya127/quant_algorithms_ai.git
cd quant_algorithms_ai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install alpaca-py

# 2 — Set API keys (minimum: Finnhub for most features)
export FINNHUB_API_KEY=your_key
export POLYGON_API_KEY=your_key        # optional — Polygon OHLCV (free tier)
export ALPACA_API_KEY=your_key_id      # optional — Alpaca OHLCV/news
export ALPACA_SECRET_KEY=your_secret
export FISCAL_AI_API_KEY=your_key     # optional — structured fundamentals

# 3 — Download FinBERT once
python -c "from transformers import pipeline; pipeline('text-classification', model='ProsusAI/finbert')"

# 4 — Run the web app
cd backend && python app.py
# → http://localhost:5000

# 5 — Test the data pipeline
python -c "
import sys; sys.path.insert(0,'.')
from algorithms.machine_learning_algorithms.data_pipelines import DataExtractor
DataExtractor().test_apis('AAPL')
"
```

---

## 17. Project Structure

```
quant_algorithms_ai/
├── backend/                              Flask app + static assets
│   ├── app.py                              29 REST endpoints, TTL cache
│   ├── economic_events.json                40+ calendar events 2026
│   ├── templates/index.html                SPA (Bootstrap 5, Chart.js)
│   ├── static/css|js|research/
│   └── entrypoint.sh                       Gunicorn startup script
│
├── algorithms/
│   ├── machine_learning_algorithms/
│   │   ├── data_pipelines/
│   │   │   ├── extractor.py              ★ Multi-source data extraction
│   │   │   ├── transform.py              ★ 84-col ML feature matrix
│   │   │   └── __init__.py
│   │   ├── time_series_models/           ARIMA, GARCH, VAR, cointegration
│   │   └── fine_tune/lora.py             LoRA fine-tuning
│   ├── volatility_forecasting/
│   │   ├── backtest_engine/              P&L, attribution, stress
│   │   ├── portfolio_engine/             sizing, construction, execution sim
│   │   ├── volatility_models/
│   │   │   ├── sabr_pricer.py
│   │   │   ├── calibration/
│   │   │   ├── market_validation/
│   │   │   └── signals/
│   │   └── research/
│   ├── greeks/                           Options Greeks + theory
│   ├── macd_rsi/                         MACD/RSI strategy
│   └── market_making_algorithms/         VAR-based market making
│
├── models/
│   ├── interest_rate_models/             Hull-White, HJM, FMM, LMM/BGM
│   └── execution_models/TWAP|VWAP/
│
├── data/                                 Per-provider API modules
│   ├── finnhub.py, alphavantage.py
│   ├── alpaca_news.py, charts.py
│   ├── company_statistics.py, prices.py
│   ├── nvidia_llm.py, twitter_feed.py
│   └── streaming/
│
├── sentiment/
│   ├── finbert.py                        FinBERT US stocks
│   └── finbert_canadian.py              FinBERT Canadian stocks
│
├── performance/
│   ├── cpp_execution/                    < 10 μs C++ order engine (pybind11)
│   └── go_services/                      100k msg/s Go risk engine (gRPC)
│
├── backtesting/                          Top-level backtest entry points
├── quant_research/stochastic_volatility/ Research papers
├── tests/                                API integration tests
├── Dockerfile                            Railway deploy image
├── railway.toml                          Railway config
├── requirements.txt
└── keys.txt                              ← .gitignored, never committed
```

---

## 18. Roadmap

### In Progress
- [ ] Deep learning (LSTM, Transformer) in `deep_learning/`
- [ ] Factor strategies — momentum, mean reversion, quality
- [ ] Full backtesting harness wired to `DataTransformer` feature matrix
- [ ] Point-in-time fundamental database for rigorous backtest training

### Completed ✅
- [x] **Unified data extraction layer** (`DataExtractor`) — 5 providers, auto-fallback
- [x] **84-column ML feature matrix** (`DataTransformer`) — all sources merged daily
- [x] **Security audit** — all API keys removed from source; env var / keys.txt only
- [x] **Alpaca SDK migration** — raw HTTP → `alpaca-py` SDK
- [x] **Fiscal.ai integration** — company-ID format, 26 free-tier tickers mapped
- [x] SABR / Heston calibration pipeline
- [x] Trading signals with transaction costs and reality checks
- [x] Market validation suite (Greeks, rolling OOS, parameter diagnostics)
- [x] FinBERT sentiment analysis (US + Canadian)
- [x] NVIDIA LLM company overviews (Mistral Large 3)
- [x] Economic calendar (40+ events, 2026)
- [x] Portfolio engine (constructor, sizer, execution simulator)
- [x] C++ execution engine (sub-10 μs, pybind11)
- [x] Go risk service (100k msg/s, gRPC, Prometheus)
- [x] Multi-exchange stock search (NYSE, NASDAQ, TSX)
- [x] Railway + Docker deployment with health checks

---

## 📄 License

MIT License — see LICENSE file

## 📧 Contact

**Aarya Shah** — [@aarya127](https://github.com/aarya127)


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
- **Trading Signals**: Model-based signals with transaction cost modeling and reality checks
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

## 📊 Volatility Models & Trading Signals

Production-grade SABR/Heston calibration with signal generation and market validation:

```bash
cd models/volatility_models/calibration

# Step 1: Run calibration
python run_calibration.py --ticker SPY --model sabr

# Step 2: Generate trading signals
cd ../signals
python run_signals.py --ticker SPY --model sabr --export-csv

# Step 3: Validate model
cd ../market_validation
python run_all_validations.py --ticker SPY --model sabr
```

**Calibration Features:**
- Surface density gates (production desk standards)
- Parameter path diagnostics with regime detection
- Greeks validation with put-call parity checks
- Rolling window out-of-sample testing

**Signal Generation Features (⚠️ Research Only):**
- Transaction cost modeling (spreads + commissions)
- Reality checks (liquidity, bid-ask spreads, strike availability)
- Multi-leg strategies (spreads, calendars, volatility trades)
- Stress testing (2x/3x cost scenarios, liquidity shocks)
- Net edge calculation with minimum threshold (50 bps)

**Recent Results (SPY SABR):**
```
Calibration:
  Parameters: α=1.15, ν=0.64, ρ=0.56
  Surface: 24 IV points, 4.7s execution
  Greeks MAE: Δ=0.041, Γ=0.00002
  Put-call parity: PASS ✓

Signal Example:
  Signal: LONG_STRADDLE
  Model IV: 25.0%, Market IV: 22.0%
  Raw edge: 300 bps
  Costs: 246 bps (spread 116, commission 130)
  Net edge: 54 bps ✓
  Stress test: Breaks at 2x spreads
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
│       ├── signals/      # Trading signal generation (research)
│       │   ├── signal_generator.py
│       │   ├── strategy_signals.py
│       │   ├── run_signals.py
│       │   └── README.md
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
- [x] Trading signals with transaction costs & reality checks
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