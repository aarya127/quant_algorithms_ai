# 🤖 Quant Algorithms AI

An end-to-end quantitative research and trading platform. The system combines a real-time web dashboard, a multi-source market data pipeline, FinBERT AI sentiment analysis, NVIDIA LLM overviews, volatility model calibration (SABR/Heston), an ML feature-matrix builder with **daily automated model retraining**, a high-performance C++ execution engine, and a Go-based risk service — all deployed on Render via Docker.

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
9. [Scheduled Retraining & MLOps](#9-scheduled-retraining--mlops)
10. [Quant Algorithms](#10-quant-algorithms)
11. [Volatility Models & Trading Signals](#11-volatility-models--trading-signals)
12. [Interest Rate & Pricing Models](#12-interest-rate--pricing-models)
13. [Backtesting & Portfolio Engine](#13-backtesting--portfolio-engine)
14. [High-Performance Components](#14-high-performance-components)
15. [Economic Calendar](#15-economic-calendar)
16. [Security & Key Management](#16-security--key-management)
17. [Quick Start](#17-quick-start)
18. [Project Structure](#18-project-structure)
19. [Component Status](#19-component-status)
20. [Roadmap](#20-roadmap)

> **Working in this repo with a coding agent (Claude Code, Copilot, Cursor)?**
> Read [`AGENTS.md`](AGENTS.md) first — it captures the conventions, the pipeline
> contract, and the gotchas that aren't obvious from the code. Task-specific
> playbooks live in [`.claude/skills/`](.claude/skills/).
>
> **Deep per-file references** live in [`docs/`](docs/):
> [data providers](docs/data-providers.md) ·
> [research & analysis scripts](docs/research-scripts.md) ·
> [C++/Go performance components](docs/performance.md).

---

## 1. System Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│  Browser  (Bootstrap 5 + Chart.js)                                   │
│  Dashboard · News · Calendar · Quant · Trading                       │
└───────────────────────────┬──────────────────────────────────────────┘
                            │ HTTP/JSON
┌───────────────────────────▼──────────────────────────────────────────┐
│  Flask Backend  (31 routes, TTL cache, Gunicorn)                     │
│  backend/app.py                                                      │
└──┬──────────────┬──────────────┬──────────────┬──────────────────────┘
   │              │              │              │
   ▼              ▼              ▼              ▼
Data Layer    AI / NLP       Quant Models   Perf Services
(7 APIs)      (FinBERT,      (SABR, Heston, (C++ engine,
              NVIDIA LLM)    time series)   Go risk svc)
   │
   ▼
ML Pipeline + MLOps  (orchestrator → MLflow → model registry → drift alerts)
```

**Stack:**
| Layer | Technology |
|---|---|
| Web framework | Flask 3.0 + Gunicorn |
| Frontend | Bootstrap 5, Chart.js, vanilla JS |
| Primary data | yfinance, Finnhub, Polygon, Alpaca SDK, Fiscal.ai |
| AI / NLP | FinBERT (ProsusAI), NVIDIA Mistral Large 3, multi-provider LLM router |
| Quant | QuantLib, scipy, numpy, pandas |
| ML / MLOps | scikit-learn, XGBoost, LightGBM, MLflow, local model registry |
| High-perf | C++ (pybind11), Go (gRPC) |
| Deployment | Docker, Render (+ GitHub Actions for scheduled retraining) |
| Python | 3.11 (Docker image), `.venv` local |

---

## 2. Deployment

**Docker** (`Dockerfile`):
- Base: `python:3.11-slim`
- CPU-only PyTorch (~180 MB vs 2.5 GB CUDA build)
- FinBERT weights pre-downloaded at build time (so the first request never blocks on a ~440 MB download)
- `backend/entrypoint.sh` launches Gunicorn on port 8080

**Render** (`render.yaml` — Blueprint spec):
```yaml
services:
  - type: web
    name: invest-ai-backend
    runtime: docker
    dockerfilePath: ./Dockerfile
    plan: standard            # 1 CPU / 2 GB RAM — required for FinBERT
    healthCheckPath: /health  # GET /health → 200 OK
    disk:
      name: invest-ai-data
      mountPath: /app/mnt     # persistent — survives redeploys; holds models + feature CSVs
      sizeGB: 5
```

Render auto-builds from the Dockerfile on every push, health-checks via `/health`, and restarts on failure. Secrets are set in the Render dashboard (marked `sync: false` in `render.yaml`), never committed.

> **Note:** the app runs a **single Gunicorn worker** (`entrypoint.sh` hardcodes `--workers 1`). `render.yaml` sets `GUNICORN_WORKERS=2`, but the entrypoint ignores it. This matters — the retraining pipeline's job store lives in-process (see [§9](#9-scheduled-retraining--mlops)). Do not raise the worker count without moving that state to shared storage.

### Two Docker images

| Image | Dockerfile | Contents | Used by |
|---|---|---|---|
| Web app | `Dockerfile` | Flask + Gunicorn + PyTorch + FinBERT (~3.5 GB) | Render web service |
| Pipeline | `Dockerfile.pipeline` | Training deps only from `requirements-pipeline.txt` (no Flask/PyTorch/FinBERT, ~900 MB); `ENTRYPOINT` = `orchestrator.py` | The Kubernetes CronJob below |

### Kubernetes manifests (`k8s/`) — ⚠️ scaffolding, not the live deploy

A complete manifest set (`namespace`, `deployment`, `service`, `ingress`, `hpa`,
`pvc`, `configmap`, `secret`, `cronjob`) for an alternative K8s deployment in
namespace `invest-ai`. It uses **placeholder image tags and hostnames**
(`invest-ai:latest`, `example.com`), nothing in CI applies it, and **Render is the
live target**. Treat `k8s/` as a reference/aspirational setup until wired to a
registry + cluster. Notably `k8s/cronjob.yaml` runs retraining a *different* way
than production — see the scheduling note in [§9](#9-scheduled-retraining--mlops).

### Helper scripts (`scripts/`)

| Script | Purpose |
|---|---|
| `build_cpp.sh` | CMake build of the C++ engine (`performance/cpp_execution/`) |
| `build_go.sh` | Build the Go risk service + protoc codegen (`performance/go_services/`) |
| `build_all.sh` | Runs all three (C++, Go, Python deps) |
| `daily_predict.sh` | **Local-only** cron helper (hardcoded machine paths) — see [§9](#9-scheduled-retraining--mlops) |

### CI (`.github/workflows/ci.yml`)

Runs on every push and PR to `main`. Installs light deps (numpy/pandas/sklearn/
xgboost/lightgbm/pytest — **no torch/transformers**) and runs the four pytest
suites with coverage. It deliberately excludes the heavy ML deps and the
live-server smoke tests. (The daily retrain is a separate workflow — see [§9](#9-scheduled-retraining--mlops).)

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

`backend/app.py` — 31 REST endpoints with a thread-safe TTL cache.

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
| `/api/sentiment/news/<symbol>` | GET | Per-article FinBERT sentiment for recent news |
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
| `/api/research/<paper>/markdown` | GET | Research paper rendered as markdown |
| `/api/research/diagnostics/notebook` | GET | Jupyter diagnostic notebook |
| `/api/algorithm/<name>` | GET | Algorithm source + docs |
| `/api/trading/chart` | POST | Custom TradingView-style chart render |
| `/api/pipeline/run` | POST | **Start a scheduled model-retraining job** (see [§9](#9-scheduled-retraining--mlops)) |
| `/api/pipeline/status/<job_id>` | GET | **Poll a retraining job's status** |

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

Merges all sources into a single daily-indexed ML feature matrix. Point-in-time joins (`merge_asof(direction="backward")`) prevent look-ahead bias.

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

`ai_platform/nvidia_llm.py` + `/api/ai-overview/<symbol>`

Calls **NVIDIA's hosted Mistral Large 3** API to generate a natural-language overview of any company. The stock detail **Overview** tab renders this summary alongside the real-time price and news feed. API key is read from the `NVIDIA_API_KEY` environment variable.

The broader LLM layer lives in `ai_platform/` — see [§9](#9-scheduled-retraining--mlops) for the multi-provider router, signal narrator, and LLM-as-judge.

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

Model loads lazily on first request to avoid blocking the Gunicorn worker at startup.

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

> These are **standalone research scripts** (run by hand, fetch live from yfinance —
> not part of the automated pipeline). Full per-file reference incl. `eda/eda.py`:
> [docs/research-scripts.md](docs/research-scripts.md).

---

## 9. Scheduled Retraining & MLOps

The platform retrains its supervised models on a daily schedule, tracks every run in MLflow, and gates promotion of new models behind performance checks. This is the "ops" layer wrapped around the [ML feature matrix](#8-ml-feature-matrix).

### Orchestrator (`algorithms/machine_learning_algorithms/orchestrator.py`)

Chains five pipeline steps, each run as a subprocess (`python <script>.py TICKER`). Every stage reads the previous stage's CSV and writes the next, all keyed by a `<SYMBOL>_` prefix in `data_pipelines/`:

| # | Step | Script | Reads → Writes |
|---|---|---|---|
| 1 | extract | `data_pipelines/run_pipeline.py` | APIs → `<SYM>_features.csv` (incremental append) |
| 2 | clean | `data_pipelines/clean.py` | `<SYM>_features.csv` → `<SYM>_features_clean.csv` |
| 3 | normalize | `data_pipelines/normalize.py` | `..._clean.csv` → `..._normalized.csv`, `<SYM>_scaler.pkl`, `<SYM>_targets.csv` |
| 4 | unsupervised | `unsupervised/unsupervised.py` | `..._normalized.csv` → `..._with_regimes.csv` (PCA/KMeans/IsolationForest) |
| 5 | supervised | `supervised/supervised.py` | `..._with_regimes.csv` → model registry + holdout results |

The orchestrator emits a **line protocol** parsed by the backend, and wraps the run in an MLflow trace (one child span per step) before running the LLM-as-judge:

```
STEP:<name>:start     — step beginning          STATUS:up_to_date   — data current; ML steps skipped
LOG:<text>            — passthrough log line     STATUS:done         — all steps completed
STEP:<name>:done      — step completed           STATUS:error:<name> — step failed (exit 1)
```

Run it directly:
```bash
python algorithms/machine_learning_algorithms/orchestrator.py NVDA
# USE_TUNING=1 enables hyperparameter search (default 0 for the fast daily path)
```

### Triggering via the API

The pipeline runs **inside the Render web container** as a background thread, so it shares the persistent disk where models and feature CSVs live.

```bash
# Start a job
curl -X POST https://<app>.onrender.com/api/pipeline/run \
  -H "Content-Type: application/json" -d '{"ticker":"NVDA"}'
# → {"success": true, "job_id": "a1b2c3d4", "status": "queued"}

# Poll it
curl https://<app>.onrender.com/api/pipeline/status/a1b2c3d4
# → {"status": "running", "current_step": "supervised", "last_logs": [...]}
```

`status` ∈ `queued` → `running` → `done` | `up_to_date` | `error`.

> **Design caveat:** the job store (`_PIPELINE_JOBS` in `app.py`) is an **in-process dict**. It is correct only under a single Gunicorn worker (see [§2](#2-deployment)); with multiple workers, the POST and the poll could hit different processes and status would 404. A worker restart mid-run also orphans the job. Fine for the current single-worker daily cadence; revisit before scaling out.

### Daily schedule (`.github/workflows/daily-retrain.yml`)

A GitHub Actions workflow (cron `0 2 * * 1-5` — 02:00 UTC weekdays, plus manual dispatch) POSTs to `/api/pipeline/run` on the live app, then polls `/status` every 90 s until the job finishes. The run's green/red badge is your daily retraining health signal.

**Why GitHub Actions and not a Render Cron Job?** Render disks attach to only one service, so a separate cron job couldn't share the trained models with the web service. Running the pipeline *inside* the web container sidesteps that.

**Setup:** add a repo secret `RENDER_APP_URL` (`https://<app>.onrender.com`, no trailing slash) under Settings → Secrets and variables → Actions. If this is missing — or the `/api/pipeline/*` endpoints aren't deployed yet — the workflow fails within seconds on the first `curl`.

### Three scheduling paths exist — only one is live

The repo contains three mechanisms that all run the same retrain. Know which is real:

| Path | When | Where | Status |
|---|---|---|---|
| `.github/workflows/daily-retrain.yml` | 02:00 UTC Mon–Fri | Render (via the API) | **✅ Active — this is production** |
| `k8s/cronjob.yaml` | 02:00 UTC Mon–Fri (same) | Kubernetes (runs `Dockerfile.pipeline`) | ⚠️ Scaffolding — unused (see [§2](#2-deployment)) |
| `scripts/daily_predict.sh` | 22:00 UTC / 5 PM ET | A local dev machine (hardcoded paths) | 🔧 Local convenience only |

They don't run concurrently (different platforms), but only the GitHub Action is
wired to the live app. The K8s CronJob is the "separate cron + shared PVC" pattern
that Render can't do; if you ever migrate to K8s it becomes the real scheduler.

### MLflow tracking (`mlflow.db`, `mlruns/`)

SQLite backend at the repo root, one experiment per ticker, runs named `<target>_v<version>_<date>` (e.g. `target_5d_vB_2026-05-30`). Browse locally with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

### Model registry (`algorithms/machine_learning_algorithms/supervised/model_registry/`)

Per ticker: an `active.json` index plus one directory per target holding `model.pkl`, scalers, `features.json`, `train_stats.json` (drift baselines), `metadata.json`, and a `prev/` snapshot for rollback.

**Promotion gate** (`registry.py`) — a new model replaces production only if it clears **all three** (unless forced):
1. Absolute floor — regression `ic > 0.02`, classification `f1_w > 0.30`
2. Beats the naive baseline by ≥ 5%
3. Improves over the current production model by ≥ 0.005

The outgoing model is copied to `prev/` first, so `rollback_registry()` can restore it.

### Serving & drift monitoring (`backend/predictor.py`)

`predict_latest()` serves predictions from the registry; the module also computes **data drift** (feature z-score > 3.0) and **model drift** (rolling IC falling > 0.10 below the registered IC), writing reports to `supervised/output/monitoring/` and POSTing alerts to `ALERT_WEBHOOK_URL` (Slack-style webhook) when thresholds trip.

> **⚠️ Not yet wired:** `predictor.py`'s docstring advertises `GET /api/predict/<ticker>`, `/api/drift/<ticker>`, and `/api/model/status`, but **those routes are not registered in `app.py`** (neither in HEAD nor the working tree). The serving/monitoring logic exists as importable functions; the HTTP endpoints still need to be added.

### LLM layer (`ai_platform/`)

- `llm_router.py` — unified multi-provider client (OpenAI → Anthropic → NVIDIA priority; override with `LLM_PROVIDER`)
- `signal_narrator.py` — turns predictions into prose (`narrate_predictions()`) — ⚠️ scaffolded, not yet wired to a caller
- `llm_judge.py` — `run_judge(ticker)` runs at the end of the orchestrator, asks an LLM for a promotion verdict, and writes it to `supervised/output/monitoring/`

---

## 10. Quant Algorithms

`algorithms/`

Research algorithms follow a **`theory.tex` + `prototype.py`** convention — a LaTeX write-up beside a runnable Python prototype.

| Module | Description |
|---|---|
| `greeks/prototype.py` | Options pricing and Greeks (Δ, Γ, Θ, V) — backed by `theory.tex` |
| `macd_rsi/prototype.py` | MACD + RSI strategy prototype |
| `market_making_algorithms/var_calculations/prototype.py` | VAR-based market-making prototype |
| `execution_algorithms/` | Order execution research (market microstructure, advanced trading) |
| `volatility_forecasting/` | Full production volatility stack (see [§11](#11-volatility-models--trading-signals)) |

---

## 11. Volatility Models & Trading Signals

`algorithms/volatility_forecasting/` (the deepest production module)

### Volatility Models
- **SABR** (`volatility_models/sabr_pricer.py`) — stochastic vol surface calibrated to market IV
- **Heston** — mean-reverting variance model
- **Calibration** (`volatility_models/calibration/`): objective function minimisation, constraint handling, caplet stripping, data acquisition

### Market Validation (`volatility_models/market_validation/`)
- `greeks_validation.py` — Δ, Γ, V accuracy; put-call parity check
- `parameter_diagnostics.py` — autocorrelation, regime detection, path analysis
- `rolling_validation.py` — rolling out-of-sample forecast error
- `run_all_validations.py` — single-command full validation pass

### Signal Generation (`volatility_models/signals/`) — ⚠️ Research only
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

## 12. Interest Rate & Pricing Models

`models/` — scaffolding for pricing-model research (directories present; implementations in progress).

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

> **Note:** trained ML model artefacts do **not** live here — they live in `algorithms/machine_learning_algorithms/supervised/model_registry/` (see [§9](#9-scheduled-retraining--mlops)). `models/` is for pricing-theory research.

---

## 13. Backtesting & Portfolio Engine

`backtesting/` — top-level entry points (delegate to `algorithms/volatility_forecasting/backtest_engine/`)

- `/api/backtest` (POST) — accepts strategy name, symbol, date range, parameters
- `/api/backtest/strategies` (GET) — lists available strategies

`algorithms/volatility_forecasting/backtest_engine/`:
- `engine.py` — core P&L loop
- `attribution.py` — factor attribution
- `stress.py` — tail risk / scenario stress tests
- `schemas.py` — typed dataclasses for results

---

## 14. High-Performance Components

> Full per-file API (C++ order book/engine, pybind11 surface, Go gRPC service,
> build steps, and the "not wired into the app / won't build as-is" caveats):
> [docs/performance.md](docs/performance.md).

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

## 15. Economic Calendar

`backend/economic_events.json` — 40+ events for 2026.

**Categories:**
- **Market Holidays** — NYSE closures (New Year, MLK Day, Presidents' Day, Good Friday, Memorial Day, Independence Day, Labor Day, Thanksgiving, Christmas)
- **Economic Data** — Non-Farm Payrolls, CPI, PPI, GDP, Retail Sales, Durable Goods, Unemployment Claims
- **FOMC Meetings** — Federal Reserve rate decisions (Jan 29, Mar 18, May 6, Jun 17, Jul 29, Sep 16, Oct 28, Dec 9)

Served via `/api/calendar` and rendered in the Calendar tab with importance badges and type filters.

---

## 16. Security & Key Management

All API credentials are loaded in priority order: **environment variable → `keys.txt`**. `keys.txt` is listed in `.gitignore` and never committed. No secret values exist anywhere in the source code.

| Provider | Environment variable |
|---|---|
| Finnhub | `FINNHUB_API_KEY` |
| Polygon | `POLYGON_API_KEY` |
| Fiscal.ai | `FISCAL_AI_API_KEY` |
| Alpaca key ID | `ALPACA_API_KEY` |
| Alpaca secret | `ALPACA_SECRET_KEY` |
| NVIDIA LLM | `NVIDIA_API_KEY` |
| OpenAI / Anthropic (LLM router) | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` |
| Twitter/X | `TWITTER_API_KEY`, `TWITTER_API_SECRET`, `TWITTER_BEARER_TOKEN` |
| chart-img.com | `CHART_IMG_KEY` |
| Alerting webhook | `ALERT_WEBHOOK_URL` |

On Render, all of the above are set as service environment variables (dashboard, `sync: false`). Locally, export them in your shell or write them to `keys.txt`.

---

## 17. Quick Start

```bash
# 1 — Clone & install
git clone https://github.com/aarya127/quant_algorithms_ai.git
cd quant_algorithms_ai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2 — Set API keys (minimum: Finnhub for most features)
export FINNHUB_API_KEY=your_key
export POLYGON_API_KEY=your_key        # optional — Polygon OHLCV (free tier)
export ALPACA_API_KEY=your_key_id      # optional — Alpaca OHLCV/news
export ALPACA_SECRET_KEY=your_secret
export FISCAL_AI_API_KEY=your_key      # optional — structured fundamentals

# 3 — Download FinBERT once
python -c "from transformers import pipeline; pipeline('text-classification', model='ProsusAI/finbert')"

# 4 — Run the web app
cd backend && python app.py
# → http://localhost:5001   (PORT env var overrides; Docker/Render use 8080)

# 5 — Test the data pipeline
python -c "
import sys; sys.path.insert(0,'.')
from algorithms.machine_learning_algorithms.data_pipelines import DataExtractor
DataExtractor().test_apis('AAPL')
"

# 6 — Run the full ML pipeline once (extract → … → supervised)
python algorithms/machine_learning_algorithms/orchestrator.py NVDA
```

**Build the polyglot components** (optional — C++/Go):
```bash
make all        # cpp + go + python deps
make test       # ctest + go test + tests/run_all_tests.py
make run-backend
```

---

## 18. Project Structure

```
quant_algorithms_ai/
├── backend/                              Flask app + static assets
│   ├── app.py                              31 REST endpoints, TTL cache, pipeline jobs
│   ├── predictor.py                        prediction serving + drift monitoring
│   ├── economic_events.json                40+ calendar events 2026
│   ├── templates/index.html                SPA (Bootstrap 5, Chart.js)
│   ├── static/css|js|research/
│   └── entrypoint.sh                       Gunicorn startup (--workers 1)
│
├── algorithms/
│   ├── machine_learning_algorithms/
│   │   ├── orchestrator.py               ★ 5-step retraining pipeline driver
│   │   ├── data_pipelines/
│   │   │   ├── extractor.py              ★ Multi-source data extraction
│   │   │   ├── transform.py              ★ 84-col ML feature matrix
│   │   │   ├── run_pipeline.py           extract step (incremental)
│   │   │   ├── clean.py / normalize.py   clean + target construction
│   │   │   └── __init__.py
│   │   ├── supervised/                   ★ model ladders, walk-forward, registry
│   │   │   ├── models.py                   ← add a new model here
│   │   │   ├── registry.py                 promotion gate + rollback
│   │   │   ├── mlflow_tracker.py
│   │   │   └── model_registry/<TICKER>/    trained artefacts (active.json + per-target)
│   │   ├── unsupervised/unsupervised.py  PCA / KMeans / IsolationForest regimes
│   │   ├── factor_discovery/             recommended_features.txt (feature allow-list)
│   │   ├── time_series_models/           ARIMA, GARCH, VAR, cointegration
│   │   └── eda/
│   ├── volatility_forecasting/
│   │   ├── backtest_engine/              P&L, attribution, stress
│   │   ├── portfolio_engine/             sizing, construction, execution sim
│   │   ├── volatility_models/            sabr_pricer, calibration, market_validation, signals
│   │   └── research/                     theory.tex write-ups
│   ├── greeks/  macd_rsi/  execution_algorithms/  market_making_algorithms/
│   └── (each: theory.tex + prototype.py)
│
├── ai_platform/                          LLM layer
│   ├── llm_router.py                       multi-provider client
│   ├── signal_narrator.py                  predictions → prose
│   ├── llm_judge.py                        LLM-as-judge promotion verdict
│   └── nvidia_llm.py
│
├── models/                               pricing-model research scaffolding
│   ├── interest_rate_models/             Hull-White, HJM, FMM, LMM/BGM
│   └── execution_models/TWAP|VWAP/
│
├── data/                                 Per-provider API modules
├── sentiment/                            FinBERT (US + Canadian)
├── performance/
│   ├── cpp_execution/                    C++ order engine (pybind11)
│   └── go_services/                      Go risk engine (gRPC)
│
├── backtesting/                          Top-level backtest entry points
├── tests/                                pytest unit/integration + API smoke tests
├── .github/workflows/                    ci.yml, daily-retrain.yml
├── .claude/skills/                       coding-agent playbooks
├── AGENTS.md / CLAUDE.md                 coding-agent guide
├── mlflow.db / mlruns/                   MLflow tracking store
├── Dockerfile                            Render deploy image
├── render.yaml                           Render Blueprint
├── requirements.txt
└── keys.txt                              ← .gitignored, never committed
```

---

## 19. Component Status

Not everything in the tree is production code. This repo mixes shipped features,
research prototypes, and scaffolding for planned work. Use this as the honest map.

**✅ Working & wired into the app**
- Flask backend + all data providers (`data/`), FinBERT sentiment (`sentiment/`),
  NVIDIA LLM overview + multi-provider router (`ai_platform/`)
- The full ML retraining pipeline (`orchestrator.py` + `data_pipelines/` +
  `supervised/` + `unsupervised/`), model registry, MLflow, and the daily retrain
- Volatility stack (`volatility_forecasting/`: SABR/Heston calibration, market
  validation, backtest & portfolio engines)

**🔬 Real code, run standalone (not called by the app/pipeline)**
- `time_series_models/` (ARIMA, GARCH, VAR, cointegration — ~2,000 lines; fetch
  live from yfinance, not the feature CSVs)
- `factor_discovery/` and `eda/eda.py` (produce the feature allow-list and EDA plots)
- `greeks/`, `macd_rsi/` prototypes
- `performance/cpp_execution/` (C++ order book/engine) and `performance/go_services/`
  (Go gRPC risk engine) — real but **not imported by the deployed Flask app**; their
  READMEs describe more subdirs than actually exist

**🚧 Not yet wired**
- `backend/predictor.py` serving/monitoring — functions exist; the `/api/predict`,
  `/api/drift`, `/api/model/status` routes are not registered (see [§9](#9-scheduled-retraining--mlops))
- `ai_platform/signal_narrator.py` — no caller yet

**📦 Placeholder / empty (present in the tree, no implementation)**
- All of top-level `models/` (interest-rate, execution, allocation, credit, options)
- `deep_learning/`, `monte_carlo/`, `data/streaming/`, `backtesting/*.py` (empty files)
- Several `prototype.py` stubs under `execution_algorithms/`, `market_making_algorithms/`,
  and `volatility_forecasting/research/` (theory `.tex` only)

---

## 20. Roadmap

### In Progress
- [ ] Wire `predictor.py` serving/monitoring into HTTP routes (`/api/predict`, `/api/drift`, `/api/model/status`)
- [ ] Deep learning (LSTM, Transformer) in `deep_learning/`
- [ ] Factor strategies — momentum, mean reversion, quality
- [ ] Full backtesting harness wired to `DataTransformer` feature matrix
- [ ] Point-in-time fundamental database for rigorous backtest training

### Completed ✅
- [x] **Scheduled daily retraining** — orchestrator + GitHub Actions + `/api/pipeline/*`
- [x] **MLflow experiment tracking** — per-ticker experiments, per-step spans
- [x] **Model registry with promotion gate + rollback** — `supervised/registry.py`
- [x] **Drift detection & webhook alerting** — data-drift (z-score) + model-drift (rolling IC)
- [x] **LLM-as-judge** promotion verdicts + multi-provider LLM router
- [x] **Unified data extraction layer** (`DataExtractor`) — 5 providers, auto-fallback
- [x] **84-column ML feature matrix** (`DataTransformer`) — all sources merged daily
- [x] **Security audit** — all API keys removed from source; env var / keys.txt only
- [x] SABR / Heston calibration pipeline + market validation suite
- [x] Trading signals with transaction costs and reality checks
- [x] FinBERT sentiment analysis (US + Canadian)
- [x] NVIDIA LLM company overviews (Mistral Large 3)
- [x] Economic calendar (40+ events, 2026)
- [x] Portfolio engine (constructor, sizer, execution simulator)
- [x] C++ execution engine (pybind11) + Go risk service (gRPC, Prometheus)
- [x] Multi-exchange stock search (NYSE, NASDAQ, TSX)
- [x] Render + Docker deployment with health checks and persistent disk

---

## 📄 License

MIT License — see LICENSE file

## 📧 Contact

**Aarya Shah** — [@aarya127](https://github.com/aarya127)
