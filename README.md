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
7. [Sentiment Analysis](#7-sentiment-analysis)
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
| AI / NLP | FinBERT (ProsusAI, pipeline features), NVIDIA Mistral Large 3 (overviews), Nemotron 3.5 Super → Nemotron 3.5 Lightning (chat/briefs), multi-provider LLM router |
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
      mountPath: /app/mnt     # unused — nothing the app needs lives here
      sizeGB: 5
```

The live service actually runs on the free tier (512 MB) with no persistent disk; trained models come from the `models` branch (see [§9](#9-scheduled-retraining--mlops)). Render auto-builds from the Dockerfile on every push, health-checks via `/health`, and restarts on failure. Secrets are set in the Render dashboard (marked `sync: false` in `render.yaml`), never committed.

> **Note:** worker count is `GUNICORN_WORKERS` (default **1**; `entrypoint.sh` honors the env var). The retrain job store is now a shared SQLite store (see [§9](#9-scheduled-retraining--mlops)), so >1 worker is safe for correctness — but each sync worker can lazily load FinBERT (~512 MB), so keep it at **1** on the 512 MB free tier and only raise it after upgrading RAM.

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
| **Sentiment** | Alpha Vantage ticker sentiment per article (relevance-weighted overall score, ±0.15 neutral band) + Finnhub headlines, daily trend over 7 days – 1 year; says "no scored news" instead of a neutral 0 when Alpha Vantage is out |
| **Scenarios** | Bull / base / bear = 80th / 50th / 20th percentiles of a filtered historical simulation at 1W – 1Y; the ML 5-day forecast drives week one when fresh and plausible; a history check of the band on non-overlapping past windows |
| **Metrics & Grades** | Finnhub fundamentals graded A–F by percentile among up to 8 industry peers (market-wide scale as fallback); negative-earnings/equity inputs marked not meaningful; balance-sheet health skipped for financials; ETFs not graded |
| **Recommendations** | Buy / Hold / Sell per horizon from weighted votes — price trend (200-day average + 12-month momentum), the served ML model (1W/1M, if fresh), quality grade (3M+) — with each price rule backtested on the ticker against buy-and-hold |

---

## 4. API Layer (Flask Backend)

`backend/app.py` — 31 REST endpoints with a thread-safe TTL cache.

**Cache TTLs:** quotes 30 s · profile 1 h · financials/news 10 min

| Route | Method | Purpose |
|---|---|---|
| `/health` | GET | Deployment health check |
| `/` | GET | SPA entry point |
| `/api/search/<query>` | GET | Symbol/name search (yfinance Search, plus the NYSE/NASDAQ/TSX lists) |
| `/api/indices` | GET | Live index levels (SPX, RUT, INDU, VIX) |
| `/api/indices/history` | GET | 30-day index performance |
| `/api/dashboard` | GET | Aggregated overview for a stock |
| `/api/stock/<symbol>` | GET | Price + profile + news bundle |
| `/api/ai-overview/<symbol>` | GET | NVIDIA Mistral Large 3 company overview |
| `/api/statistics/<symbol>` | GET | Ratios, margins, growth metrics |
| `/api/sentiment/news/<symbol>` | GET | Sentiment tab: Alpha Vantage-scored articles + Finnhub headlines, overall score, daily trend (`?days=`, ≤ 365) |
| `/api/scenarios/<symbol>` | GET | Bull/base/bear percentiles, risk stats and a history check (`?timeframe=1W…1Y`) |
| `/api/metrics/<symbol>` | GET | Fundamental grades vs industry peers |
| `/api/recommendations/<symbol>` | GET | Per-horizon Buy/Hold/Sell votes + backtested evidence |
| `/api/narrate/<symbol>` | GET | AI Signal Brief — LLM narrative over computed numbers (cached 1 h) |
| `/api/chat` | POST | AI chat (streams; sees the user's screen and mentioned tickers) |
| `/api/llm/status` | GET | Which LLM provider/models are configured |
| `/api/predict/<ticker>` | GET | Latest ML predictions from the model registry |
| `/api/drift/<ticker>` | GET | Data- and model-drift report |
| `/api/model/status` | GET | Registry overview |
| `/api/mlflow/runs/<ticker>` | GET | MLflow run history |
| `/api/ml/backtest/<ticker>` | GET | Walk-forward backtest of the trading signal (published by the daily retrain) |
| `/api/paper/quote` | GET | Paper trading: latest prices (+ FX to USD) |
| `/api/paper/fill` | POST | Paper trading: decide fills for open orders |
| `/api/paper/model` | GET | The ML model's paper-trading account |
| `/api/backtest` | POST | Options-volatility strategy backtest (`backend/backtest_runner.py`) |
| `/api/backtest/strategies` | GET | List the options-volatility strategies |
| `/api/calendar` | GET | Earnings (watchlist + top companies), dividends, macro events; `?symbols=` |
| `/api/charts/<symbol>` | GET | OHLCV chart data + indicators |
| `/api/charts/<symbol>/all-timeframes` | GET | 1D / 5D / 1M / 3M / 6M / 1Y / 5Y / MAX |
| `/api/charts/compare` | GET | Multi-ticker chart comparison |
| `/api/charts/<symbol>/indicators` | GET | SMA, EMA, RSI, MACD, ATR, OBV |
| `/api/trading/chart` | POST | Custom TradingView-style chart render |
| `/api/trading/ohlcv` | POST | OHLCV bars for the trading chart |
| `/api/news/twitter` | GET | Twitter/X feed (market influencers) |
| `/api/news/alpaca` | GET | Alpaca real-time news stream |
| `/api/news/combined` | GET | Merged multi-source news |
| `/api/research/<paper>` | GET | Research paper content (LaTeX/markdown) |
| `/api/research/diagnostics/notebook` | GET | Jupyter diagnostic notebook |
| `/api/algorithm/<name>` | GET | Algorithm source + docs |
| `/api/pipeline/run` | POST | **Start a model-retraining job** — local runs; refused on Render (see [§9](#9-scheduled-retraining--mlops)) |
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

## 7. Sentiment Analysis

**Sentiment tab** — `backend/sentiment_feed.py` behind `/api/sentiment/news/<symbol>`:
1. One Alpha Vantage `NEWS_SENTIMENT` call (newest first, up to 1,000 articles) for the window,
   plus Finnhub company news — at most two upstream calls, no sleeps, cached 30 min per
   symbol/window, and a per-process Alpha Vantage budget of 10/day (the daily retrain shares the key)
2. Each article's score is Alpha Vantage's ticker-specific sentiment ∈ [−1, 1]; labels use its
   ±0.15 neutral band; Finnhub headlines are shown unscored
3. The overall score and the daily trend are the same relevance-weighted average. With no scored
   article (quota used up, a ticker Alpha Vantage doesn't cover) the tab says so instead of
   showing a neutral 0, and notes when the 1,000-article cap shortens the covered window

**FinBERT** — `sentiment/finbert.py` (US) and `finbert_canadian.py` score raw headlines with
**ProsusAI/finbert** for the ML pipeline's news-sentiment features
(`DataExtractor.news_sentiment_history`). The model loads lazily, once, under a lock; the web
app no longer loads it.

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

### Triggering via the API (local)

The ML Signals tab's **Run Pipeline** button calls these endpoints, which run the orchestrator as a background thread in the Flask process. Use them locally — on Render the trigger returns 503 while `PIPELINE_TRIGGER_TOKEN` is unset (the intended production state; with a token set, callers must send it as `X-Pipeline-Token`).

```bash
# Start a job
curl -X POST http://localhost:5001/api/pipeline/run \
  -H "Content-Type: application/json" -d '{"ticker":"NVDA"}'
# → {"success": true, "job_id": "a1b2c3d4", "status": "queued"}

# Poll it
curl http://localhost:5001/api/pipeline/status/a1b2c3d4
# → {"status": "running", "current_step": "supervised", "last_logs": [...]}
```

`status` ∈ `queued` → `running` → `done` | `up_to_date` | `error`.

> **Job store:** retrain-job state lives in a small SQLite store (`backend/pipeline_store.py`, WAL mode) — consistent across Gunicorn workers, with bounded per-job logs and automatic eviction of old jobs. It survives worker recycles as long as the DB file survives (`PIPELINE_DB_PATH` overrides its location). The trigger endpoint is single-flight (returns 409 if a job is already running).

### Daily schedule (`.github/workflows/daily-retrain.yml`)

A GitHub Actions workflow (cron `0 2 * * 1-5` — 02:00 UTC weekdays, plus manual dispatch with a ticker input) runs `orchestrator.py` **on the Actions runner**. Each run is a full rebuild; only the model registry and `mlflow.db` carry over, restored from the `models` branch so the promotion gate compares against the models being served. Afterwards it force-pushes a single-commit `models` branch containing `supervised/model_registry/**`, `data_pipelines/<TICKER>_features_with_regimes.csv`, `mlflow.db`, the model's paper ledger (`paper/`) and the backtest summary (`backtesting/output/<TICKER>_backtest_summary.json`). The run's green/red badge is your daily retraining health signal; a failure email means a pipeline step failed on the runner — read that step's log in the Actions tab.

**Serving the results:** on Render, `backend/model_sync.py` downloads the `models` branch tarball at most every 6 h (triggered by the prediction, drift, model-status and MLflow-runs endpoints) and unpacks only those paths over the copies baked into the image. Local runs never sync.

**Why not on Render?** The free 512 MB instance ran out of memory running the pipeline in-process.

**Setup:** data-provider keys come from repo secrets under Settings → Secrets and variables → Actions — `FINNHUB_API_KEY`, `ALPHAVANTAGE_API_KEY`, `POLYGON_API_KEY`, `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `NVIDIA_API_KEY`. A missing key just skips that source.

### Three scheduling paths exist — only one is live

The repo contains three mechanisms that all run the same retrain. Know which is real:

| Path | When | Where | Status |
|---|---|---|---|
| `.github/workflows/daily-retrain.yml` | 02:00 UTC Mon–Fri | GitHub Actions runner → `models` branch | **✅ Active — this is production** |
| `k8s/cronjob.yaml` | 02:00 UTC Mon–Fri (same) | Kubernetes (runs `Dockerfile.pipeline`) | ⚠️ Scaffolding — unused (see [§2](#2-deployment)) |
| `scripts/daily_predict.sh` | 22:00 UTC / 5 PM ET | A local dev machine (hardcoded paths) | 🔧 Local convenience only |

They don't run concurrently (different platforms), but only the GitHub Action
publishes to the `models` branch the live app serves from. If you ever migrate to
K8s, the CronJob becomes the real scheduler.

### MLflow tracking (`mlflow.db`, `mlruns/`)

SQLite backend at the repo root, one experiment per ticker, runs named `<target>_v<version>_<date>` (e.g. `target_5d_vB_2026-05-30`). Browse locally with `mlflow ui --backend-store-uri sqlite:///mlflow.db`. Production history is the `mlflow.db` on the `models` branch, served by `GET /api/mlflow/runs/<ticker>`.

### Model registry (`algorithms/machine_learning_algorithms/supervised/model_registry/`)

Per ticker: an `active.json` index plus one directory per target holding `model.pkl`, scalers, `features.json`, `train_stats.json` (drift baselines), `metadata.json`, and a `prev/` snapshot for rollback.

**Promotion gate** (`registry.py`) — candidates are first limited to models whose edge held up across the walk-forward CV folds (`cv_stability`: mean CV metric ≥ the floor below *and* beat that fold's naive baseline in ≥ 50% of folds). If none qualifies the target isn't served — one good holdout year isn't enough. The production model gets no protection when its model type's CV record in this run is unstable, so it can't keep its seat on one holdout year either. The best qualifying model then replaces production only if it clears **all three** (unless forced):
1. Absolute floor — regression `ic > 0.02`, classification `f1_w > 0.30`
2. Beats the naive baseline by ≥ 5%
3. Improves over the current production model by ≥ 0.005

The outgoing model is copied to `prev/` first, so `rollback_registry()` can restore it.

### Serving & drift monitoring (`backend/predictor.py`)

`predict_latest()` serves predictions from the registry; the module also computes **data drift** (feature z-score > 3.0) and **model drift** (rolling IC falling > 0.10 below the registered IC), writing reports to `supervised/output/monitoring/` and POSTing alerts to `ALERT_WEBHOOK_URL` (Slack-style webhook) when thresholds trip.

Served over HTTP by `backend/routes/ml.py`: `GET /api/predict/<ticker>`, `/api/drift/<ticker>`, `/api/model/status`, and `/api/mlflow/runs/<ticker>`.

### LLM layer (`ai_platform/`)

- `llm_router.py` — unified multi-provider client (OpenAI → Anthropic → NVIDIA priority; override with `LLM_PROVIDER`). The NVIDIA chain is Nemotron 3.5 Super then Nemotron 3.5 Lightning, thinking off (`NVIDIA_MODELS` overrides)
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

`backtesting/` — walk-forward backtest of the supervised models' trading signal:

```bash
python backtesting/backtest.py NVDA --model all     # or --model lgb,ridge; default = the served target_5d model
```

- Retrains every `--retrain-every` rows (default 21) from `INIT_TRAIN`, only on rows
  whose `target_5d` label was already known, with the pipeline's own fold
  preprocessing and model ladder
- Trades the live rule (long / short / cash at ±1% predicted 5d return) with the
  model paper account's fills: decide at the close, fill at the next open, 5 bps slippage
- Also runs rule-based factor strategies (`strategies.py`, textbook parameters fixed in
  advance): 200-day trend, 12-month time-series momentum, RSI(14) mean reversion
- Compares against buy-and-hold and a naive 5-day momentum rule: return, CAGR,
  Sharpe, max drawdown, trades, long/short time, out-of-sample forecast IC
- Writes `backtesting/output/<TICKER>_backtest_{daily.csv,summary.json}` (gitignored); the daily
  retrain runs it for every model and publishes the summary (with equity curves) to the
  `models` branch → `GET /api/ml/backtest/<TICKER>` → the Paper tab's backtest table
- The feature allow-list was chosen on the full history, so model results are
  slightly optimistic; `simulator.py` is dependency-light and unit-tested (`tests/test_backtest.py`)

**Cross-sectional factors on a universe** — rank many stocks at once instead of timing one:

```bash
U=sp500   # or large50
python algorithms/machine_learning_algorithms/data_pipelines/universe.py $U          # sp500 ~2.5 min, large50 ~10 s
python algorithms/machine_learning_algorithms/data_pipelines/fundamentals_pit.py $U  # sp500 ~1.5 h cold (rate limits), cached 24 h
python backtesting/factor_backtest.py $U
python backtesting/pooled_backtest.py $U                                              # sp500 ~8 min
```
(`make factors UNIVERSE=sp500` runs all four.)

- **Universes** (`universe.py` → `UNIVERSE_<name>_panel.csv.gz`, gitignored): adjusted OHLCV, the
  pipeline's price-only features and the six targets, computed within each ticker, plus
  `in_index` — the dates the company belonged to the universe; everything below ranks and holds
  only those
  - `sp500` — **point-in-time S&P 500**: every company that was a member at any time in the window
    (`membership.py`, from github.com/fja05680/sp500, MIT), flagged only while it was one; renamed
    tickers mapped (FB→META, ANTM→ELV, …). yfinance doesn't price most acquired or failed
    companies (Activision, Xilinx, SVB, First Republic, …): 54 of 626, **4.9% of member-days**,
    listed in `UNIVERSE_sp500_coverage.json`. A holding whose prices end is cashed out at its last
    price. Swap in a source with delisted prices to close the gap
  - `large50` — a fixed list of 50 large caps picked in 2026. Survivorship-biased: over the same
    window its equal-weight return was +15.3%/yr against +8.9% for the point-in-time S&P 500
- `fundamentals_pit.py` adds **point-in-time fundamentals** from SEC filings (Finnhub "financials
  as reported", with each filing's date) and EPS surprises (yfinance): gross profitability, ROA,
  ROE, accruals, asset and revenue growth, leverage, earnings yield, book-to-market, sales/price,
  latest surprise. Each number is usable from the business day after its filing, as first
  reported; 10-Q year-to-date figures become trailing twelve months only from earlier filings;
  market caps convert filed share counts with the splits after the filing
- `factor_backtest.py` holds the top fifth by a factor — 12-month momentum, low volatility,
  1-week reversal, gross profitability, earnings yield, book-to-market, low accruals, low asset
  growth, post-earnings drift (textbook settings) — re-ranked monthly (weekly for reversal),
  traded at the next open with 5 bps on the traded value, holdings drifting between rebalances
  (`cross_section.py`). Benchmark: the equal-weight universe. Reports each factor's rank IC and
  its t-statistic on non-overlapping dates
- `pooled_backtest.py` trains **one model across the universe** (~700k member rows for sp500) to
  rank stocks by next-month return: target and the 22 price features ranked across members per
  date, refit monthly on labels known at the refit (21-session purge), top fifth held, plus a
  `+fund` variant with the fundamentals. Tests confirm it finds a planted predictive feature and
  nothing in noise

Results on the point-in-time S&P 500 (single factors 2021-11 → 2026-11, equal weight +8.9%/yr;
pooled 2022-11 → 2026-11, equal weight +13.5%/yr; excess CAGR, IC t-stat):

| | Excess / yr | IC t |
|---|---|---|
| Book-to-market | +4.1% | 0.87 |
| Earnings yield | +3.7% | 1.61 |
| 12-month momentum | +2.8% | 0.75 |
| Low accruals | +2.1% | 0.32 |
| Low asset growth | −0.1% | 0.35 |
| Gross profitability | −2.2% | −0.44 |
| Pooled LightGBM (price features) | +3.1% | 0.76 |
| Pooled Ridge / either + fundamentals | −0.5% to −0.1% | ≤ 0.47 |

Value leads, nothing clears t ≈ 2, and the models don't beat plain momentum (+3.8% on the pooled
window). The factors that led on `large50` (low asset growth, low accruals) vanish on the full
index: on 50 hand-picked stocks they were noise.

Options-volatility backtests are separate: `/api/backtest` (POST) and
`/api/backtest/strategies` (GET) run `backend/backtest_runner.py` on the engine below.

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

`/api/calendar?symbols=NVDA,TD,...` merges four sources over the next year:

- **Earnings** — Finnhub's full US earnings calendar (next 120 days, swept in
  weekly windows, cached 30 min per window), kept for the user's watchlist
  (sent from the browser) and the 50 "top companies" (`TOP_COMPANIES` in
  `backend/common.py`, = the research `large50` universe). Names Finnhub lacks
  (OTC / TSX-only listings) fall back to Yahoo's next earnings date.
- **Dividends** — Yahoo ex-dividend dates for watchlist symbols.
- **Market holidays** — generated from NYSE rules in `backend/market_calendar.py`, so they never run out.
- **FOMC / CPI / PPI / jobs / GDP** — `backend/economic_events.json`, from the
  Fed, BLS and BEA schedules (sources and valid-through date in the file).
  **Refresh it each autumn** when the next year's schedules are published.

The Calendar tab tags each earnings event *Watchlist* or *Top company* and can filter by either.

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
│   ├── app.py                              Flask entrypoint (~58 lines; registers blueprints)
│   ├── routes/                             ALL API routes, one blueprint per domain:
│   │                                         market, stock, charts, news, trading,
│   │                                         backtest, research, pipeline, chat, ml
│   ├── services.py                         guarded data-layer imports + TTL caches
│   ├── common.py                           shared config (indices, tickers) + helpers
│   ├── predictor.py                        prediction serving + drift monitoring
│   ├── model_sync.py                       pulls the `models` branch on Render
│   ├── economic_events.json                FOMC/BLS/BEA dates (refresh yearly)
│   ├── templates/index.html                SPA (Bootstrap 5, Chart.js)
│   ├── static/css|js|research/
│   ├── pipeline_store.py                   SQLite retrain-job store (bounded, shared)
│   └── entrypoint.sh                       Gunicorn startup (GUNICORN_WORKERS, default 1)
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
├── backtesting/                          Walk-forward backtest of the ML signal
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

**📦 Placeholder / empty (present in the tree, no implementation)**
- All of top-level `models/` (interest-rate, execution, allocation, credit, options)
- `deep_learning/`, `monte_carlo/`, `data/streaming/`, `backtesting/stress_testing/` (empty files)
- Several `prototype.py` stubs under `execution_algorithms/`, `market_making_algorithms/`,
  and `volatility_forecasting/research/` (theory `.tex` only)

---

## 20. Roadmap

### In Progress
- [ ] Deep learning (LSTM, Transformer) in `deep_learning/`
- [ ] Factor strategies — time-series trend / momentum / mean reversion (`backtesting/strategies.py`) and cross-sectional momentum / low-vol / reversal on a 50-stock universe (`backtesting/factor_backtest.py`) done; quality needs point-in-time fundamentals
- [x] Pooled ML model across the universe (`backtesting/pooled_backtest.py`) — no consistent edge from price-only features or point-in-time fundamentals on 50 large caps
- [x] Point-in-time fundamentals + EPS surprises (`data_pipelines/fundamentals_pit.py`)
- [x] Point-in-time S&P 500 universe (`data_pipelines/membership.py`) — 95% of member-days priced
- [ ] Prices for the acquired/failed members yfinance lacks (the last 4.9%), from a source with delisted tickers

### Completed ✅
- [x] **Walk-forward backtesting harness** — `backtesting/backtest.py` trades the model signal on the feature matrix vs buy-and-hold
- [x] **Scheduled daily retraining** — orchestrator in GitHub Actions, published to the `models` branch
- [x] **Prediction & drift endpoints** — `predictor.py` served via `backend/routes/ml.py`
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
- [x] Render + Docker deployment with health checks

---

## 📄 License

MIT License — see LICENSE file

## 📧 Contact

**Aarya Shah** — [@aarya127](https://github.com/aarya127)
