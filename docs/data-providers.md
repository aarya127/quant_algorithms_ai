# Data Provider Reference (`data/`)

Per-module reference for the market-data layer. Six of the seven modules are
imported by `backend/app.py` (inside the guarded import block at ~lines 90–114);
`prices.py` is standalone. See [README §5](../README.md#5-data-pipeline) for the
higher-level extraction/transform layer that sits on top of these.

## Auth model

Two patterns coexist — know which a module uses:

- **env-only**: `alpaca_news.py`, `twitter_feed.py` (read env vars at import; default to empty string, fail gracefully at call time).
- **env → `keys.txt` fallback**: `finnhub.py`, `alphavantage.py`. If the env var is unset they scan `keys.txt` for a line containing a header substring and use the **next line** as the key.

`keys.txt` header substrings (case-insensitive):

| Module | Env var | `keys.txt` match | Notes |
|---|---|---|---|
| `finnhub.py` | `FINNHUB_API_KEY` | `finnhub` **or** `finhub` (misspelling tolerated) | searches `keys.txt`, `../keys.txt`, `<dir>/../keys.txt` |
| `alphavantage.py` | `ALPHAVANTAGE_API_KEY` | `alphavantage` | searches `../keys.txt` |

`charts.py`, `prices.py`, `company_statistics.py` need no direct auth (yfinance, or they inherit from finnhub/alphavantage).

---

## finnhub.py — WIRED

Thin wrapper over the official `finnhub` SDK using a **module-level singleton client** (`_client`, created at import, reads key once via `_load_api_key()`).

| Function | Signature |
|---|---|
| `get_company_profile` | `(symbol)` |
| `get_stock_quote` | `(symbol)` |
| `get_company_news` | `(symbol, from_date, to_date)` |
| `get_market_news` | `(category='general')` |
| `get_basic_financials` | `(symbol, metric='all')` |
| `get_insider_transactions` | `(symbol, from_date=None, to_date=None)` |
| `get_insider_sentiment` | `(symbol, from_date, to_date)` |
| `get_earnings_surprises` | `(symbol)` |
| `get_earnings_calendar` | `(from_date, to_date, symbol=None)` |
| `get_usa_spending` | `(symbol, from_date, to_date)` |

Returns SDK dicts / lists of dicts. Rate limit 60 req/min (not enforced in-module). No threads/caches.

## alphavantage.py — WIRED (`AlphaVantage` class)

REST client over `https://www.alphavantage.co/query` via `requests`. One `requests.get` per call, no caching, no sleeps (the 25/day free limit is on you). Most methods return raw parsed JSON `dict`.

Constructor: `AlphaVantage(api_key=None)`. Key groups:
- **News / fundamentals**: `get_market_news_sentiment(tickers=None, topics=None, time_from=None, time_to=None, sort='LATEST', limit=50)`, `get_company_overview(symbol)`, `get_global_quote(symbol)` (returns the `Global Quote` sub-dict), `get_income_statement/get_balance_sheet/get_cash_flow(symbol)`, `get_earnings_history/get_earnings_estimates(symbol)`, `get_earnings_calendar(symbol=None, horizon='3month')`, `get_insider_transactions(symbol)`, `get_shares_outstanding(symbol)`, `get_top_gainers_losers()`, `get_earnings_call_transcript(symbol)`.
- **Macro**: `get_wti_crude_oil/get_brent_crude_oil/get_natural_gas(interval='monthly')`, `get_real_gdp(interval='annual')`, `get_treasury_yield(interval='monthly', maturity='10year')`, `get_federal_funds_rate(interval='monthly')`, `get_inflation(interval='monthly')`.
- **Technicals**: `get_sma/get_ema(symbol, interval='daily', time_period=20, series_type='close')`, `get_rsi(symbol, interval='daily', time_period=14, series_type='close')`, `get_adx(symbol, interval='daily', time_period=14)`.

## charts.py — WIRED

yfinance OHLCV shaped for Chart.js. All returns are `dict` with a `success: bool`; failures return `{success: False, error: str}`. No auth, no persistent state. `get_technical_indicators` retries with longer periods (`3mo→6mo→1y→2y`) if it gets <50 points.

| Function | Signature |
|---|---|
| `get_chart_data` | `(symbol, period="1y", interval="1d")` |
| `get_multiple_timeframes` | `(symbol)` |
| `get_comparison_data` | `(symbols, period="1y", interval="1d")` |
| `get_technical_indicators` | `(symbol, period="1y", interval="1d")` → nested `price` + `indicators` (sma_20/50/200, ema_12/26, rsi, macd{macd,signal,histogram}, bollinger{upper,middle,lower}) |

Indicator helpers (not exported to app): `calculate_sma/ema/rsi/macd/bollinger_bands`.

## company_statistics.py — WIRED

Aggregates a full company profile from **yfinance (`.info`) + Finnhub** (`get_basic_financials`); an Alpha Vantage path is imported but commented out. The only data module that imports other data modules.

- `get_company_statistics(symbol) -> dict` — nested keys: `profile, margins, returns, valuation_ttm, valuation_forward, financial_health, growth, dividends`.
- `format_statistics_for_display(stats) -> dict` — same shape, human-formatted strings (`$B/M/K`, `%`, `—` for missing).

## twitter_feed.py — WIRED

Twitter/X API v2 via `tweepy.Client` (`wait_on_rate_limit=True`). Env: `TWITTER_API_KEY`, `TWITTER_API_SECRET`, `TWITTER_BEARER_TOKEN`. `get_financial_news_feed` caps to the first 5 accounts to limit calls.

- `get_market_tweets(symbol=None, count=20) -> List[Dict]`
- `get_financial_news_feed(count=30) -> List[Dict]`
- `get_user_timeline(username, count=10) -> List[Dict]`, `get_twitter_client()`
- Module lists: `INFLUENTIAL_ACCOUNTS`, `FINANCIAL_ACCOUNTS`.

Tweet dict: `id, text, created_at, author{username,name,verified,profile_image}, metrics{likes,retweets,replies}, url, source`.

## alpaca_news.py — WIRED (real-time, stateful)

Alpaca news WebSocket (`websockets`, async). **Has real side effects**: module-level `recent_news = deque(maxlen=100)` guarded by `news_lock`, singletons `_news_stream`/`_stream_thread`, and `start_news_stream` spawns a **daemon thread** running its own asyncio loop. `app.py` starts the stream and polls `get_recent_news`.

- `start_news_stream(symbols=None, use_sandbox=False)`
- `stop_news_stream()`
- `get_recent_news(count=20, symbol=None) -> List[Dict]`
- `async fetch_news_snapshot(symbols=None, timeout=5) -> List[Dict]`
- `class AlpacaNewsStream` — `connect()`, `subscribe()`, `unsubscribe()`, `listen(callback)`, `close()`.

News dict: `id, headline, summary, author, created_at, updated_at, url, content, symbols, source, type`.

## prices.py — STANDALONE (not imported by app.py)

Simple yfinance price utilities, no auth, CLI/notebook use.

- `get_historical_prices(ticker, period="1mo", interval="1d") -> pd.DataFrame`
- `get_latest_price(ticker) -> Optional[float]` (falls back to 5d history if `fast_info` missing)
- `get_multiple_latest(tickers) -> pd.Series`
- `print_summary(tickers)`
