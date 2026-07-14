# Research & Analysis Scripts Reference

Standalone scripts under `algorithms/machine_learning_algorithms/`. **None of these
are called by `orchestrator.py`** — they are run by hand for research and diagnostics.
See [README §9](../README.md#9-scheduled-retraining--mlops) for the automated pipeline.

**Two data-sourcing patterns — important:**
- `time_series_models/*` fetch **live from yfinance** (3-year history) at runtime. They do *not* read the pipeline CSVs.
- `eda/eda.py` reads a **pipeline CSV** (`data_pipelines/<SYM>_features_clean.csv`) and needs the pipeline to have run first.

All `time_series_models` scripts share conventions: `python <file>.py [TICKER]`
(defaults to `NVDA` via `sys.argv[1]`), matplotlib `Agg` (headless), and write to
`time_series_models/output/<name>/`. Common deps: `statsmodels`, `scipy`,
`yfinance`, `matplotlib`, `pandas`, `numpy`. GARCH/ARCH additionally need the pip
**`arch`** package.

---

## time_series_models/

### arima.py — ARIMA price forecasting + white-noise demonstration
Auto ARIMA(p,d,q) order selection (AIC grid), Ljung-Box residual check, OOS forecast with CIs. A second `ARIMAXSuite` runs an expanding-window walk-forward comparing a naive baseline, ARMA-on-returns, and ARIMAX (VIX; VIX+SPY) to show returns are ≈white noise (motivating the ML pipeline).
- Data: `ARIMAModel.run` → 2y Close; the suite → `[ticker, SPY, ^VIX]` at 3y.
- Key API: `ARIMAModel(order=None, max_p=4, max_q=2)`, `.auto_select_order(series, d=1)`, `.fit_and_forecast(series, steps=5, d=1, confidence=0.95) -> ARIMAForecast`, `.run(ticker, period='2y', forecast_days=5)`; `run_arima_suite(ticker='NVDA')`. Dataclasses `ARIMAOrder`, `ARIMAForecast`.
- Libraries: `statsmodels ARIMA`, `acorr_ljungbox(lags=[10])`, `spearmanr` (IC). Falls back to a numpy AR(1) if statsmodels is absent.
- Out (`output/arima/`): `{T}_arima_comparison.csv`, `{T}_arima_comparison.png`, `{T}_arima_forecast.png`.

### garch.py — GARCH(1,1) volatility (hand-rolled MLE + `arch` suite)
A from-scratch GARCH(1,1) fit by Gaussian MLE (scipy `minimize`, L-BFGS-B), giving conditional vol, persistence, half-life, long-run annual vol, 1-day 95% VaR/CVaR, AIC/BIC. Then `GARCHSuite` uses the `arch` package to compare GARCH(1,1), GARCH(1,1)-t, GJR-GARCH, EGARCH with 1d/5d forecasts vs a 20-day realized-vol baseline.
- Data: 3y Close; log returns × 100.
- Key API: `GARCHModel().fit(series) -> GARCHResult`, `.run(ticker, period='3y')`; `fit_suite(returns) -> (DataFrame, dict)`; `run_suite(ticker='NVDA')`; plot helpers. Dataclasses `GARCHParams`, `GARCHResult`.
- Constraints: α∈(1e-6,0.5), β∈(1e-6,0.9999), α+β<1. Half-life = ln0.5/ln(persistence); long-run vol = √(ω/(1−α−β))·√252.
- Note: manipulates `sys.path` to import the installed `arch` package around the sibling `arch.py` name collision.
- Out (`output/garch/`): `{T}_garch_model_comparison.csv` + `_conditional_vol/_model_comparison/_forecast/_vs_realized.png`.

### arch.py — ARCH(q) benchmark
Fits ARCH(1/2/5/10) and GARCH(1,1) to show GARCH dominates on AIC/BIC (motivates `garch.py`). Same `sys.path` `arch`-import workaround.
- Key API: `load_returns(ticker, period='3y')`, `fit_models(returns) -> DataFrame`, `get_conditional_vol(returns, model_spec, q=1)`, `run(ticker='NVDA')`.
- Out (`output/arch/`): `{T}_arch_aic_bic.csv`, `{T}_arch_comparison.png`, `{T}_arch_vol_path.png`.

### var.py — Vector Autoregression
VAR(p) over a **fixed** set `NVDA, QQQ, SPY, SOXX, ^VIX` (the `ticker` arg only labels output). Lag selection by BIC (capped 1–5), Granger causality toward NVDA, IRF, FEVD, 5-day forecast vs naive.
- Key API: `load_var_data(period='3y')`, `select_lag(df, max_lags=10)`, `fit_var(df, lag)`, `granger_causality(result, df)`, `plot_irf/plot_fevd(result, ticker, periods=10)`, `plot_var_forecast(result, df, ticker, horizon=5)`, `run(ticker='NVDA')`.
- Out (`output/var/`): `var_lag_selection.csv`, `var_granger.csv` (not ticker-prefixed), `{T}_var_irf/_fevd/_forecast.png`.

### cointegrations.py — pairs trading
Base ticker vs semiconductor peers `AMD, SMH, SOXX, QQQ, AVGO, TSM`. Engle-Granger + Johansen tests, AR(1) spread half-life, Hurst exponent, rolling z-score signals (entry |z|>1.5, exit |z|<0.5, window 60). "Tradeable" = cointegrated ∧ half-life<60d ∧ Hurst<0.5. Analysis on log prices.
- Key API: `load_prices(tickers, period='3y')`, `engle_granger(y, x) -> dict`, `half_life(spread)`, `hurst_exponent(series, max_lag=20)`, `johansen_test(y, x) -> dict`, `compute_zscore(spread, window=60)`, `run(base='NVDA')`.
- Out (`output/cointegrations/`): `cointegration_pairs.csv`, `johansen_results.csv`, `hurst.csv`, `{base}_{peer}_spread.png`.

### volatility_clustering.py — clustering & regime tests
Tests for volatility clustering and whether it's stronger under stress. ACF of |returns|/squared returns, calm-vs-stress comparison (split at median 20d realized vol), Ljung-Box, and Engle's ARCH-LM test.
- Key API: `load_data(ticker, period='3y')`, `plot_returns_overview/plot_acf(df, ticker, nlags=40)/plot_regime_clustering`, `ljung_box_test(df, ticker) -> DataFrame`, `arch_lm_test(df, ticker, lags=12) -> dict`, `run(ticker='NVDA')`.
- Out (`output/volatility_clustering/`): `{T}_returns_overview/_acf/_regime_clustering.png`, `{T}_ljung_box.csv`, `{T}_arch_lm.csv`.

---

## eda/eda.py — feature-matrix EDA report

A 12-section EDA over the **pre-built** feature matrix (target distribution + Shapiro/ADF, price/volume, return ACF/PACF, volatility regimes, RSI/MACD/BB signals, momentum, news sentiment, macro/cross-asset, target correlation + heatmap, per-feature distributions).

- Run: `python eda.py [SYMBOL]` (default `NVDA`). **No `__main__` guard** — it executes top-to-bottom on import.
- Input: `data_pipelines/{SYMBOL}_features_clean.csv` (exits with a message if missing — "run run_pipeline.py then clean.py first"). Adds `target_1d = log_return.shift(-1)`.
- Libraries: `scipy.stats` (`shapiro`, `probplot`), `seaborn` heatmap, `matplotlib` Agg; `statsmodels` ADF + ACF/PACF are **optional** (sections skip gracefully if it's absent).
- Only helper: `save(fig, name)`.
- Output: PNGs land in the **`eda/` directory itself** (`OUT_DIR = HERE`) — `02_target_return.png` … `12_feature_distributions.png`. (The module docstring says `eda_output/`, but the code writes to `eda/`.)
