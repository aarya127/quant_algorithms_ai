"""
ARIMA Price Forecasting Model

Auto-Regressive Integrated Moving Average (ARIMA) for financial time series.

Model:  ARIMA(p, d, q)
  p — AR order: uses past p prices / returns
  d — differencing order (1 = log-return differencing for stationarity)
  q — MA order: uses past q forecast errors

Order selection via AIC / BIC grid-search over (p, d, q) combinations.
Includes Ljung-Box residual test: p-value > 0.05 → white-noise residuals (good).

Requires: statsmodels  (pip install statsmodels)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from itertools import product
from typing import List
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ARIMAOrder:
    p: int
    d: int
    q: int

    def __str__(self) -> str:
        return f"ARIMA({self.p},{self.d},{self.q})"


@dataclass
class ARIMAForecast:
    order: ARIMAOrder
    aic: float
    bic: float
    in_sample_rmse: float
    ljung_box_p_value: float          # > 0.05 = white-noise residuals
    forecast: List[float] = field(default_factory=list)
    confidence_lower: List[float] = field(default_factory=list)
    confidence_upper: List[float] = field(default_factory=list)


class ARIMAModel:
    """
    ARIMA forecaster with automatic order selection.

    Parameters
    ----------
    order     : ARIMAOrder, optional — if None, auto-selects via AIC grid search.
    max_p     : int — maximum AR order to consider during grid search.
    max_q     : int — maximum MA order to consider during grid search.
    """

    def __init__(self, order: ARIMAOrder = None, max_p: int = 4, max_q: int = 2):
        self.order = order
        self.max_p = max_p
        self.max_q = max_q

    # ------------------------------------------------------------------
    # Order selection
    # ------------------------------------------------------------------

    def auto_select_order(self, series: pd.Series, d: int = 1) -> ARIMAOrder:
        """Grid-search (p, q) over [0..max_p] x [0..max_q], minimising AIC."""
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError:
            return ARIMAOrder(p=1, d=d, q=0)

        best_aic = np.inf
        best_order = ARIMAOrder(1, d, 0)

        for p, q in product(range(self.max_p + 1), range(self.max_q + 1)):
            if p == 0 and q == 0:
                continue
            try:
                res = ARIMA(series, order=(p, d, q)).fit()
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_order = ARIMAOrder(p=p, d=d, q=q)
            except Exception:
                pass

        return best_order

    # ------------------------------------------------------------------
    # Fit & forecast
    # ------------------------------------------------------------------

    def fit_and_forecast(
        self,
        series: pd.Series,
        steps: int = 5,
        d: int = 1,
        confidence: float = 0.95,
    ) -> ARIMAForecast:
        """Fit ARIMA and return out-of-sample forecast with confidence intervals."""
        order = self.order or self.auto_select_order(series, d=d)

        try:
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.stats.diagnostic import acorr_ljungbox

            model  = ARIMA(series, order=(order.p, order.d, order.q))
            result = model.fit()

            resid = result.resid.dropna()
            rmse  = float(np.sqrt((resid ** 2).mean()))

            lb    = acorr_ljungbox(resid, lags=[10], return_df=True)
            lb_p  = float(lb['lb_pvalue'].iloc[0])

            fc    = result.get_forecast(steps=steps)
            ci    = fc.conf_int(alpha=1 - confidence)

            return ARIMAForecast(
                order=order,
                aic=float(result.aic),
                bic=float(result.bic),
                in_sample_rmse=rmse,
                ljung_box_p_value=lb_p,
                forecast=fc.predicted_mean.tolist(),
                confidence_lower=ci.iloc[:, 0].tolist(),
                confidence_upper=ci.iloc[:, 1].tolist(),
            )

        except ImportError:
            # Fallback: simple differenced AR(1)
            diffs = series.diff(order.d).dropna().values
            phi   = np.cov(diffs[1:], diffs[:-1])[0, 1] / np.var(diffs[:-1])
            sigma = float(np.std(diffs - phi * np.roll(diffs, 1)))
            last  = diffs[-1]
            fc_vals = [float(last * phi ** i) for i in range(1, steps + 1)]
            margin  = 1.96 * sigma
            return ARIMAForecast(
                order=order, aic=np.nan, bic=np.nan,
                in_sample_rmse=sigma, ljung_box_p_value=np.nan,
                forecast=fc_vals,
                confidence_lower=[v - margin for v in fc_vals],
                confidence_upper=[v + margin for v in fc_vals],
            )

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------

    def run(self, ticker: str, period: str = '2y', forecast_days: int = 5) -> None:
        import yfinance as yf
        data  = yf.download(ticker, period=period, progress=False)
        close = data['Close'].squeeze()

        result = self.fit_and_forecast(close, steps=forecast_days)
        last_price = float(close.iloc[-1])

        print(f"\n{ticker}  {result.order}  [{period}]")
        print(f"  AIC: {result.aic:.1f}   BIC: {result.bic:.1f}")
        print(f"  In-sample RMSE: {result.in_sample_rmse:.4f}")
        lb = result.ljung_box_p_value
        status = 'white noise ✓' if (not np.isnan(lb) and lb > 0.05) else 'autocorrelation ✗'
        print(f"  Ljung-Box p-value: {lb:.4f}  ({status})")
        print(f"\n  {forecast_days}-day forecast  (last price ${last_price:.2f}):")
        for i, (fc, lo, hi) in enumerate(
            zip(result.forecast, result.confidence_lower, result.confidence_upper), 1
        ):
            print(f"    Day {i}: ${fc:.2f}  [{lo:.2f}, {hi:.2f}]")


# ─────────────────────────────────────────────────────────────────────────────
# ARIMAXSuite — ARMA / ARIMAX vs naive baselines, rolling walk-forward backtest
# ─────────────────────────────────────────────────────────────────────────────
"""
ARIMAXSuite compares:
  1.  Zero-return naive baseline  (always predict μ of training window)
  2.  ARMA(p,q) on returns        (order selected by AIC)
  3.  ARIMAX with VIX change      (one exogenous macro regressor)
  4.  ARIMAX with VIX + SPY       (two exogenous regressors)

Walk-forward evaluation:
  - Train on expanding window; predict 1-step ahead
  - Metrics: RMSE, MAE, directional accuracy, IC (rank correlation)

Key insight from the guide:
  Stock returns are close to white noise, so ARIMA may barely beat the naive
  zero-return baseline.  That is useful information: it is the reason ML with
  richer features is needed.

Outputs (output/arima/):
  {TICKER}_arima_comparison.csv      — walk-forward metrics per model
  {TICKER}_arima_comparison.png      — RMSE / dir_acc bar chart
  {TICKER}_arima_forecast.png        — 5-day price forecast with CI

Usage: python arima.py [TICKER]   (default: NVDA)
"""

import sys as _sys
import warnings as _warnings
_warnings.filterwarnings('ignore')

import numpy as _np
import pandas as _pd
import matplotlib as _mpl
_mpl.use('Agg')
import matplotlib.pyplot as _plt
from pathlib import Path as _Path
from itertools import product as _product

_ARIMA_OUT = _Path(__file__).parent / 'output' / 'arima'
_ARIMA_OUT.mkdir(parents=True, exist_ok=True)


def _arima_save(fig, name: str) -> None:
    path = _ARIMA_OUT / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    _plt.close(fig)
    print(f'  ✓ {path.name}')


def _load_price_and_exog(ticker: str, period: str = '3y') -> _pd.DataFrame:
    """Download price + VIX + SPY; align to common index."""
    import yfinance as yf
    tickers = [ticker, 'SPY', '^VIX']
    raw = {}
    for t in tickers:
        d = yf.download(t, period=period, progress=False, auto_adjust=True)
        raw[t] = d['Close'].squeeze()

    df = _pd.DataFrame(raw).dropna()
    df.columns = [ticker, 'SPY', 'VIX']

    # Log returns
    df['ret']     = _np.log(df[ticker] / df[ticker].shift(1))
    df['ret_spy'] = _np.log(df['SPY']  / df['SPY'].shift(1))
    df['d_vix']   = df['VIX'].diff()        # VIX change (level is I(1))
    return df.dropna()


def _best_arma_order(returns: _pd.Series, max_p: int = 3, max_q: int = 2) -> tuple:
    """AIC grid search over ARMA(p,q) orders. Returns (p,q)."""
    from statsmodels.tsa.arima.model import ARIMA as _ARIMA_sm
    best_aic, best_order = _np.inf, (1, 0)
    for p, q in _product(range(max_p + 1), range(max_q + 1)):
        if p == 0 and q == 0:
            continue
        try:
            res = _ARIMA_sm(returns, order=(p, 0, q)).fit()
            if res.aic < best_aic:
                best_aic, best_order = res.aic, (p, q)
        except Exception:
            pass
    return best_order


def _walk_forward_compare(df: _pd.DataFrame, ticker: str,
                           init_train: int = 120) -> _pd.DataFrame:
    """
    Expanding-window 1-step-ahead comparison across four models.
    Order is selected once on the initial training window to keep runtime fast.
    Returns DataFrame of metrics per model.
    """
    from statsmodels.tsa.arima.model import ARIMA as _ARIMA_sm
    from scipy.stats import spearmanr

    rets    = df['ret'].values
    d_vix   = df['d_vix'].values
    ret_spy = df['ret_spy'].values
    n       = len(rets)

    # Select ARMA order once on the initial window (avoid re-fitting grid at every step)
    init_p, init_q = _best_arma_order(_pd.Series(rets[:init_train]))

    preds   = {m: [] for m in ['naive', 'arma', 'arimax_vix', 'arimax_vix_spy']}
    actuals = []

    for t in range(init_train, n - 1):
        y_train = rets[:t]
        actual  = rets[t]
        actuals.append(actual)

        # 1. Naive: mean of training returns
        preds['naive'].append(float(_np.mean(y_train)))

        # 2. ARMA (fixed order)
        try:
            res = _ARIMA_sm(y_train, order=(init_p, 0, init_q)).fit()
            fc  = res.forecast(1)
            preds['arma'].append(float(fc.iloc[0]))
        except Exception:
            preds['arma'].append(float(_np.mean(y_train)))

        # 3. ARIMAX with VIX change
        try:
            exog_tr  = d_vix[:t].reshape(-1, 1)
            exog_fc  = d_vix[t].reshape(1, 1)
            res      = _ARIMA_sm(y_train, order=(init_p, 0, init_q),
                                 exog=exog_tr).fit()
            fc       = res.forecast(1, exog=exog_fc)
            preds['arimax_vix'].append(float(fc.iloc[0]))
        except Exception:
            preds['arimax_vix'].append(float(_np.mean(y_train)))

        # 4. ARIMAX with VIX + SPY
        try:
            exog_tr  = _np.column_stack([d_vix[:t], ret_spy[:t]])
            exog_fc  = _np.array([[d_vix[t], ret_spy[t]]])
            res      = _ARIMA_sm(y_train, order=(init_p, 0, init_q),
                                 exog=exog_tr).fit()
            fc       = res.forecast(1, exog=exog_fc)
            preds['arimax_vix_spy'].append(float(fc.iloc[0]))
        except Exception:
            preds['arimax_vix_spy'].append(float(_np.mean(y_train)))

    actuals = _np.array(actuals)
    rows = []
    for name, pred in preds.items():
        p       = _np.array(pred)
        rmse    = float(_np.sqrt(_np.mean((actuals - p) ** 2)))
        mae     = float(_np.mean(_np.abs(actuals - p)))
        dir_acc = float(_np.mean(_np.sign(actuals) == _np.sign(p)))
        ic, _   = spearmanr(actuals, p)
        rows.append({'Model': name, 'RMSE': round(rmse, 6),
                     'MAE': round(mae, 6),
                     'Dir_Acc': round(dir_acc, 4),
                     'IC': round(float(ic), 4)})
    return _pd.DataFrame(rows)


def _plot_arima_comparison(df_comp: _pd.DataFrame, ticker: str) -> None:
    """Bar chart of RMSE and directional accuracy."""
    fig, axes = _plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f'{ticker} — ARMA / ARIMAX Walk-Forward Comparison\n'
        'Baselines: zero-return naive  |  Exog: VIX Δ, SPY return',
        fontsize=12,
    )
    labels = ['Naive\nbaseline', 'ARMA', 'ARIMAX\n+VIX', 'ARIMAX\n+VIX+SPY']
    colors = ['#9E9E9E', '#2196F3', '#FF9800', '#4CAF50']

    for ax, metric, title in zip(
        axes,
        ['RMSE', 'Dir_Acc'],
        ['RMSE (lower = better)', 'Directional Accuracy (higher = better)'],
    ):
        ax.bar(labels, df_comp[metric], color=colors, edgecolor='white')
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', labelsize=9)

    fig.tight_layout()
    _arima_save(fig, f'{ticker}_arima_comparison.png')


def _plot_arima_forecast(df: _pd.DataFrame, ticker: str,
                         forecast_days: int = 10) -> None:
    """Price-level 10-day forecast from best ARMA with confidence interval."""
    from statsmodels.tsa.arima.model import ARIMA as _ARIMA_sm

    rets    = df['ret']
    closes  = df[ticker]
    p, q    = _best_arma_order(rets)

    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        model  = _ARIMA_sm(rets, order=(p, 0, q))
        result = model.fit()
        fc     = result.get_forecast(forecast_days)
        ci     = fc.conf_int(alpha=0.05)
        fc_ret = fc.predicted_mean.values

        last_price = float(closes.iloc[-1])
        prices_fc  = last_price * _np.exp(_np.cumsum(fc_ret))
        lo_prices  = last_price * _np.exp(_np.cumsum(ci.iloc[:, 0].values))
        hi_prices  = last_price * _np.exp(_np.cumsum(ci.iloc[:, 1].values))

        future_idx = _pd.date_range(closes.index[-1], periods=forecast_days + 1,
                                    freq='B')[1:]

        fig, ax = _plt.subplots(figsize=(14, 5))
        ax.plot(closes.index[-60:], closes.iloc[-60:],
                label='Historical Price', color='#212121', linewidth=1.5)
        ax.plot(future_idx, prices_fc,
                label=f'ARMA({p},{q}) Forecast', color='#2196F3',
                linewidth=2, linestyle='--')
        ax.fill_between(future_idx, lo_prices, hi_prices,
                        alpha=0.2, color='#2196F3', label='95% CI')
        ax.axvline(closes.index[-1], color='gray', linestyle=':', linewidth=1)
        ax.set_title(f'{ticker} — ARMA({p},{q}) 10-Day Price Forecast')
        ax.set_ylabel(f'{ticker} Price ($)')
        ax.legend(fontsize=9)
        fig.tight_layout()
        _arima_save(fig, f'{ticker}_arima_forecast.png')
    except Exception as e:
        print(f'  [warn] forecast plot failed: {e}')


def run_arima_suite(ticker: str = 'NVDA') -> None:
    print(f'\n[arima suite]  {ticker}')
    print(f'  Output → {_ARIMA_OUT}')
    print('  Running walk-forward comparison (this takes ~30s) …')

    df = _load_price_and_exog(ticker)
    df_comp = _walk_forward_compare(df, ticker)

    path = _ARIMA_OUT / f'{ticker}_arima_comparison.csv'
    df_comp.to_csv(path, index=False)
    print(f'  ✓ {path.name}')

    sep = '=' * 62
    print(f'\n{sep}')
    print(f'  ARIMA / ARIMAX COMPARISON — {ticker}')
    print(sep)
    print(df_comp.to_string(index=False))
    naive_rmse = df_comp.loc[df_comp['Model'] == 'naive', 'RMSE'].iloc[0]
    best_rmse  = df_comp['RMSE'].min()
    best_name  = df_comp.loc[df_comp['RMSE'].idxmin(), 'Model']
    improve    = (naive_rmse - best_rmse) / naive_rmse * 100
    print(f'\n  Best RMSE model: {best_name}  '
          f'(improvement vs naive: {improve:.2f}%)')
    if improve < 5:
        print('  → Returns are near-white-noise — ARIMA barely improves on naive.')
        print('    This confirms why ML with richer features is needed.')
    print(sep)

    _plot_arima_comparison(df_comp, ticker)
    _plot_arima_forecast(df, ticker)


if __name__ == '__main__':
    ticker = _sys.argv[1] if len(_sys.argv) > 1 else 'NVDA'
    ARIMAModel().run(ticker)
    run_arima_suite(ticker)

