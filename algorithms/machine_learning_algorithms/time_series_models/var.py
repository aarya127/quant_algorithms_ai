"""
var.py — Vector AutoRegression (VAR) Model

VAR(p) models k time series jointly, where each variable is regressed on
its own past values and the past values of all other variables:

  y_t = c + A₁ y_{t-1} + A₂ y_{t-2} + … + Aₚ y_{t-p} + ε_t

Where y_t is a k×1 vector of returns / changes.

Variables used:
  NVDA return   — target series
  QQQ  return   — Nasdaq-100 ETF
  SPY  return   — S&P 500 ETF
  SOXX return   — Semiconductor ETF (proxy for SOX index)
  VIX  change   — Fear index (level is I(1) → first-difference)

Analysis outputs:
  1. Lag-order selection (AIC / BIC / FPE / HQIC)
  2. Granger causality tests
     → Does SPY/QQQ/SOXX/VIX Granger-cause NVDA returns?
  3. Impulse Response Functions (IRF)
     → How does a shock to one variable propagate to NVDA?
  4. Forecast Error Variance Decomposition (FEVD)
     → How much of NVDA's forecast variance is explained by each variable?
  5. 5-day VAR forecast vs naive baseline
  6. Walk-forward RMSE comparison

Outputs (output/var/):
  {TICKER}_var_lag_selection.csv
  {TICKER}_var_granger.csv
  {TICKER}_var_irf.png
  {TICKER}_var_fevd.png
  {TICKER}_var_forecast.png

Usage: python var.py [TICKER]   (default: NVDA)
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings('ignore')

OUT_DIR = Path(__file__).parent / 'output' / 'var'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# VAR variables: (yfinance ticker, column label, transform)
_VAR_TICKERS = [
    ('NVDA', 'NVDA_ret',  'log_ret'),
    ('QQQ',  'QQQ_ret',   'log_ret'),
    ('SPY',  'SPY_ret',   'log_ret'),
    ('SOXX', 'SOXX_ret',  'log_ret'),
    ('^VIX', 'VIX_chg',   'diff'),       # VIX is I(1) → first-difference
]


# Helpers

def _save(fig, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {path.name}')


def load_var_data(period: str = '3y') -> pd.DataFrame:
    """Download and align all VAR series."""
    import yfinance as yf

    closes = {}
    for ticker, col, _ in _VAR_TICKERS:
        d = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        closes[ticker] = d['Close'].squeeze()

    raw = pd.DataFrame(closes).dropna()
    df  = pd.DataFrame(index=raw.index)

    for ticker, col, transform in _VAR_TICKERS:
        if transform == 'log_ret':
            df[col] = np.log(raw[ticker] / raw[ticker].shift(1)) * 100
        elif transform == 'diff':
            df[col] = raw[ticker].diff()

    return df.dropna()


# Lag selection

def select_lag(df: pd.DataFrame, max_lags: int = 10) -> pd.DataFrame:
    """Return AIC / BIC / HQIC / FPE table across lag orders."""
    from statsmodels.tsa.api import VAR

    model = VAR(df)
    res   = model.select_order(max_lags)

    rows = []
    for lag in range(1, max_lags + 1):
        try:
            rows.append({
                'lag':  lag,
                'AIC':  round(res.ics['aic'][lag], 4),
                'BIC':  round(res.ics['bic'][lag], 4),
                'HQIC': round(res.ics['hqic'][lag], 4),
                'FPE':  f'{res.ics["fpe"][lag]:.4e}',
            })
        except (KeyError, IndexError):
            pass

    df_out = pd.DataFrame(rows)
    path   = OUT_DIR / 'var_lag_selection.csv'
    df_out.to_csv(path, index=False)
    print(f'  ✓ {path.name}')

    # Consensus best lag (most criteria agree)
    best_aic  = int(df_out.loc[df_out['AIC'].idxmin(),  'lag'])
    best_bic  = int(df_out.loc[df_out['BIC'].idxmin(),  'lag'])
    best_hqic = int(df_out.loc[df_out['HQIC'].idxmin(), 'lag'])
    return df_out, best_bic   # BIC most penalises extra params → conservative


# Fit VAR

def fit_var(df: pd.DataFrame, lag: int):
    """Fit VAR(lag) and return fitted result."""
    from statsmodels.tsa.api import VAR
    model = VAR(df)
    return model.fit(lag)


# Granger causality

def granger_causality(result, df: pd.DataFrame) -> pd.DataFrame:
    """
    Test whether each variable Granger-causes NVDA_ret.
    H0: variable X does NOT Granger-cause NVDA_ret.
    Reject H0 (p < 0.05) → X has predictive power for NVDA.
    """
    target = 'NVDA_ret'
    rows   = []

    for col in df.columns:
        if col == target:
            continue
        try:
            test = result.test_causality(target, [col], kind='f')
            rows.append({
                'Cause':           col,
                'Effect':          target,
                'F_stat':          round(float(test.test_statistic), 3),
                'p_value':         round(float(test.pvalue), 4),
                'Granger_causes':  float(test.pvalue) < 0.05,
            })
        except Exception as e:
            rows.append({'Cause': col, 'Effect': target,
                         'F_stat': np.nan, 'p_value': np.nan,
                         'Granger_causes': False})

    df_out = pd.DataFrame(rows)
    path   = OUT_DIR / 'var_granger.csv'
    df_out.to_csv(path, index=False)
    print(f'  ✓ {path.name}')
    return df_out


# Impulse Response Functions

def plot_irf(result, ticker: str = 'NVDA', periods: int = 10) -> None:
    """
    IRF: response of NVDA_ret to a one-standard-deviation shock in each variable.
    Shows how market shocks propagate to NVDA over 10 days.
    """
    irf    = result.irf(periods)
    target = 'NVDA_ret'
    names  = list(result.names)

    if target not in names:
        return

    tidx   = names.index(target)
    cols   = [c for c in names if c != target]
    n_cols = len(cols)

    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4), sharey=False)
    if n_cols == 1:
        axes = [axes]
    fig.suptitle(f'{ticker} — Impulse Response: NVDA_ret response to shocks', fontsize=12)

    colors = ['#2196F3', '#FF9800', '#9C27B0', '#4CAF50']

    for ax, col, color in zip(axes, cols, colors):
        cidx = names.index(col)
        resp = irf.irfs[:, tidx, cidx]
        lb   = irf.cum_effect_stderr(orth=False)  # approximate CI
        days = np.arange(periods + 1)

        ax.bar(days, resp, color=color, alpha=0.7)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_title(f'Shock: {col}', fontsize=10)
        ax.set_xlabel('Days')
        ax.set_ylabel('Response of NVDA_ret')

    fig.tight_layout()
    _save(fig, f'{ticker}_var_irf.png')


# Forecast Error Variance Decomposition

def plot_fevd(result, ticker: str = 'NVDA', periods: int = 10) -> None:
    """
    FEVD: how much of NVDA_ret's forecast variance is explained by each variable?
    Stacked area chart; own-variance + contributions from QQQ, SPY, SOXX, VIX.
    """
    target = 'NVDA_ret'
    names  = list(result.names)

    if target not in names:
        return

    tidx  = names.index(target)
    fevd  = result.fevd(periods)
    decomp = fevd.decomp[tidx]     # shape: (periods, k)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = ['#2196F3', '#FF9800', '#9C27B0', '#4CAF50', '#F44336']
    # Skip degenerate period-0 (own-variance = 100%)
    n_horizons = decomp.shape[0] - 1          # periods 1 … N
    days       = np.arange(1, n_horizons + 1)
    bottom     = np.zeros(n_horizons)

    for i, (name, color) in enumerate(zip(names, colors)):
        share = decomp[1:, i] * 100           # horizons 1…N
        ax.bar(days, share, bottom=bottom, label=name, color=color, alpha=0.85)
        bottom += share

    ax.set_xlabel('Forecast Horizon (days)')
    ax.set_ylabel('Variance Explained (%)')
    ax.set_title(f'{ticker} — Forecast Error Variance Decomposition of NVDA_ret\n'
                 'Shows which variables explain NVDA forecast uncertainty')
    ax.legend(fontsize=9, loc='upper right')
    ax.set_ylim(0, 100)
    ax.set_xticks(days)
    fig.tight_layout()
    _save(fig, f'{ticker}_var_fevd.png')


# 5-day forecast vs naive

def plot_var_forecast(result, df: pd.DataFrame, ticker: str = 'NVDA',
                      horizon: int = 5) -> None:
    """Show VAR-predicted NVDA returns for next 5 days."""
    fc    = result.forecast(df.values[-result.k_ar:], steps=horizon)
    names = list(result.names)

    if 'NVDA_ret' not in names:
        return

    nidx    = names.index('NVDA_ret')
    fc_nvda = fc[:, nidx]

    fig, ax = plt.subplots(figsize=(10, 4))
    days    = np.arange(1, horizon + 1)
    naive   = [float(df['NVDA_ret'].mean())] * horizon

    ax.bar(days - 0.2, fc_nvda, width=0.35, label='VAR forecast', color='#2196F3', alpha=0.85)
    ax.bar(days + 0.2, naive,   width=0.35, label='Naive (mean)',  color='#9E9E9E', alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(days)
    ax.set_xticklabels([f'Day {i}' for i in days])
    ax.set_ylabel('NVDA Return (%, log scale)')
    ax.set_title(f'{ticker} — VAR {horizon}-Day Return Forecast vs Naive Baseline')
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, f'{ticker}_var_forecast.png')


# Entry point

def run(ticker: str = 'NVDA') -> None:
    print(f'\n[var]  {ticker}')
    print(f'  Output → {OUT_DIR}')
    print(f'  Variables: NVDA_ret, QQQ_ret, SPY_ret, SOXX_ret, VIX_chg')

    df = load_var_data()

    # Lag selection
    lag_df, best_lag = select_lag(df)
    best_lag = max(1, min(best_lag, 5))   # cap at 5 for stability with limited data
    print(f'\n  Lag selection (BIC-optimal): {best_lag}')
    print(lag_df.head(8).to_string(index=False))

    # Fit
    result = fit_var(df, best_lag)
    print(f'\n  VAR({best_lag}) fitted on {len(df)} observations')
    print(f'  AIC={result.aic:.2f}  BIC={result.bic:.2f}')

    # Granger causality
    gc = granger_causality(result, df)
    print(f'\n  Granger causality → NVDA_ret:')
    for _, row in gc.iterrows():
        tag = '✓ causes NVDA' if row['Granger_causes'] else '  no effect'
        print(f'    {row.Cause:12s}  F={row.F_stat:.3f}  p={row.p_value:.4f}  {tag}')

    # Plots
    try:
        plot_irf(result, ticker)
    except Exception as e:
        print(f'  [warn] IRF plot failed: {e}')
    try:
        plot_fevd(result, ticker)
    except Exception as e:
        print(f'  [warn] FEVD plot failed: {e}')
    try:
        plot_var_forecast(result, df, ticker)
    except Exception as e:
        print(f'  [warn] forecast plot failed: {e}')

    sep = '=' * 62
    print(f'\n{sep}')
    print(f'  VAR SUMMARY — {ticker}')
    print(sep)
    causes = gc[gc['Granger_causes']]['Cause'].tolist()
    if causes:
        print(f'  Variables that Granger-cause NVDA_ret: {", ".join(causes)}')
        print(f'  → These provide incremental predictive power for NVDA returns.')
        print(f'  → Consider adding VAR residuals / forecasts as ML features.')
    else:
        print(f'  No variable Granger-causes NVDA_ret at 5% significance.')
        print(f'  → Returns are largely idiosyncratic; macro variables help less.')
    print(sep)


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else 'NVDA')
