"""
volatility_clustering.py — Volatility Clustering Analysis

Central finding: large moves tend to be followed by large moves; small by small.

Evidence collected via:
  1.  Daily returns time-series (bars)
  2.  Squared returns (clustering visible as bursts)
  3.  Rolling 20d / 60d realized volatility
  4.  ACF of absolute returns      (non-zero → persistence)
  5.  ACF of squared returns       (non-zero → ARCH effects)
  6.  Ljung-Box test on returns, |returns|, squared returns
  7.  ARCH LM test (Engle)         (H0: no ARCH effects)
  8.  Calm vs stress regime comparison

Research question:
  Does NVDA exhibit volatility clustering, and is it stronger in the
  high-volatility / stress regime?

Outputs (saved to output/volatility_clustering/):
  {TICKER}_returns_overview.png   — returns, squared returns, rolling vol
  {TICKER}_acf.png                — ACF of |returns| and squared returns
  {TICKER}_regime_clustering.png  — calm vs stress ACF comparison
  {TICKER}_ljung_box.csv
  {TICKER}_arch_lm.csv

Usage:
  python volatility_clustering.py [TICKER]   (default: NVDA)
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

OUT_DIR = Path(__file__).parent / 'output' / 'volatility_clustering'
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {path.name}')


def load_data(ticker: str, period: str = '3y') -> pd.DataFrame:
    import yfinance as yf
    data   = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    close  = data['Close'].squeeze()
    volume = data['Volume'].squeeze()

    df = pd.DataFrame({'Close': close, 'Volume': volume})
    df['log_return']  = np.log(df['Close'] / df['Close'].shift(1))
    df['abs_return']  = df['log_return'].abs()
    df['sq_return']   = df['log_return'] ** 2
    df['rv_20d']      = df['log_return'].rolling(20).std() * np.sqrt(252) * 100
    df['rv_60d']      = df['log_return'].rolling(60).std() * np.sqrt(252) * 100
    return df.dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_returns_overview(df: pd.DataFrame, ticker: str) -> None:
    """Three-panel figure: returns, squared returns, rolling vol."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'{ticker} — Volatility Clustering Evidence', fontsize=14, fontweight='bold')

    r = df['log_return'] * 100

    ax = axes[0]
    ax.bar(df.index, r, color=np.where(r >= 0, '#2196F3', '#F44336'), width=1, alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel('Return (%)')
    ax.set_title('Daily Log Returns — large moves cluster together')

    ax = axes[1]
    ax.bar(df.index, df['sq_return'] * 1e4, color='#FF9800', width=1, alpha=0.85)
    ax.set_ylabel('Squared Return (×10⁻⁴)')
    ax.set_title('Squared Returns — bursts confirm ARCH effects')

    ax = axes[2]
    ax.plot(df.index, df['rv_20d'], label='20d Realized Vol', color='#9C27B0', linewidth=1.5)
    ax.plot(df.index, df['rv_60d'], label='60d Realized Vol', color='#FF5722', linewidth=1.2, linestyle='--')
    ax.fill_between(df.index, df['rv_20d'], alpha=0.15, color='#9C27B0')
    ax.set_ylabel('Annualised Vol (%)')
    ax.set_title('Rolling Realized Volatility — persistence of vol regimes')
    ax.legend(loc='upper right', fontsize=9)

    fig.tight_layout()
    _save(fig, f'{ticker}_returns_overview.png')


def plot_acf(df: pd.DataFrame, ticker: str, nlags: int = 40) -> None:
    """ACF of absolute returns and squared returns — both should be non-zero."""
    from statsmodels.graphics.tsaplots import plot_acf as sm_acf

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'{ticker} — Autocorrelation of Absolute and Squared Returns\n'
        'Non-zero autocorrelation = volatility clustering / ARCH effects',
        fontsize=12,
    )

    sm_acf(
        df['abs_return'].dropna(), lags=nlags, ax=axes[0], alpha=0.05,
        title='ACF of |Returns|\n(persistence of volatility magnitude)',
    )
    sm_acf(
        df['sq_return'].dropna(), lags=nlags, ax=axes[1], alpha=0.05,
        title='ACF of Squared Returns\n(direct test of ARCH effects)',
    )
    for ax in axes:
        ax.set_xlabel('Lag (days)')

    fig.tight_layout()
    _save(fig, f'{ticker}_acf.png')


def plot_regime_clustering(df: pd.DataFrame, ticker: str) -> None:
    """
    Split into calm / stress by 20d-vol median.
    Compare ACF of squared returns in each regime.
    Research question: is clustering stronger in the stress regime?
    """
    from statsmodels.graphics.tsaplots import plot_acf as sm_acf

    median_vol  = df['rv_20d'].median()
    calm_mask   = df['rv_20d'] <= median_vol
    stress_mask = ~calm_mask

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'{ticker} — Volatility Clustering: Calm vs Stress Regime\n'
        f'Split at median 20d Realized Vol = {median_vol:.1f}%',
        fontsize=12,
    )

    sm_acf(
        df.loc[calm_mask,   'sq_return'].dropna(), lags=20, ax=axes[0], alpha=0.05,
        title=f'Calm Regime  (n={calm_mask.sum()} days)\nACF Squared Returns',
    )
    sm_acf(
        df.loc[stress_mask, 'sq_return'].dropna(), lags=20, ax=axes[1], alpha=0.05,
        title=f'Stress Regime  (n={stress_mask.sum()} days)\nACF Squared Returns',
    )
    axes[0].set_facecolor('#E8F5E9')
    axes[1].set_facecolor('#FFEBEE')
    for ax in axes:
        ax.set_xlabel('Lag (days)')

    fig.tight_layout()
    _save(fig, f'{ticker}_regime_clustering.png')


# ─────────────────────────────────────────────────────────────────────────────
# Statistical tests
# ─────────────────────────────────────────────────────────────────────────────

def ljung_box_test(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Ljung-Box test at lags 5, 10, 20 for:
      - Returns           (test for return autocorrelation)
      - |Returns|         (test for volatility clustering)
      - Squared Returns   (direct ARCH test)
    H0: no autocorrelation  →  p < 0.05 = reject H0 = autocorrelation present.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    rows = []
    for col, label in [
        ('log_return', 'Returns'),
        ('abs_return', '|Returns|'),
        ('sq_return',  'Squared Returns'),
    ]:
        for lag in [5, 10, 20]:
            lb = acorr_ljungbox(df[col].dropna(), lags=[lag], return_df=True)
            rows.append({
                'Series':       label,
                'Lag':          lag,
                'LB_stat':      round(float(lb['lb_stat'].iloc[0]), 3),
                'p_value':      round(float(lb['lb_pvalue'].iloc[0]), 4),
                'significant':  float(lb['lb_pvalue'].iloc[0]) < 0.05,
            })

    result = pd.DataFrame(rows)
    path   = OUT_DIR / f'{ticker}_ljung_box.csv'
    result.to_csv(path, index=False)
    print(f'  ✓ {path.name}')
    return result


def arch_lm_test(df: pd.DataFrame, ticker: str, lags: int = 12) -> dict:
    """
    Engle's ARCH LM test on return residuals.
    H0: no ARCH effects (squared residuals are white noise).
    Reject H0 (p < 0.05) → ARCH effects present → GARCH is appropriate.
    """
    from statsmodels.stats.diagnostic import het_arch

    r = df['log_return'].dropna().values
    lm_stat, lm_p, f_stat, f_p = het_arch(r, nlags=lags)

    result = {
        'LM_stat':               round(lm_stat, 3),
        'LM_p_value':            round(lm_p, 4),
        'F_stat':                round(f_stat, 3),
        'F_p_value':             round(f_p, 4),
        'ARCH_effects_present':  lm_p < 0.05,
    }
    pd.DataFrame([result]).to_csv(OUT_DIR / f'{ticker}_arch_lm.csv', index=False)
    print(f'  ✓ {ticker}_arch_lm.csv')
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(ticker: str, df: pd.DataFrame,
                  lb: pd.DataFrame, arch_res: dict) -> None:
    r = df['log_return'] * 100
    sep = '=' * 62
    print(f'\n{sep}')
    print(f'  VOLATILITY CLUSTERING — {ticker}')
    print(sep)
    print(f'  Period       : {df.index[0].date()} → {df.index[-1].date()}')
    print(f'  Trading days : {len(df)}')
    print(f'  Mean return  : {r.mean():.4f}%  |  Std: {r.std():.4f}%')
    print(f'  Skewness     : {r.skew():.3f}    |  Excess kurtosis: {r.kurtosis():.3f}')
    print(f'\n  Ljung-Box on |Returns| (lag=10):')
    row = lb.query("Series == '|Returns|' and Lag == 10").iloc[0]
    tag = 'autocorrelation present ✓' if row.significant else 'no autocorrelation'
    print(f'    LB stat={row.LB_stat:.3f}  p={row.p_value:.4f}  ({tag})')
    print(f'\n  Ljung-Box on Squared Returns (lag=10):')
    row2 = lb.query("Series == 'Squared Returns' and Lag == 10").iloc[0]
    tag2 = 'ARCH effects ✓' if row2.significant else 'no ARCH effects'
    print(f'    LB stat={row2.LB_stat:.3f}  p={row2.p_value:.4f}  ({tag2})')
    print(f'\n  ARCH LM Test (lag={12}):')
    tag3 = 'ARCH effects present → use GARCH ✓' if arch_res['ARCH_effects_present'] else 'no ARCH effects'
    print(f'    LM stat={arch_res["LM_stat"]:.3f}  p={arch_res["LM_p_value"]:.4f}  ({tag3})')
    print(f'\n  Current 20d Realized Vol : {df["rv_20d"].iloc[-1]:.1f}%  annualised')
    print(f'  Max 20d Realized Vol     : {df["rv_20d"].max():.1f}%  annualised')
    print(f'\n  → Research conclusion:')
    if arch_res['ARCH_effects_present']:
        print(f'    {ticker} exhibits statistically significant volatility clustering.')
        print(f'    GARCH-family models are warranted (see garch.py).')
    else:
        print(f'    No strong volatility clustering detected.')
    print(f'{sep}\n')


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(ticker: str = 'NVDA') -> None:
    print(f'\n[volatility_clustering]  {ticker}')
    print(f'  Output → {OUT_DIR}')
    df       = load_data(ticker)
    plot_returns_overview(df, ticker)
    plot_acf(df, ticker)
    lb       = ljung_box_test(df, ticker)
    arch_res = arch_lm_test(df, ticker)
    plot_regime_clustering(df, ticker)
    print_summary(ticker, df, lb, arch_res)


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else 'NVDA')
