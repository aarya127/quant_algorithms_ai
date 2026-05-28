"""
cointegrations.py — Cointegration Analysis

Two non-stationary series are cointegrated if a linear combination of them is
stationary (I(0)).  This implies a long-run equilibrium relationship and
mean-reverting spread — the foundation of statistical arbitrage / pairs trading.

Tests implemented:
  1.  Engle-Granger two-step test  (pairwise)
  2.  Johansen trace & max-eigenvalue test  (multivariate; detects rank)
  3.  Spread z-score + half-life (via AR(1) on spread)
  4.  Hurst exponent               (H < 0.5 → mean-reverting)

Candidate pairs (semiconductor universe):
  NVDA vs AMD    — direct GPU/AI competitor
  NVDA vs SMH    — Semiconductor ETF
  NVDA vs SOXX   — iShares Semiconductor ETF
  NVDA vs QQQ    — Nasdaq-100
  NVDA vs AVGO   — Broadcom (AI chips)
  NVDA vs TSM    — TSMC (fab relationship)

Outputs (output/cointegrations/):
  cointegration_pairs.csv          — p-values + half-lives for all pairs
  {pair}_spread.png                — price ratio / spread + z-score + signal
  {pair}_johansen.csv              — Johansen trace/maxeig statistics
  hurst.csv                        — Hurst exponents

Usage: python cointegrations.py [BASE_TICKER]   (default: NVDA)
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

OUT_DIR = Path(__file__).parent / 'output' / 'cointegrations'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Pairs to test against the base ticker
_PEERS = ['AMD', 'SMH', 'SOXX', 'QQQ', 'AVGO', 'TSM']


# Helpers

def _save(fig, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {path.name}')


def load_prices(tickers: list, period: str = '3y') -> pd.DataFrame:
    """Download adjusted close prices for all tickers, return aligned DataFrame."""
    import yfinance as yf
    closes = {}
    for t in tickers:
        try:
            d = yf.download(t, period=period, progress=False, auto_adjust=True)
            closes[t] = d['Close'].squeeze()
        except Exception as e:
            print(f'  [warn] {t}: {e}')
    df = pd.DataFrame(closes).dropna()
    return df


# Engle-Granger test

def engle_granger(y: pd.Series, x: pd.Series) -> dict:
    """
    Two-step Engle-Granger cointegration test.

    Step 1: OLS regression  y = α + β·x  →  compute residuals ê
    Step 2: ADF test on ê   (H0: ê is I(1), i.e. NO cointegration)

    Returns p-value, hedge ratio β, and ADF statistic.
    """
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tsa.stattools import adfuller

    x_c   = np.column_stack([np.ones(len(x)), x.values])
    model = OLS(y.values, x_c).fit()
    beta  = float(model.params[1])
    alpha = float(model.params[0])
    resid = y.values - (alpha + beta * x.values)

    adf   = adfuller(resid, autolag='AIC')
    return {
        'hedge_ratio':  round(beta, 4),
        'intercept':    round(alpha, 4),
        'adf_stat':     round(float(adf[0]), 4),
        'p_value':      round(float(adf[1]), 4),
        'cointegrated': float(adf[1]) < 0.05,
        'spread':       pd.Series(resid, index=y.index),
    }


# Half-life and Hurst exponent

def half_life(spread: pd.Series) -> float:
    """
    Estimate mean-reversion half-life via AR(1) regression on the spread.
    Half-life = -ln(2) / ln(φ),  where spread_t = φ · spread_{t-1} + ε
    Shorter half-life → faster mean reversion → more tradeable.
    """
    from statsmodels.regression.linear_model import OLS
    s     = spread.values
    delta = np.diff(s)
    lag   = s[:-1].reshape(-1, 1)
    x     = np.column_stack([np.ones(len(lag)), lag])
    res   = OLS(delta, x).fit()
    phi   = float(res.params[1])
    if phi >= 0:
        return np.inf
    return round(-np.log(2) / phi, 1)


def hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
    """
    Hurst exponent H via R/S analysis.
    H < 0.5 → mean-reverting (supports pairs trading)
    H ≈ 0.5 → random walk
    H > 0.5 → trending / momentum
    """
    lags  = range(2, min(max_lag, len(series) // 4))
    rs    = []
    for lag in lags:
        chunks = [series.values[i:i+lag] for i in range(0, len(series) - lag, lag)]
        r_s_vals = []
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            mean   = np.mean(chunk)
            cumdev = np.cumsum(chunk - mean)
            R      = np.max(cumdev) - np.min(cumdev)
            S      = np.std(chunk, ddof=1)
            if S > 0:
                r_s_vals.append(R / S)
        if r_s_vals:
            rs.append((lag, np.mean(r_s_vals)))

    if len(rs) < 2:
        return np.nan
    lags_arr = np.log([r[0] for r in rs])
    rs_arr   = np.log([r[1] for r in rs])
    H        = float(np.polyfit(lags_arr, rs_arr, 1)[0])
    return round(H, 4)


# Johansen test

def johansen_test(y: pd.Series, x: pd.Series) -> dict:
    """
    Johansen trace and max-eigenvalue tests for cointegration rank.
    H0 (trace): rank ≤ r  →  reject at 5% if trace stat > critical value.
    """
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    data = np.column_stack([y.values, x.values])
    try:
        res   = coint_johansen(data, det_order=0, k_ar_diff=1)
        trace = res.lr1       # trace statistics
        crit  = res.cvt       # critical values [90%, 95%, 99%] per rank
        row0  = {
            'r0_trace':  round(float(trace[0]), 3),
            'r0_cv95':   round(float(crit[0, 1]), 3),
            'r0_reject': trace[0] > crit[0, 1],
            'r1_trace':  round(float(trace[1]), 3),
            'r1_cv95':   round(float(crit[1, 1]), 3),
            'r1_reject': trace[1] > crit[1, 1],
        }
        cointegration_rank = int(row0['r0_reject']) + int(row0['r1_reject'])
        row0['coint_rank'] = cointegration_rank
        return row0
    except Exception as e:
        return {'error': str(e)}


# Spread z-score and trading signal

def compute_zscore(spread: pd.Series, window: int = 60) -> pd.Series:
    """Rolling z-score of the spread (lookback = window days)."""
    mean = spread.rolling(window).mean()
    std  = spread.rolling(window).std()
    return ((spread - mean) / std).dropna()


def plot_spread(base: str, peer: str,
                prices: pd.DataFrame, eg_result: dict,
                z_entry: float = 1.5, z_exit: float = 0.5) -> None:
    """
    Three-panel plot:
      Top:    Normalised price paths of both assets
      Middle: Spread (hedge-ratio-adjusted residual)
      Bottom: Rolling z-score with entry/exit bands
    """
    y   = np.log(prices[base])
    x   = np.log(prices[peer])
    spr = eg_result['spread']
    z   = compute_zscore(spr)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f'{base} vs {peer} — Cointegration Spread Analysis\n'
        f'p={eg_result["p_value"]:.4f}  '
        f'β={eg_result["hedge_ratio"]:.3f}  '
        f'cointegrated={"✓" if eg_result["cointegrated"] else "✗"}',
        fontsize=12,
    )

    # Normalised log-prices
    ax = axes[0]
    ax.plot((y - y.iloc[0]), label=base, color='#2196F3', linewidth=1.5)
    ax.plot((x - x.iloc[0]), label=peer, color='#FF5722', linewidth=1.5)
    ax.set_ylabel('Log Price (normalised)')
    ax.set_title('Log-Price Paths')
    ax.legend()

    # Spread
    ax = axes[1]
    ax.plot(spr, color='#9C27B0', linewidth=1.2)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.fill_between(spr.index, spr, 0, alpha=0.15, color='#9C27B0')
    ax.set_ylabel('Spread (log OLS residual)')
    ax.set_title('Cointegration Spread — should be stationary if cointegrated')

    # Z-score with signals
    ax = axes[2]
    ax.plot(z, color='#212121', linewidth=1.2, label='Z-score')
    ax.axhline( z_entry,  color='#F44336', linestyle='--', linewidth=1.2, label=f'Short entry (+{z_entry})')
    ax.axhline(-z_entry,  color='#4CAF50', linestyle='--', linewidth=1.2, label=f'Long entry  (-{z_entry})')
    ax.axhline( z_exit,   color='#FF9800', linestyle=':',  linewidth=1.0, label=f'Exit ({z_exit})')
    ax.axhline(-z_exit,   color='#FF9800', linestyle=':',  linewidth=1.0)
    ax.axhline(0, color='black', linewidth=0.4)
    ax.fill_between(z.index, z, 0,
                    where=(z >  z_entry), alpha=0.15, color='#F44336')
    ax.fill_between(z.index, z, 0,
                    where=(z < -z_entry), alpha=0.15, color='#4CAF50')
    ax.set_ylabel('Z-Score')
    ax.set_title('Rolling Z-Score — entry/exit signals for pairs strategy')
    ax.legend(fontsize=8, loc='upper right')

    fig.tight_layout()
    pair_name = f'{base}_{peer}'
    _save(fig, f'{pair_name}_spread.png')


# Entry point

def run(base: str = 'NVDA') -> None:
    print(f'\n[cointegrations]  {base} vs {_PEERS}')
    print(f'  Output → {OUT_DIR}')

    all_tickers = [base] + _PEERS
    prices      = load_prices(all_tickers)
    available   = [p for p in _PEERS if p in prices.columns]

    rows         = []
    johansen_rows = []
    hurst_rows   = []

    for peer in available:
        y = np.log(prices[base])
        x = np.log(prices[peer])

        eg  = engle_granger(y, x)
        joh = johansen_test(y, x)
        hl  = half_life(eg['spread'])
        H   = hurst_exponent(eg['spread'])

        rows.append({
            'Pair':          f'{base}/{peer}',
            'EG_p_value':    eg['p_value'],
            'EG_cointegrated': eg['cointegrated'],
            'hedge_ratio':   eg['hedge_ratio'],
            'half_life_d':   hl,
            'hurst':         H,
            'tradeable':     eg['cointegrated'] and hl < 60 and H < 0.5,
        })

        johansen_rows.append({
            'Pair':         f'{base}/{peer}',
            **{k: v for k, v in joh.items() if k != 'error'},
        })

        hurst_rows.append({'Pair': f'{base}/{peer}', 'Hurst': H})

        plot_spread(base, peer, prices, eg)

    df_pairs = pd.DataFrame(rows)
    df_joh   = pd.DataFrame(johansen_rows)
    df_hurst = pd.DataFrame(hurst_rows)

    df_pairs.to_csv(OUT_DIR / 'cointegration_pairs.csv', index=False)
    df_joh.to_csv(OUT_DIR / 'johansen_results.csv', index=False)
    df_hurst.to_csv(OUT_DIR / 'hurst.csv', index=False)
    print(f'  ✓ cointegration_pairs.csv')
    print(f'  ✓ johansen_results.csv')
    print(f'  ✓ hurst.csv')

    sep = '=' * 70
    print(f'\n{sep}')
    print(f'  COINTEGRATION ANALYSIS — {base}')
    print(sep)
    print(df_pairs[['Pair', 'EG_p_value', 'EG_cointegrated',
                    'hedge_ratio', 'half_life_d', 'hurst', 'tradeable']].to_string(index=False))
    tradeable = df_pairs[df_pairs['tradeable']]['Pair'].tolist()
    print(f'\n  Tradeable pairs (cointegrated + HL < 60d + H < 0.5):')
    if tradeable:
        for p in tradeable:
            row = df_pairs[df_pairs['Pair'] == p].iloc[0]
            print(f'    {p}  half_life={row.half_life_d:.0f}d  hurst={row.hurst:.3f}')
        print(f'\n  Strategy: long/short based on spread z-score')
        print(f'  Entry: |z| > 1.5   Exit: |z| < 0.5')
        print(f'  Feature ideas: spread_zscore, dist_from_equilibrium')
    else:
        print(f'    No strongly tradeable pairs found in current window.')
        print(f'    → Cointegration may be time-varying; try rolling tests.')
    print(sep)


if __name__ == '__main__':
    run(sys.argv[1] if len(sys.argv) > 1 else 'NVDA')
