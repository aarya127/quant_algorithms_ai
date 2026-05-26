"""
GARCH(1,1) Conditional Volatility Model

Generalised AutoRegressive Conditional Heteroskedasticity.

Model equations:
  r_t   = μ + ε_t,      ε_t = σ_t * z_t,      z_t ~ N(0,1)
  σ²_t  = ω + α * ε²_{t-1} + β * σ²_{t-1}

Key parameters:
  ω  — baseline (long-run) variance floor  (must be > 0)
  α  — ARCH term: sensitivity to recent shocks
  β  — GARCH term: persistence of past variance
  α+β — total persistence (< 1 for stationarity; close to 1 = long memory)

Outputs:
  - Conditional volatility path (annualised)
  - Long-run unconditional volatility  =  sqrt(ω / (1 − α − β))
  - Volatility half-life = ln(0.5) / ln(α + β)  days
  - 1-day 95% VaR and CVaR (Expected Shortfall)
  - AIC / BIC for model comparison

Fitted by maximum likelihood (Gaussian innovations); scipy L-BFGS-B solver.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


@dataclass
class GARCHParams:
    omega: float   # ω
    alpha: float   # α  (ARCH coefficient)
    beta:  float   # β  (GARCH coefficient)
    mu:    float   # mean daily return


@dataclass
class GARCHResult:
    params: GARCHParams
    log_likelihood: float
    aic: float
    bic: float
    persistence: float           # α + β
    half_life_days: float        # ln(0.5) / ln(α+β)
    long_run_vol_annual: float   # sqrt(ω/(1−α−β)) * sqrt(252)
    conditional_vol: pd.Series  # annualised conditional σ, length = n
    var_1d_95: float             # 1-day 95% VaR as % of price
    cvar_1d_95: float            # 1-day 95% CVaR (Expected Shortfall) as %


class GARCHModel:
    """
    GARCH(1,1) model fitted by maximum likelihood.

    Accepts either a price series (auto-converts to log-returns)
    or a pre-computed returns series.
    """

    def __init__(self):
        self._params: GARCHParams = None

    # ------------------------------------------------------------------
    # Negative log-likelihood
    # ------------------------------------------------------------------

    @staticmethod
    def _neg_loglik(theta: np.ndarray, returns: np.ndarray) -> float:
        mu, omega, alpha, beta = theta
        if omega <= 1e-10 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10

        n   = len(returns)
        eps = returns - mu

        sigma2 = np.empty(n)
        sigma2[0] = np.var(returns)
        for t in range(1, n):
            sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]

        ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + eps ** 2 / sigma2)
        return -ll

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, series: pd.Series) -> GARCHResult:
        """
        Fit GARCH(1,1) to prices or returns.

        If series values are mostly > 5 they are treated as prices and
        converted to log-returns automatically.
        """
        if float(series.dropna().median()) > 5:
            returns = np.log(series / series.shift(1)).dropna().values
        else:
            returns = series.dropna().values

        n    = len(returns)
        mu0  = np.mean(returns)
        var0 = np.var(returns)

        x0     = [mu0, 0.05 * var0, 0.10, 0.85]
        bounds = [(None, None), (1e-9, None), (1e-6, 0.5), (1e-6, 0.9999)]

        res = minimize(
            self._neg_loglik, x0, args=(returns,),
            method='L-BFGS-B', bounds=bounds,
        )

        mu, omega, alpha, beta = res.x
        params = GARCHParams(omega=omega, alpha=alpha, beta=beta, mu=mu)
        self._params = params

        # Conditional variance path
        eps    = returns - mu
        sigma2 = np.empty(n)
        sigma2[0] = var0
        for t in range(1, n):
            sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]

        cond_vol = pd.Series(np.sqrt(sigma2) * np.sqrt(252))

        # Fit statistics
        ll  = -res.fun
        k   = 4
        aic = 2 * k - 2 * ll
        bic = k * np.log(n) - 2 * ll

        # Derived
        persistence = alpha + beta
        half_life   = np.log(0.5) / np.log(persistence) if persistence < 1 else np.inf
        lrv_daily   = omega / (1 - persistence) if persistence < 1 else np.nan
        lr_vol_ann  = np.sqrt(lrv_daily) * np.sqrt(252) if not np.isnan(lrv_daily) else np.nan

        # Next-day VaR / CVaR
        next_sig2  = omega + alpha * eps[-1] ** 2 + beta * sigma2[-1]
        next_sigma = np.sqrt(next_sig2)
        var_95     = float(abs(norm.ppf(0.05)) * next_sigma) * 100
        cvar_95    = float(norm.pdf(norm.ppf(0.05)) / 0.05 * next_sigma) * 100

        return GARCHResult(
            params=params, log_likelihood=ll, aic=aic, bic=bic,
            persistence=persistence, half_life_days=half_life,
            long_run_vol_annual=lr_vol_ann,
            conditional_vol=cond_vol,
            var_1d_95=var_95, cvar_1d_95=cvar_95,
        )

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------

    def run(self, ticker: str, period: str = '3y') -> None:
        import yfinance as yf
        data  = yf.download(ticker, period=period, progress=False)
        close = data['Close'].squeeze()

        r = self.fit(close)
        p = r.params
        print(f"\n{ticker}  GARCH(1,1)  [{period}]")
        print(f"  μ     = {p.mu*100:.5f}%  (daily mean return)")
        print(f"  ω     = {p.omega:.2e}")
        print(f"  α     = {p.alpha:.4f}  (ARCH — shock impact)")
        print(f"  β     = {p.beta:.4f}  (GARCH — variance persistence)")
        print(f"  α+β   = {r.persistence:.4f}")
        print(f"  Half-life   : {r.half_life_days:.1f} days")
        print(f"  Long-run vol: {r.long_run_vol_annual*100:.1f}% annualised")
        print(f"  Current vol : {float(r.conditional_vol.iloc[-1]):.1f}% annualised")
        print(f"  1-day 95% VaR : {r.var_1d_95:.3f}%")
        print(f"  1-day 95% CVaR: {r.cvar_1d_95:.3f}%")
        print(f"  AIC: {r.aic:.1f}   BIC: {r.bic:.1f}")



# ─────────────────────────────────────────────────────────────────────────────
# GARCHSuite — multi-model comparison using arch-py
# ─────────────────────────────────────────────────────────────────────────────
"""
GARCHSuite compares four GARCH-family models:
  1.  GARCH(1,1)  — Bollerslev (1986). Benchmark.
  2.  GARCH(1,1)-t — Fat-tailed (Student-t) errors.  Stock returns have fat
      tails; normal errors understate extreme moves.
  3.  GJR-GARCH   — Glosten, Jagannathan, Runkle (1993).
      Leverage effect: bad news raises volatility more than equivalent good news.
      σ²_t = ω + (α + γ·I_{ε<0}) ε²_{t-1} + β σ²_{t-1}
  4.  EGARCH       — Nelson (1991). Log-variance form; no positivity constraints.
      ln σ²_t = ω + α|z_{t-1}| + γ z_{t-1} + β ln σ²_{t-1}

Forecasting:
  - 1-day ahead volatility (h.1)
  - 5-day cumulative volatility (h.5)

Comparison vs baselines:
  - Naive: rolling 20-day realized volatility
  - ML:    XGBoost target_vol_5d prediction (from supervised pipeline CSV if present)

Outputs (output/garch/):
  {TICKER}_garch_model_comparison.csv   — AIC / BIC / persistence table
  {TICKER}_garch_conditional_vol.png    — all four conditional vol paths
  {TICKER}_garch_model_comparison.png   — AIC / BIC bar chart
  {TICKER}_garch_forecast.png           — 5-day vol forecast strip
  {TICKER}_garch_vs_realized.png        — rolling backtest vs realized vol

Usage: python garch.py [TICKER]   (default: NVDA)
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

# ── Import the arch package, bypassing local arch.py in the same directory
_this_dir   = str(Path(__file__).parent.resolve())
_saved_path = sys.path[:]
sys.path = [p for p in sys.path if p not in ('', '.', _this_dir)]
try:
    from arch import arch_model as _arch_model
finally:
    sys.path = _saved_path
del _saved_path, _this_dir

_SUITE_OUT = Path(__file__).parent / 'output' / 'garch'
_SUITE_OUT.mkdir(parents=True, exist_ok=True)


def _save_suite(fig, name: str) -> None:
    path = _SUITE_OUT / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {path.name}')


def _load_returns(ticker: str, period: str = '3y') -> pd.Series:
    import yfinance as yf
    data  = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    close = data['Close'].squeeze()
    r     = np.log(close / close.shift(1)).dropna() * 100
    r.name = 'log_return_pct'
    return r


def _rv20(returns: pd.Series) -> pd.Series:
    """20-day rolling realized volatility, annualised percent."""
    return returns.rolling(20).std() * np.sqrt(252)


# ── model specs ──────────────────────────────────────────────────────────────

_MODEL_SPECS = [
    ('GARCH(1,1)',   dict(vol='GARCH', p=1, q=1, dist='normal')),
    ('GARCH(1,1)-t', dict(vol='GARCH', p=1, q=1, dist='t')),
    ('GJR-GARCH',    dict(vol='GARCH', p=1, o=1, q=1, dist='normal')),
    ('EGARCH',       dict(vol='EGARCH', p=1, q=1, dist='normal')),
]


def fit_suite(returns: pd.Series) -> tuple[pd.DataFrame, dict]:
    """
    Fit all four GARCH-family models.
    Returns (comparison_df, {name: fitted_result}).
    """
    rows    = {}
    results = {}

    for name, spec in _MODEL_SPECS:
        try:
            m   = _arch_model(returns, rescale=False, **spec)
            res = m.fit(disp='off')
            p   = res.params

            # persistence = α + β (GARCH), α + γ/2 + β (GJR approx), |β| (EGARCH)
            if name == 'EGARCH':
                persist = abs(float(p.get('beta[1]', 0)))
            elif name == 'GJR-GARCH':
                persist = float(p.get('alpha[1]', 0)) + \
                          0.5 * float(p.get('gamma[1]', 0)) + \
                          float(p.get('beta[1]', 0))
            else:
                persist = float(p.get('alpha[1]', 0)) + float(p.get('beta[1]', 0))

            half_life = (np.log(0.5) / np.log(persist)) if 0 < persist < 1 else np.inf

            rows[name] = {
                'Model':        name,
                'LogL':         round(res.loglikelihood, 2),
                'AIC':          round(res.aic, 2),
                'BIC':          round(res.bic, 2),
                'Persistence':  round(persist, 4),
                'Half_life_d':  round(half_life, 1) if half_life < 1e6 else '∞',
            }
            results[name] = res
        except Exception as e:
            print(f'  [warn] {name} failed: {e}')

    df = pd.DataFrame(list(rows.values()))
    return df, results


def _vol_forecast_1d_5d(res, returns: pd.Series) -> tuple[float, float]:
    """1-day and 5-day annualised volatility forecasts."""
    try:
        fc  = res.forecast(horizon=5, reindex=False)
        v1  = float(fc.variance.iloc[-1, 0])    # variance in (percent)²
        v5  = float(fc.variance.iloc[-1, :].sum())
        return np.sqrt(v1) * np.sqrt(252), np.sqrt(v5 / 5) * np.sqrt(252)
    except Exception:
        return np.nan, np.nan


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_conditional_vol(results: dict, returns: pd.Series, ticker: str) -> None:
    """All four conditional volatility paths on one chart."""
    colors = ['#2196F3', '#FF9800', '#9C27B0', '#4CAF50']
    fig, ax = plt.subplots(figsize=(14, 6))

    rv = _rv20(returns)
    ax.fill_between(rv.index, rv, alpha=0.1, color='gray', label='20d Realized Vol (actual)')

    for (name, _), color in zip(_MODEL_SPECS, colors):
        if name not in results:
            continue
        cond_v = results[name].conditional_volatility * np.sqrt(252)
        ax.plot(returns.index, cond_v.values, label=name, color=color,
                linewidth=1.2, alpha=0.85)

    ax.set_title(f'{ticker} — GARCH-Family Conditional Volatility (annualised %)')
    ax.set_ylabel('Volatility (% annualised)')
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save_suite(fig, f'{ticker}_garch_conditional_vol.png')


def plot_model_comparison(df_comp: pd.DataFrame, ticker: str) -> None:
    """AIC / BIC bar chart for model selection."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(f'{ticker} — GARCH-Family Model Selection (lower = better)', fontsize=12)

    colors = ['#2196F3', '#FF9800', '#9C27B0', '#4CAF50'][:len(df_comp)]

    for ax, metric in zip(axes, ['AIC', 'BIC']):
        ax.bar(df_comp['Model'], df_comp[metric], color=colors, edgecolor='white')
        ax.tick_params(axis='x', rotation=20)
        ax.set_ylabel(metric)
        ax.set_title(f'{metric}')
        best = df_comp.loc[df_comp[metric].idxmin(), 'Model']
        ax.axhline(df_comp[metric].min(), color='red', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.set_xlabel(f'Best: {best}', fontsize=9, color='red')

    fig.tight_layout()
    _save_suite(fig, f'{ticker}_garch_model_comparison.png')


def plot_forecast_strip(results: dict, returns: pd.Series, ticker: str) -> None:
    """Annotated 5-day volatility forecast for each model."""
    forecasts = {}
    for name in results:
        try:
            fc = results[name].forecast(horizon=5, reindex=False)
            daily_vols = np.sqrt(fc.variance.iloc[-1].values) * np.sqrt(252)
            forecasts[name] = daily_vols
        except Exception:
            pass

    if not forecasts:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#2196F3', '#FF9800', '#9C27B0', '#4CAF50']
    days   = np.arange(1, 6)

    for (name, color) in zip(forecasts, colors):
        ax.plot(days, forecasts[name], marker='o', label=name, color=color, linewidth=2)

    ax.set_xticks(days)
    ax.set_xticklabels([f'Day {i}' for i in days])
    ax.set_ylabel('Forecast Volatility (% annualised)')
    ax.set_title(f'{ticker} — 5-Day Ahead Volatility Forecast (all GARCH variants)')
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save_suite(fig, f'{ticker}_garch_forecast.png')


def plot_vs_realized(results: dict, returns: pd.Series, ticker: str) -> None:
    """
    Rolling 1-step-ahead GARCH(1,1) vol forecast vs 20d realized vol.
    Shows how well GARCH tracks actual realized volatility.
    """
    if 'GARCH(1,1)' not in results:
        return

    res      = results['GARCH(1,1)']
    cond_vol = res.conditional_volatility * np.sqrt(252)   # in-sample conditional vol
    rv20     = _rv20(returns)

    # Align indices
    common = cond_vol.index.intersection(rv20.dropna().index)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(common, rv20.loc[common],   label='20d Realized Vol (actual)', color='#F44336', linewidth=1.2)
    ax.plot(common, cond_vol.loc[common], label='GARCH(1,1) Conditional Vol', color='#2196F3', linewidth=1.2, alpha=0.8)

    corr = np.corrcoef(rv20.loc[common].values, cond_vol.loc[common].values)[0, 1]
    ax.set_title(
        f'{ticker} — GARCH(1,1) vs Realized Volatility\n'
        f'Pearson correlation = {corr:.3f}  (higher = better tracking)'
    )
    ax.set_ylabel('Volatility (% annualised)')
    ax.legend()
    fig.tight_layout()
    _save_suite(fig, f'{ticker}_garch_vs_realized.png')


# ── entry point ───────────────────────────────────────────────────────────────

def run_suite(ticker: str = 'NVDA') -> None:
    print(f'\n[garch suite]  {ticker}')
    print(f'  Output → {_SUITE_OUT}')

    returns   = _load_returns(ticker)
    df_comp, results = fit_suite(returns)

    # Save comparison table
    path = _SUITE_OUT / f'{ticker}_garch_model_comparison.csv'
    df_comp.to_csv(path, index=False)
    print(f'  ✓ {path.name}')

    # Print summary
    sep = '=' * 62
    print(f'\n{sep}')
    print(f'  GARCH SUITE — {ticker}  (n={len(returns)} daily returns)')
    print(sep)
    print(df_comp[['Model', 'AIC', 'BIC', 'Persistence', 'Half_life_d']].to_string(index=False))

    best_aic = df_comp.loc[df_comp['AIC'].idxmin(), 'Model']
    print(f'\n  Best by AIC: {best_aic}')

    # 1-day / 5-day forecasts
    print(f'\n  Volatility forecasts:')
    for name, res in results.items():
        v1, v5 = _vol_forecast_1d_5d(res, returns)
        print(f'    {name:15s}  1-day={v1:.2f}%  5-day={v5:.2f}%  annualised')

    # GJR-GARCH leverage effect comment
    if 'GJR-GARCH' in results:
        p = results['GJR-GARCH'].params
        gamma = p.get('gamma[1]', None)
        if gamma is not None:
            direction = 'positive' if float(gamma) > 0 else 'negative'
            print(f'\n  GJR-GARCH γ = {float(gamma):.4f} ({direction})')
            if float(gamma) > 0:
                print('    → Leverage effect confirmed: bad news raises vol more than good news.')
            else:
                print('    → No leverage effect (γ ≤ 0).')
    print(sep)

    # Plots
    plot_conditional_vol(results, returns, ticker)
    plot_model_comparison(df_comp, ticker)
    plot_forecast_strip(results, returns, ticker)
    plot_vs_realized(results, returns, ticker)


if __name__ == '__main__':
    ticker = sys.argv[1] if len(sys.argv) > 1 else 'NVDA'
    # Original educational GARCH(1,1) output
    import yfinance as yf
    data  = yf.download(ticker, period='3y', progress=False, auto_adjust=True)
    close = data['Close'].squeeze()
    r = GARCHModel().fit(close)
    p = r.params
    print(f"\n{ticker}  GARCH(1,1) [custom MLE]")
    print(f"  μ={p.mu*100:.5f}%  ω={p.omega:.2e}  α={p.alpha:.4f}  β={p.beta:.4f}")
    print(f"  α+β={r.persistence:.4f}  half-life={r.half_life_days:.1f}d"
          f"  LR-vol={r.long_run_vol_annual*100:.1f}%")
    print(f"  VaR 95%={r.var_1d_95:.3f}%  CVaR 95%={r.cvar_1d_95:.3f}%")
    print(f"  AIC={r.aic:.1f}  BIC={r.bic:.1f}")
    # Full suite
    run_suite(ticker)

