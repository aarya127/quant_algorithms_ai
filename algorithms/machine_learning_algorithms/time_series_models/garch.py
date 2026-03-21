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


if __name__ == '__main__':
    GARCHModel().run('NVDA')
