"""
Black-Scholes Option Greeks Calculator

Analytical Greeks for European options (no dividends or continuously-compounded
dividend yield q supported).

First-order Greeks
  Delta  — sensitivity to underlying price
  Gamma  — rate of change of delta
  Theta  — time decay (per calendar day)
  Vega   — sensitivity to implied volatility (per 1% vol move)
  Rho    — sensitivity to risk-free rate (per 1% rate move)

Second-order Greeks
  Charm  — delta bleed (d delta / d time, per day)
  Vanna  — d delta / d vol = d vega / d S
  Volga  — d vega / d vol  (vol convexity)
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import List, Literal


@dataclass
class OptionGreeks:
    price: float
    delta: float
    gamma: float
    theta: float    # per calendar day
    vega:  float    # per 1% vol move
    rho:   float    # per 1% rate move
    # Second-order
    charm: float    # d(delta)/d(time) per day
    vanna: float    # d(delta)/d(vol)
    volga: float    # d(vega)/d(vol) per 1% vol move


class BlackScholesGreeks:
    """
    Closed-form Black-Scholes Greeks for European options.

    Parameters
    ----------
    S     : spot / underlying price
    K     : strike price
    T     : time to expiration in years (must be > 0)
    r     : continuously compounded risk-free rate (e.g. 0.05 = 5%)
    sigma : implied volatility as a decimal (e.g. 0.20 = 20%)
    q     : continuous dividend yield (default 0.0)
    option_type : 'call' or 'put'
    """

    def __init__(
        self,
        S: float, K: float, T: float,
        r: float, sigma: float,
        q: float = 0.0,
        option_type: Literal['call', 'put'] = 'call',
    ):
        if T <= 0:
            raise ValueError("T must be positive")
        if sigma <= 0:
            raise ValueError("sigma must be positive")

        self.S, self.K, self.T = S, K, T
        self.r, self.sigma, self.q = r, sigma, q
        self.option_type = option_type.lower()

        self.d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        self.d2 = self.d1 - sigma * np.sqrt(T)
        self._phi1 = norm.pdf(self.d1)
        self._N1   = norm.cdf(self.d1)
        self._N2   = norm.cdf(self.d2)

    def price(self) -> float:
        S, K, T, r, q = self.S, self.K, self.T, self.r, self.q
        if self.option_type == 'call':
            return S * np.exp(-q * T) * self._N1 - K * np.exp(-r * T) * self._N2
        else:
            return (K * np.exp(-r * T) * norm.cdf(-self.d2)
                    - S * np.exp(-q * T) * norm.cdf(-self.d1))

    def delta(self) -> float:
        eqT = np.exp(-self.q * self.T)
        return eqT * self._N1 if self.option_type == 'call' else eqT * (self._N1 - 1)

    def gamma(self) -> float:
        return (np.exp(-self.q * self.T) * self._phi1
                / (self.S * self.sigma * np.sqrt(self.T)))

    def theta(self) -> float:
        """Per calendar day."""
        S, K, T, r, q = self.S, self.K, self.T, self.r, self.q
        base = -(S * np.exp(-q * T) * self._phi1 * self.sigma) / (2 * np.sqrt(T))
        if self.option_type == 'call':
            annual = base + q * S * np.exp(-q * T) * self._N1 - r * K * np.exp(-r * T) * self._N2
        else:
            annual = (base
                      - q * S * np.exp(-q * T) * norm.cdf(-self.d1)
                      + r * K * np.exp(-r * T) * norm.cdf(-self.d2))
        return annual / 365.0

    def vega(self) -> float:
        """Per 1% change in implied volatility."""
        return self.S * np.exp(-self.q * self.T) * self._phi1 * np.sqrt(self.T) / 100.0

    def rho(self) -> float:
        """Per 1% change in risk-free rate."""
        T, r, K = self.T, self.r, self.K
        if self.option_type == 'call':
            return  K * T * np.exp(-r * T) * self._N2 / 100.0
        else:
            return -K * T * np.exp(-r * T) * norm.cdf(-self.d2) / 100.0

    def charm(self) -> float:
        """Per calendar day."""
        d1, d2 = self.d1, self.d2
        T, r, q, sigma = self.T, self.r, self.q, self.sigma
        correction = np.exp(-q * T) * self._phi1 * (
            2 * (r - q) * T - d2 * sigma * np.sqrt(T)
        ) / (2 * T * sigma * np.sqrt(T))
        if self.option_type == 'call':
            return (-q * np.exp(-q * T) * self._N1 + correction) / 365.0
        else:
            return ( q * np.exp(-q * T) * norm.cdf(-d1) + correction) / 365.0

    def vanna(self) -> float:
        """d(delta)/d(vol) = vega/S * (1 - d1/(sigma*sqrt(T)))."""
        return (self.vega() * 100) * (1 - self.d1 / (self.sigma * np.sqrt(self.T))) / self.S

    def volga(self) -> float:
        """d(vega)/d(vol) per 1% move in vol."""
        return (self.vega() * 100) * self.d1 * self.d2 / self.sigma / 100.0

    def all_greeks(self) -> OptionGreeks:
        return OptionGreeks(
            price=self.price(), delta=self.delta(), gamma=self.gamma(),
            theta=self.theta(), vega=self.vega(), rho=self.rho(),
            charm=self.charm(), vanna=self.vanna(), volga=self.volga(),
        )


# ---------------------------------------------------------------------------
# Portfolio aggregation
# ---------------------------------------------------------------------------

@dataclass
class PortfolioPosition:
    S: float; K: float; T: float; r: float; sigma: float
    option_type: str
    quantity: float   # +ve = long, -ve = short
    q: float = 0.0


def portfolio_greeks(positions: List[PortfolioPosition]) -> dict:
    """Aggregate Greeks across a portfolio of European options."""
    totals = {k: 0.0 for k in ('price', 'delta', 'gamma', 'theta', 'vega', 'rho')}
    for pos in positions:
        bs = BlackScholesGreeks(pos.S, pos.K, pos.T, pos.r, pos.sigma, pos.q, pos.option_type)
        g = bs.all_greeks()
        for key in totals:
            totals[key] += getattr(g, key) * pos.quantity
    return totals


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # ATM call: S=K=100, 30 DTE, 20% IV, 5% rate
    bs = BlackScholesGreeks(S=100, K=100, T=30 / 365, r=0.05, sigma=0.20, option_type='call')
    g = bs.all_greeks()
    print(f"ATM Call  (S=100, K=100, T=30d, IV=20%, r=5%)")
    print(f"  Price : ${g.price:.4f}")
    print(f"  Delta : {g.delta:.4f}")
    print(f"  Gamma : {g.gamma:.4f}")
    print(f"  Theta : {g.theta:.4f}  (per day)")
    print(f"  Vega  : {g.vega:.4f}  (per 1% vol)")
    print(f"  Rho   : {g.rho:.4f}  (per 1% rate)")
    print(f"  Charm : {g.charm:.6f}  (per day)")
    print(f"  Vanna : {g.vanna:.4f}")
    print(f"  Volga : {g.volga:.4f}")
