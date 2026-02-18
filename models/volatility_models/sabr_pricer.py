"""
SABR Model Volatility Calculation

Implements the Hagan approximation for SABR implied volatility.
This is a fast, accurate approximation that avoids solving SDEs.

Reference: Hagan et al. (2002) "Managing Smile Risk"
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class SABRParams:
    """SABR model parameters"""
    alpha: float  # Initial volatility level
    beta: float   # CEV exponent (fixed)
    rho: float    # Correlation
    nu: float     # Volatility of volatility


def sabr_implied_vol(F: float, K: float, T: float, params: SABRParams) -> float:
    """
    Calculate SABR implied volatility using Hagan approximation.
    
    Args:
        F: Forward price
        K: Strike price
        T: Time to expiration (years)
        params: SABR parameters
        
    Returns:
        Implied volatility (decimal, not percentage)
    """
    alpha = params.alpha
    beta = params.beta
    rho = params.rho
    nu = params.nu
    
    # Handle ATM case
    if abs(F - K) < 1e-10:
        return sabr_atm_vol(F, T, params)
    
    # Log-moneyness
    logFK = np.log(F / K)
    
    # FK midpoint raised to (1-beta)
    FK_mid = (F * K) ** ((1 - beta) / 2)
    
    if FK_mid < 1e-10:
        return 0.0
    
    # z parameter
    z = (nu / alpha) * FK_mid * logFK
    
    # x(z) function
    if abs(z) < 1e-10:
        x = 1.0
    else:
        sqrt_term = np.sqrt(1 - 2 * rho * z + z * z)
        numerator = sqrt_term + z - rho
        denominator = 1 - rho
        
        if abs(denominator) < 1e-10 or numerator <= 0:
            # Fallback for numerical issues
            x = 1.0
        else:
            x = np.log(numerator / denominator) / z
    
    # First term
    term1 = alpha / (FK_mid * (1 + ((1 - beta) ** 2 / 24) * (logFK ** 2) +
                                   ((1 - beta) ** 4 / 1920) * (logFK ** 4)))
    
    # Second term (time correction)
    term2_part1 = ((1 - beta) ** 2 / 24) * (alpha ** 2 / (FK_mid ** 2))
    term2_part2 = (rho * beta * nu * alpha) / (4 * FK_mid)
    term2_part3 = ((2 - 3 * rho ** 2) / 24) * (nu ** 2)
    
    term2 = 1 + (term2_part1 + term2_part2 + term2_part3) * T
    
    # Combined volatility
    vol = term1 * x * term2
    
    # Sanity check
    if vol <= 0 or np.isnan(vol) or np.isinf(vol):
        return 0.0
    
    return vol


def sabr_atm_vol(F: float, T: float, params: SABRParams) -> float:
    """
    Calculate SABR ATM implied volatility (simplified formula).
    
    Args:
        F: Forward price
        T: Time to expiration (years)
        params: SABR parameters
        
    Returns:
        ATM implied volatility
    """
    alpha = params.alpha
    beta = params.beta
    rho = params.rho
    nu = params.nu
    
    F_beta = F ** (1 - beta)
    
    if F_beta < 1e-10:
        return 0.0
    
    # ATM vol formula
    term1 = alpha / F_beta
    
    term2_part1 = ((1 - beta) ** 2 / 24) * (alpha ** 2 / (F ** (2 - 2 * beta)))
    term2_part2 = (rho * beta * nu * alpha) / (4 * F ** (1 - beta))
    term2_part3 = ((2 - 3 * rho ** 2) / 24) * (nu ** 2)
    
    term2 = 1 + (term2_part1 + term2_part2 + term2_part3) * T
    
    vol = term1 * term2
    
    if vol <= 0 or np.isnan(vol) or np.isinf(vol):
        return 0.0
    
    return vol


def price_sabr_option(S: float, K: float, T: float, r: float, 
                      params: SABRParams, option_type: str = 'call') -> float:
    """
    Price option using SABR implied vol + Black-Scholes.
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        params: SABR parameters
        option_type: 'call' or 'put'
        
    Returns:
        Option price
    """
    from scipy.stats import norm
    
    # Forward price
    F = S * np.exp(r * T)
    
    # SABR implied vol
    vol = sabr_implied_vol(F, K, T, params)
    
    if vol <= 0:
        # Intrinsic value only
        if option_type.lower() == 'call':
            return max(S - K * np.exp(-r * T), 0.0)
        else:
            return max(K * np.exp(-r * T) - S, 0.0)
    
    # Black-Scholes with SABR vol
    sqrt_T = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * vol ** 2 * T) / (vol * sqrt_T)
    d2 = d1 - vol * sqrt_T
    
    discount = np.exp(-r * T)
    
    if option_type.lower() == 'call':
        price = discount * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        price = discount * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    
    return price


if __name__ == "__main__":
    # Example usage
    print("SABR Implied Volatility Calculator")
    print("=" * 70)
    
    # Test parameters
    F = 100.0
    K = 100.0
    T = 1.0
    params = SABRParams(alpha=0.2, beta=1.0, rho=-0.3, nu=0.4)
    
    vol = sabr_implied_vol(F, K, T, params)
    print(f"\nATM implied vol: {vol:.4f} ({vol*100:.2f}%)")
    
    # Test smile
    print("\nVolatility Smile:")
    print("-" * 70)
    strikes = [80, 90, 95, 100, 105, 110, 120]
    for K in strikes:
        vol = sabr_implied_vol(F, K, T, params)
        moneyness = K / F
        print(f"K={K:6.1f}  (m={moneyness:.3f})  IV={vol:.4f} ({vol*100:.2f}%)")
