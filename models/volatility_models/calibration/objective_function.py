"""
Objective Function for Volatility Model Calibration

Implements sophisticated objective functions with dual objectives:
1. Primary: RMSE in implied volatility
2. Secondary: Price-weighted RMSE

Why dual objectives matter:
- ATM volatilities dominate risk (highest vega)
- Deep OTM volatilities dominate tail hedging
- Pure vol RMSE can overweight noisy wings
- Price-weighting emphasizes liquid, important strikes

This approach is production-grade and used by trading desks.

Author: Volatility Research Team  
Date: February 15, 2026
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass
from scipy.optimize import minimize, differential_evolution


@dataclass
class CalibrationObjective:
    """
    Configuration for calibration objective function.
    
    Allows flexible weighting between vol-based and price-based errors.
    """
    vol_weight: float = 0.7        # Weight on vol RMSE (70%)
    price_weight: float = 0.3      # Weight on price RMSE (30%)
    vega_weighting: bool = True    # Weight by vega (emphasizes ATM)
    normalize_prices: bool = True   # Normalize prices by ATM price


@dataclass
class ModelParams:
    """Base class for model parameters"""
    pass


@dataclass
class SABRParams(ModelParams):
    """SABR model parameters"""
    alpha: float   # Initial volatility
    beta: float    # CEV exponent (typically fixed)
    rho: float     # Correlation
    nu: float      # Vol of vol
    
    def to_array(self) -> np.ndarray:
        """Convert to optimization array (excluding fixed beta)"""
        return np.array([self.alpha, self.rho, self.nu])
    
    @staticmethod
    def from_array(x: np.ndarray, beta: float = 1.0) -> 'SABRParams':
        """Create from optimization array"""
        return SABRParams(alpha=x[0], beta=beta, rho=x[1], nu=x[2])


@dataclass
class HestonParams(ModelParams):
    """Heston model parameters"""
    V0: float      # Initial variance
    kappa: float   # Mean reversion speed
    theta: float   # Long-term variance
    xi: float      # Vol of vol
    rho: float     # Correlation
    
    def to_array(self) -> np.ndarray:
        """Convert to optimization array"""
        return np.array([self.V0, self.kappa, self.theta, self.xi, self.rho])
    
    @staticmethod
    def from_array(x: np.ndarray) -> 'HestonParams':
        """Create from optimization array"""
        return HestonParams(V0=x[0], kappa=x[1], theta=x[2], xi=x[3], rho=x[4])
    
    def check_feller(self) -> float:
        """
        Check Feller condition: 2κθ >= ξ²
        
        Returns:
            Violation amount (0 if satisfied, positive if violated)
        """
        return max(0, self.xi**2 - 2*self.kappa*self.theta)


class ObjectiveFunction:
    """
    Sophisticated objective function for model calibration.
    
    Features:
    - Dual objectives (vol + price RMSE)
    - Vega weighting (emphasizes ATM)
    - Handles multiple maturities
    - Robust to outliers
    """
    
    def __init__(self,
                 market_data: pd.DataFrame,
                 model_pricer: Callable,
                 spot: float,
                 config: CalibrationObjective = None):
        """
        Initialize objective function.
        
        Args:
            market_data: DataFrame with columns:
                - strike, mid_price, implied_volatility
                - time_to_maturity, optionType
            model_pricer: Function(params, K, T, option_type) -> (price, iv)
            spot: Current spot price
            config: Calibration configuration
        """
        self.market_data = market_data
        self.model_pricer = model_pricer
        self.spot = spot
        self.config = config or CalibrationObjective()
        
        # Precompute weights
        self._compute_weights()
        
        print(f"✓ Initialized objective function")
        print(f"  Market quotes: {len(market_data)}")
        print(f"  Vol weight: {self.config.vol_weight:.1%}")
        print(f"  Price weight: {self.config.price_weight:.1%}")
        print(f"  Vega weighting: {self.config.vega_weighting}")
    
    def _compute_weights(self):
        """
        Precompute vega weights for market quotes.
        
        Vega is highest ATM and decays for OTM/ITM.
        This ensures ATM options (most important for hedging) get more weight.
        """
        if not self.config.vega_weighting:
            self.market_data['weight'] = 1.0
            return
        
        # Approximate vega ∝ S * sqrt(T) * φ(d1)
        # For simplicity: weight ∝ exp(-0.5 * (log(K/S))^2 / (σ²T))
        
        df = self.market_data.copy()
        
        log_moneyness = np.log(df['strike'] / self.spot)
        T = df['time_to_maturity']
        sigma = df['implied_volatility']
        
        # Gaussian kernel centered at ATM
        weights = np.exp(-0.5 * (log_moneyness**2) / (sigma**2 * T + 1e-6))
        
        # Normalize
        weights = weights / weights.sum() * len(weights)
        
        self.market_data['weight'] = weights
    
    def _vol_objective(self, model_ivs: np.ndarray) -> float:
        """
        Primary objective: RMSE in implied volatility.
        
        Standard approach - minimizes vol differences directly.
        """
        market_ivs = self.market_data['implied_volatility'].values
        weights = self.market_data['weight'].values
        
        # Weighted RMSE
        errors = (model_ivs - market_ivs) ** 2
        weighted_mse = np.average(errors, weights=weights)
        
        return np.sqrt(weighted_mse)
    
    def _price_objective(self, model_prices: np.ndarray) -> float:
        """
        Secondary objective: Price-weighted RMSE.
        
        Why this matters:
        - ATM options have highest dollar prices → naturally emphasized
        - Deep OTM have tiny prices → less impact even if vol error large
        - Better reflects actual P&L from calibration errors
        """
        market_prices = self.market_data['mid_price'].values
        weights = self.market_data['weight'].values
        
        if self.config.normalize_prices:
            # Normalize by ATM price to make comparable across strikes
            atm_idx = np.argmin(np.abs(self.market_data['strike'] - self.spot))
            atm_price = market_prices[atm_idx]
            
            market_prices = market_prices / atm_price
            model_prices = model_prices / atm_price
        
        # Weighted RMSE in prices
        errors = (model_prices - market_prices) ** 2
        weighted_mse = np.average(errors, weights=weights)
        
        return np.sqrt(weighted_mse)
    
    def evaluate(self, params: ModelParams) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate dual objective function.
        
        Args:
            params: Model parameters (SABR or Heston)
            
        Returns:
            total_error: Weighted combination of vol and price RMSE
            diagnostics: Dictionary with detailed error breakdown
        """
        # Generate model prices and IVs for all market quotes
        model_prices = []
        model_ivs = []
        
        for _, row in self.market_data.iterrows():
            try:
                price, iv = self.model_pricer(
                    params,
                    row['strike'],
                    row['time_to_maturity'],
                    row['optionType']
                )
                model_prices.append(price)
                model_ivs.append(iv)
            except:
                # If pricing fails, use market values (high penalty)
                model_prices.append(row['mid_price'] * 2.0)  # 100% error
                model_ivs.append(row['implied_volatility'] * 2.0)
        
        model_prices = np.array(model_prices)
        model_ivs = np.array(model_ivs)
        
        # Compute both objectives
        vol_error = self._vol_objective(model_ivs)
        price_error = self._price_objective(model_prices)
        
        # Weighted combination
        total_error = (self.config.vol_weight * vol_error + 
                      self.config.price_weight * price_error)
        
        diagnostics = {
            'total_error': total_error,
            'vol_rmse': vol_error,
            'price_rmse': price_error,
            'max_vol_error': np.max(np.abs(model_ivs - self.market_data['implied_volatility'].values)),
            'mean_vol_error': np.mean(np.abs(model_ivs - self.market_data['implied_volatility'].values)),
            'max_price_error_pct': np.max(np.abs(model_prices - self.market_data['mid_price'].values) / self.market_data['mid_price'].values) * 100
        }
        
        return total_error, diagnostics
    
    def __call__(self, x: np.ndarray, model_type: str = 'sabr', beta: float = 1.0) -> float:
        """
        Callable interface for optimization routines.
        
        Args:
            x: Parameter array
            model_type: 'sabr' or 'heston'
            beta: Fixed beta for SABR
            
        Returns:
            Objective value (scalar)
        """
        if model_type == 'sabr':
            params = SABRParams.from_array(x, beta)
        elif model_type == 'heston':
            params = HestonParams.from_array(x)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        total_error, _ = self.evaluate(params)
        
        return total_error


class RobustObjectiveFunction(ObjectiveFunction):
    """
    Robust objective function with outlier rejection.
    
    Extensions:
    - Huber loss (less sensitive to outliers)
    - Automatic outlier detection
    - Adaptive weighting
    """
    
    def __init__(self, *args, huber_delta: float = 0.05, **kwargs):
        """
        Args:
            huber_delta: Threshold for Huber loss (vol units)
        """
        super().__init__(*args, **kwargs)
        self.huber_delta = huber_delta
    
    def _huber_loss(self, errors: np.ndarray) -> np.ndarray:
        """
        Huber loss: quadratic for small errors, linear for large.
        
        More robust to outliers than pure squared error.
        """
        abs_errors = np.abs(errors)
        
        quadratic = 0.5 * errors**2
        linear = self.huber_delta * (abs_errors - 0.5 * self.huber_delta)
        
        return np.where(abs_errors <= self.huber_delta, quadratic, linear)
    
    def _vol_objective(self, model_ivs: np.ndarray) -> float:
        """Vol objective with Huber loss"""
        market_ivs = self.market_data['implied_volatility'].values
        weights = self.market_data['weight'].values
        
        errors = model_ivs - market_ivs
        losses = self._huber_loss(errors)
        
        weighted_loss = np.average(losses, weights=weights)
        
        return np.sqrt(weighted_loss * 2)  # Scale to approximate RMSE
    
    def _price_objective(self, model_prices: np.ndarray) -> float:
        """Price objective with Huber loss"""
        market_prices = self.market_data['mid_price'].values
        weights = self.market_data['weight'].values
        
        if self.config.normalize_prices:
            atm_idx = np.argmin(np.abs(self.market_data['strike'] - self.spot))
            atm_price = market_prices[atm_idx]
            market_prices = market_prices / atm_price
            model_prices = model_prices / atm_price
        
        errors = model_prices - market_prices
        losses = self._huber_loss(errors)
        
        weighted_loss = np.average(losses, weights=weights)
        
        return np.sqrt(weighted_loss * 2)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_sabr_objective(market_data: pd.DataFrame,
                         spot: float,
                         beta: float = 1.0,
                         config: CalibrationObjective = None) -> ObjectiveFunction:
    """
    Create objective function for SABR calibration.
    
    Args:
        market_data: Market option quotes
        spot: Current spot price
        beta: Fixed SABR beta
        config: Calibration configuration
        
    Returns:
        Configured objective function
    """
    from ..sabr_pricer import price_sabr_option  # Assumes this exists
    
    def model_pricer(params: SABRParams, K, T, opt_type):
        return price_sabr_option(spot, K, T, params, opt_type)
    
    return ObjectiveFunction(market_data, model_pricer, spot, config)


def create_heston_objective(market_data: pd.DataFrame,
                           spot: float,
                           config: CalibrationObjective = None) -> ObjectiveFunction:
    """
    Create objective function for Heston calibration.
    
    Args:
        market_data: Market option quotes
        spot: Current spot price
        config: Calibration configuration
        
    Returns:
        Configured objective function
    """
    from ..heston_pricer import price_heston_option  # Assumes this exists
    
    def model_pricer(params: HestonParams, K, T, opt_type):
        return price_heston_option(spot, K, T, params, opt_type)
    
    return ObjectiveFunction(market_data, model_pricer, spot, config)


def print_calibration_diagnostics(params: ModelParams,
                                  diagnostics: Dict[str, float]):
    """
    Print detailed calibration diagnostics.
    
    Useful for understanding fit quality and debugging.
    """
    print("\n" + "="*70)
    print("CALIBRATION DIAGNOSTICS")
    print("="*70)
    
    print(f"\n📊 Fit Quality:")
    print(f"  Total Error:      {diagnostics['total_error']:.6f}")
    print(f"  Vol RMSE:         {diagnostics['vol_rmse']:.4f} ({diagnostics['vol_rmse']*100:.2f}%)")
    print(f"  Price RMSE:       {diagnostics['price_rmse']:.4f}")
    print(f"  Max Vol Error:    {diagnostics['max_vol_error']:.4f} ({diagnostics['max_vol_error']*100:.2f}%)")
    print(f"  Mean Vol Error:   {diagnostics['mean_vol_error']:.4f} ({diagnostics['mean_vol_error']*100:.2f}%)")
    print(f"  Max Price Error:  {diagnostics['max_price_error_pct']:.2f}%")
    
    print(f"\n📈 Calibrated Parameters:")
    if isinstance(params, SABRParams):
        print(f"  α (alpha):  {params.alpha:.6f}")
        print(f"  β (beta):   {params.beta:.4f} [fixed]")
        print(f"  ρ (rho):    {params.rho:.4f}")
        print(f"  ν (nu):     {params.nu:.4f}")
    elif isinstance(params, HestonParams):
        print(f"  V₀:         {params.V0:.6f}")
        print(f"  κ (kappa):  {params.kappa:.4f}")
        print(f"  θ (theta):  {params.theta:.6f}")
        print(f"  ξ (xi):     {params.xi:.4f}")
        print(f"  ρ (rho):    {params.rho:.4f}")
        feller_violation = params.check_feller()
        if feller_violation > 0:
            print(f"  ⚠️  Feller condition violated by {feller_violation:.6f}")
        else:
            print(f"  ✓ Feller condition satisfied")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    print("Objective Function Module for Volatility Model Calibration")
    print("=" * 70)
    print("\nKey Features:")
    print("✓ Dual objectives (vol + price RMSE)")
    print("✓ Vega weighting (emphasizes ATM)")
    print("✓ Robust to outliers (Huber loss)")
    print("✓ Production-grade design")
    print("\nUsage:")
    print(">>> from models.volatility_models.calibration.objective_function import create_sabr_objective")
    print(">>> obj_func = create_sabr_objective(market_data, spot=100, beta=1.0)")
    print(">>> error, diagnostics = obj_func.evaluate(params)")
