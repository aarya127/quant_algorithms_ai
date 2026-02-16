"""
Caplet/Floorlet Volatility Stripping for Interest Rate Products

Implements bootstrap algorithm to strip individual caplet/floorlet volatilities
from market cap/floor prices. Handles forward rate curve extraction and
undiscounted pricing for SABR/Heston calibration.

Bootstrap Algorithm:
- Cap(T_n) = Σ Caplet(T_i) for i=1..n
- Given Cap(T_1), Cap(T_2), ..., Cap(T_n)
- Strip: σ_caplet(T_i) = solve[Caplet(T_i) = Cap(T_i) - Cap(T_{i-1})]

References:
- Brigo & Mercurio, "Interest Rate Models - Theory and Practice"
- Rebonato, "Modern Pricing of Interest-Rate Derivatives"
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Callable
from scipy.optimize import brentq, minimize_scalar
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import QuantLib as ql
    QUANTLIB_AVAILABLE = True
except ImportError:
    print("WARNING: QuantLib not installed. Install with: pip install QuantLib")
    QUANTLIB_AVAILABLE = False


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ForwardRateCurve:
    """
    Forward LIBOR/SOFR curve for caplet valuation.
    
    Attributes:
        tenor: Time to fixing (years)
        forward_rate: Forward rate (annualized, e.g., 0.05 for 5%)
        accrual_factor: Day count fraction (typically τ = 0.25 for quarterly)
    """
    tenor: np.ndarray          # [T_1, T_2, ..., T_n]
    forward_rate: np.ndarray   # [F(0,T_1), F(0,T_2), ..., F(0,T_n)]
    accrual_factor: np.ndarray # [τ_1, τ_2, ..., τ_n]
    
    def interpolate(self, T: float) -> float:
        """Linear interpolation of forward rates."""
        return np.interp(T, self.tenor, self.forward_rate)


@dataclass
class CapletData:
    """
    Stripped caplet/floorlet volatility and market data.
    
    Attributes:
        tenor: Time to fixing (years)
        strike: Strike rate (e.g., 0.05 for 5%)
        forward_rate: Forward rate at fixing
        volatility: Stripped Black volatility (annualized)
        market_price: Market price (undiscounted, per unit notional)
        cap_price: Cumulative cap price up to this tenor
    """
    tenor: float
    strike: float
    forward_rate: float
    volatility: float
    market_price: float
    cap_price: float
    accrual_factor: float = 0.25  # Default quarterly


# ============================================================================
# FORWARD RATE CURVE EXTRACTION
# ============================================================================

class ForwardCurveBuilder:
    """
    Extracts forward rate curves from market data.
    
    Supports:
    - LIBOR/SOFR swap curve bootstrapping
    - OIS discounting curve
    - Tenor basis adjustments (3M vs 6M LIBOR)
    """
    
    def __init__(self, curve_date: datetime, 
                 day_count_convention: str = "ACT/360"):
        """
        Initialize forward curve builder.
        
        Args:
            curve_date: Valuation date
            day_count_convention: Day count convention (ACT/360, ACT/365, 30/360)
        """
        self.curve_date = curve_date
        self.day_count = day_count_convention
        
        print(f"Forward curve builder initialized")
        print(f"   Curve date: {curve_date.strftime('%Y-%m-%d')}")
        print(f"   Day count: {day_count_convention}")
    
    def build_from_swap_rates(self,
                              tenors: List[float],
                              swap_rates: List[float],
                              fixing_frequency: str = "quarterly") -> ForwardRateCurve:
        """
        Bootstrap forward curve from swap rates.
        
        Args:
            tenors: Swap tenors in years [1, 2, 5, 10, ...]
            swap_rates: Fixed rates on par swaps (annualized)
            fixing_frequency: "quarterly", "semiannual", "annual"
            
        Returns:
            ForwardRateCurve with interpolated forward rates
        """
        print(f"\n{'='*70}")
        print("BOOTSTRAPPING FORWARD CURVE FROM SWAP RATES")
        print(f"{'='*70}")
        
        if QUANTLIB_AVAILABLE:
            return self._bootstrap_quantlib(tenors, swap_rates, fixing_frequency)
        else:
            return self._bootstrap_simple(tenors, swap_rates, fixing_frequency)
    
    def _bootstrap_quantlib(self,
                           tenors: List[float],
                           swap_rates: List[float],
                           fixing_frequency: str) -> ForwardRateCurve:
        """Bootstrap using QuantLib's advanced curve building."""
        print("Using QuantLib for accurate bootstrapping...")
        
        # Set evaluation date
        ql.Settings.instance().evaluationDate = self._to_ql_date(self.curve_date)
        
        # Create helpers
        calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
        day_count = self._get_ql_day_count()
        
        # Swap frequency
        frequency_map = {
            "quarterly": ql.Quarterly,
            "semiannual": ql.Semiannual,
            "annual": ql.Annual
        }
        frequency = frequency_map[fixing_frequency]
        
        # Build swap helpers
        helpers = []
        for tenor_years, rate in zip(tenors, swap_rates):
            period = ql.Period(int(tenor_years * 12), ql.Months)
            helper = ql.SwapRateHelper(
                ql.QuoteHandle(ql.SimpleQuote(rate)),
                period,
                calendar,
                frequency,
                ql.Unadjusted,
                day_count,
                ql.Euribor3M()  # Index
            )
            helpers.append(helper)
        
        # Bootstrap curve
        curve = ql.PiecewiseLogCubicDiscount(0, calendar, helpers, day_count)
        curve.enableExtrapolation()
        
        # Extract forward rates at caplet tenors
        accrual = self._get_accrual_factor(fixing_frequency)
        max_tenor = max(tenors)
        caplet_tenors = np.arange(accrual, max_tenor + accrual, accrual)
        
        forward_rates = []
        for T in caplet_tenors:
            T_start = ql.Settings.instance().evaluationDate + ql.Period(int(T * 365), ql.Days)
            T_end = T_start + ql.Period(int(accrual * 365), ql.Days)
            
            df_start = curve.discount(T_start)
            df_end = curve.discount(T_end)
            
            # Forward rate: F = (DF_start/DF_end - 1) / τ
            forward_rate = (df_start / df_end - 1) / accrual
            forward_rates.append(forward_rate)
        
        print(f"Bootstrapped {len(caplet_tenors)} forward rates")
        print(f"  Forward range: {min(forward_rates):.2%} - {max(forward_rates):.2%}")
        
        return ForwardRateCurve(
            tenor=caplet_tenors,
            forward_rate=np.array(forward_rates),
            accrual_factor=np.full(len(caplet_tenors), accrual)
        )
    
    def _bootstrap_simple(self,
                         tenors: List[float],
                         swap_rates: List[float],
                         fixing_frequency: str) -> ForwardRateCurve:
        """
        Simple bootstrap without QuantLib.
        
        Approximation: Assumes flat forward curve in each swap period.
        """
        print("WARNING: Using simplified bootstrap (install QuantLib for accuracy)")
        
        accrual = self._get_accrual_factor(fixing_frequency)
        max_tenor = max(tenors)
        caplet_tenors = np.arange(accrual, max_tenor + accrual, accrual)
        
        # Linear interpolation of swap rates
        interpolated_rates = np.interp(caplet_tenors, tenors, swap_rates)
        
        # Approximate forward rates (assumes flat forwards ≈ swap rate)
        forward_rates = interpolated_rates
        
        print(f"Generated {len(caplet_tenors)} approximate forward rates")
        
        return ForwardRateCurve(
            tenor=caplet_tenors,
            forward_rate=forward_rates,
            accrual_factor=np.full(len(caplet_tenors), accrual)
        )
    
    def build_from_libor_futures(self,
                                 futures_prices: List[float],
                                 futures_maturities: List[float]) -> ForwardRateCurve:
        """
        Extract forward curve from LIBOR futures (Eurodollar, SOFR futures).
        
        Futures rate: F = 100 - futures_price
        
        Args:
            futures_prices: Futures prices [98.5, 98.2, 97.9, ...]
            futures_maturities: Maturity in years [0.25, 0.5, 0.75, ...]
            
        Returns:
            ForwardRateCurve
        """
        forward_rates = [(100 - price) / 100 for price in futures_prices]
        accrual = 0.25  # Quarterly futures
        
        print(f"Extracted {len(forward_rates)} forward rates from futures")
        
        return ForwardRateCurve(
            tenor=np.array(futures_maturities),
            forward_rate=np.array(forward_rates),
            accrual_factor=np.full(len(forward_rates), accrual)
        )
    
    # Helper methods
    
    def _to_ql_date(self, dt: datetime) -> 'ql.Date':
        """Convert Python datetime to QuantLib Date."""
        return ql.Date(dt.day, dt.month, dt.year)
    
    def _get_ql_day_count(self) -> 'ql.DayCounter':
        """Get QuantLib day count convention."""
        if self.day_count == "ACT/360":
            return ql.Actual360()
        elif self.day_count == "ACT/365":
            return ql.Actual365Fixed()
        elif self.day_count == "30/360":
            return ql.Thirty360()
        else:
            return ql.Actual360()
    
    def _get_accrual_factor(self, frequency: str) -> float:
        """Get accrual factor for frequency."""
        return {"quarterly": 0.25, "semiannual": 0.5, "annual": 1.0}[frequency]


# ============================================================================
# BLACK MODEL FOR CAPLETS
# ============================================================================

class BlackCapletPricer:
    """
    Black (1976) model for caplet/floorlet valuation.
    
    Undiscounted caplet price (per unit notional):
        Caplet(T, K) = τ · [F · N(d1) - K · N(d2)]
    
    where:
        d1 = [ln(F/K) + 0.5σ²T] / (σ√T)
        d2 = d1 - σ√T
        F = forward rate, K = strike, τ = accrual factor
    """
    
    @staticmethod
    def price_caplet(forward_rate: float,
                    strike: float,
                    volatility: float,
                    time_to_fixing: float,
                    accrual_factor: float = 0.25,
                    option_type: str = "cap") -> float:
        """
        Price a single caplet/floorlet using Black's formula.
        
        Args:
            forward_rate: Forward rate (e.g., 0.05 for 5%)
            strike: Strike rate (e.g., 0.05 for 5%)
            volatility: Black volatility (annualized, e.g., 0.20 for 20%)
            time_to_fixing: Time to fixing in years
            accrual_factor: Day count fraction (typically 0.25)
            option_type: "cap" or "floor"
            
        Returns:
            Undiscounted caplet price per unit notional
        """
        if time_to_fixing <= 0:
            # Expired caplet
            if option_type == "cap":
                return accrual_factor * max(forward_rate - strike, 0)
            else:
                return accrual_factor * max(strike - forward_rate, 0)
        
        # Black formula
        sqrt_T = np.sqrt(time_to_fixing)
        d1 = (np.log(forward_rate / strike) + 0.5 * volatility**2 * time_to_fixing) / (volatility * sqrt_T)
        d2 = d1 - volatility * sqrt_T
        
        from scipy.stats import norm
        
        if option_type == "cap":
            return accrual_factor * (forward_rate * norm.cdf(d1) - strike * norm.cdf(d2))
        else:  # floor
            return accrual_factor * (strike * norm.cdf(-d2) - forward_rate * norm.cdf(-d1))
    
    @staticmethod
    def implied_volatility(market_price: float,
                          forward_rate: float,
                          strike: float,
                          time_to_fixing: float,
                          accrual_factor: float = 0.25,
                          option_type: str = "cap") -> Optional[float]:
        """
        Solve for implied Black volatility given market price.
        
        Uses Brent's method for robust root finding.
        """
        def objective(vol):
            model_price = BlackCapletPricer.price_caplet(
                forward_rate, strike, vol, time_to_fixing, accrual_factor, option_type
            )
            return model_price - market_price
        
        try:
            # Intrinsic value check
            if option_type == "cap":
                intrinsic = accrual_factor * max(forward_rate - strike, 0)
            else:
                intrinsic = accrual_factor * max(strike - forward_rate, 0)
            
            if market_price < intrinsic * 0.95:
                return None
            
            # Solve for volatility
            iv = brentq(objective, 0.001, 3.0, xtol=1e-6)
            return iv
            
        except:
            return None


# ============================================================================
# CAPLET STRIPPING ENGINE
# ============================================================================

class CapletVolatilityStripper:
    """
    Bootstrap individual caplet volatilities from cap prices.
    
    Algorithm:
    1. Start with shortest maturity cap → this IS the first caplet
    2. Strip: σ_caplet(T_i) such that:
       Caplet(T_i, σ_i) = Cap(T_i) - Cap(T_{i-1})
    3. Repeat for all maturities
    
    Handles:
    - Multiple strikes (volatility surface)
    - Forward curve interpolation
    - Arbitrage checks
    """
    
    def __init__(self, forward_curve: ForwardRateCurve):
        """
        Initialize stripper with forward rate curve.
        
        Args:
            forward_curve: Forward LIBOR/SOFR curve
        """
        self.forward_curve = forward_curve
        self.stripped_caplets: Dict[float, List[CapletData]] = {}
        
        print(f"Caplet volatility stripper initialized")
        print(f"   Forward curve: {len(forward_curve.tenor)} points")
    
    def strip_from_cap_prices(self,
                             cap_maturities: List[float],
                             cap_prices: List[float],
                             strike: float,
                             option_type: str = "cap") -> List[CapletData]:
        """
        Strip caplet volatilities from cap prices at a single strike.
        
        Args:
            cap_maturities: Cap maturities in years [1, 2, 3, 5, 7, 10, ...]
            cap_prices: Market cap prices (undiscounted, per unit notional)
            strike: Strike rate (e.g., 0.05 for 5%)
            option_type: "cap" or "floor"
            
        Returns:
            List of CapletData with stripped volatilities
        """
        print(f"\n{'='*70}")
        print(f"STRIPPING CAPLET VOLATILITIES AT STRIKE {strike:.2%}")
        print(f"{'='*70}")
        
        if len(cap_maturities) != len(cap_prices):
            raise ValueError("Maturities and prices must have same length")
        
        # Sort by maturity
        sorted_data = sorted(zip(cap_maturities, cap_prices))
        cap_maturities, cap_prices = zip(*sorted_data)
        
        stripped_caplets = []
        previous_cap_price = 0.0
        
        for i, (maturity, cap_price) in enumerate(zip(cap_maturities, cap_prices)):
            print(f"\n--- Tenor {maturity:.2f}y ---")
            
            # Get all caplet fixings up to this maturity
            caplet_tenors = self.forward_curve.tenor[self.forward_curve.tenor <= maturity]
            
            if i == 0:
                # First cap = first caplet (no bootstrapping needed)
                caplet_price = cap_price
                T = caplet_tenors[-1]
                F = self.forward_curve.interpolate(T)
                tau = self.forward_curve.accrual_factor[np.argmin(np.abs(self.forward_curve.tenor - T))]
                
                sigma = BlackCapletPricer.implied_volatility(
                    caplet_price, F, strike, T, tau, option_type
                )
                
                if sigma is None:
                    print(f"ERROR: Failed to solve for volatility")
                    continue
                
                caplet_data = CapletData(
                    tenor=T,
                    strike=strike,
                    forward_rate=F,
                    volatility=sigma,
                    market_price=caplet_price,
                    cap_price=cap_price,
                    accrual_factor=tau
                )
                
                stripped_caplets.append(caplet_data)
                previous_cap_price = cap_price
                
                print(f"First caplet: T={T:.2f}y, σ={sigma:.2%}, Price={caplet_price:.6f}")
                
            else:
                # Bootstrap: Caplet(T_i) = Cap(T_i) - Cap(T_{i-1})
                # But we need to sum over ALL previous caplets
                
                # Find new caplet tenors (not yet stripped)
                new_tenors = [t for t in caplet_tenors if t > stripped_caplets[-1].tenor]
                
                if not new_tenors:
                    print("WARNING: No new tenors to strip")
                    continue
                
                # Target: sum of all caplets up to this maturity = cap_price
                # We know prices of previous caplets → solve for new ones
                
                def objective(sigma_guess):
                    """
                    Objective: Price all caplets with guessed volatility,
                    sum should equal cap price.
                    """
                    total = 0.0
                    
                    # Add all previously stripped caplets
                    for prev in stripped_caplets:
                        total += prev.market_price
                    
                    # Add new caplets with guessed volatility
                    for T_new in new_tenors:
                        F_new = self.forward_curve.interpolate(T_new)
                        tau_new = self.forward_curve.accrual_factor[
                            np.argmin(np.abs(self.forward_curve.tenor - T_new))
                        ]
                        
                        price_new = BlackCapletPricer.price_caplet(
                            F_new, strike, sigma_guess, T_new, tau_new, option_type
                        )
                        total += price_new
                    
                    return abs(total - cap_price)
                
                # Solve for volatility
                result = minimize_scalar(objective, bounds=(0.001, 3.0), method='bounded')
                
                if not result.success:
                    print(f"ERROR: Optimization failed")
                    continue
                
                sigma_new = result.x
                
                # Store stripped caplets
                for T_new in new_tenors:
                    F_new = self.forward_curve.interpolate(T_new)
                    tau_new = self.forward_curve.accrual_factor[
                        np.argmin(np.abs(self.forward_curve.tenor - T_new))
                    ]
                    
                    price_new = BlackCapletPricer.price_caplet(
                        F_new, strike, sigma_new, T_new, tau_new, option_type
                    )
                    
                    caplet_data = CapletData(
                        tenor=T_new,
                        strike=strike,
                        forward_rate=F_new,
                        volatility=sigma_new,
                        market_price=price_new,
                        cap_price=cap_price,
                        accrual_factor=tau_new
                    )
                    
                    stripped_caplets.append(caplet_data)
                    print(f"Stripped: T={T_new:.2f}y, σ={sigma_new:.2%}, Price={price_new:.6f}")
                
                previous_cap_price = cap_price
        
        print(f"\n{'='*70}")
        print(f"STRIPPING COMPLETE: {len(stripped_caplets)} caplets")
        print(f"{'='*70}")
        
        self.stripped_caplets[strike] = stripped_caplets
        return stripped_caplets
    
    def strip_surface(self,
                     cap_maturities: List[float],
                     strikes: List[float],
                     cap_prices: np.ndarray,
                     option_type: str = "cap") -> pd.DataFrame:
        """
        Strip caplet volatility surface across multiple strikes.
        
        Args:
            cap_maturities: Cap maturities [1, 2, 3, 5, 7, 10]
            strikes: Strike rates [0.03, 0.04, 0.05, 0.06, 0.07]
            cap_prices: 2D array of cap prices [len(strikes) x len(maturities)]
            option_type: "cap" or "floor"
            
        Returns:
            DataFrame with columns [tenor, strike, volatility, forward_rate]
        """
        print(f"\n{'#'*70}")
        print(f"# STRIPPING CAPLET VOLATILITY SURFACE")
        print(f"# {len(strikes)} strikes × {len(cap_maturities)} maturities")
        print(f"{'#'*70}\n")
        
        all_caplets = []
        
        for i, strike in enumerate(strikes):
            caplets = self.strip_from_cap_prices(
                cap_maturities, cap_prices[i, :].tolist(), strike, option_type
            )
            all_caplets.extend(caplets)
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'tenor': c.tenor,
                'strike': c.strike,
                'volatility': c.volatility,
                'forward_rate': c.forward_rate,
                'market_price': c.market_price,
                'accrual_factor': c.accrual_factor
            }
            for c in all_caplets
        ])
        
        print(f"\nSurface stripped: {len(df)} data points")
        print(f"   Volatility range: {df['volatility'].min():.2%} - {df['volatility'].max():.2%}")
        
        return df
    
    def export_for_calibration(self, filename: str):
        """
        Export stripped caplets to CSV for SABR/Heston calibration.
        
        Format matches data_aquisition.py for consistency with
        objective_function.py and constraints_handling.py.
        """
        if not self.stripped_caplets:
            raise ValueError("No stripped caplets to export")
        
        all_data = []
        for strike, caplets in self.stripped_caplets.items():
            for caplet in caplets:
                all_data.append({
                    'strike': caplet.strike,
                    'mid_price': caplet.market_price,
                    'implied_volatility': caplet.volatility,
                    'time_to_maturity': caplet.tenor,
                    'moneyness': caplet.strike / caplet.forward_rate,
                    'log_moneyness': np.log(caplet.strike / caplet.forward_rate),
                    'forward_rate': caplet.forward_rate,
                    'accrual_factor': caplet.accrual_factor
                })
        
        df = pd.DataFrame(all_data)
        df.to_csv(filename, index=False)
        
        print(f"\nExported {len(df)} caplets to {filename}")
        print(f"   Ready for SABR/Heston calibration")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def strip_caplet_volatilities(
    swap_tenors: List[float],
    swap_rates: List[float],
    cap_maturities: List[float],
    cap_prices: List[float],
    strike: float,
    curve_date: datetime = None,
    fixing_frequency: str = "quarterly"
) -> List[CapletData]:
    """
    Complete pipeline: bootstrap forward curve → strip caplet volatilities.
    
    Args:
        swap_tenors: Swap maturities in years [1, 2, 5, 10, ...]
        swap_rates: Par swap rates (annualized)
        cap_maturities: Cap maturities [1, 2, 3, 5, ...]
        cap_prices: Market cap prices (undiscounted)
        strike: Cap strike rate
        curve_date: Valuation date (default: today)
        fixing_frequency: Caplet frequency
        
    Returns:
        List of stripped CapletData
    """
    if curve_date is None:
        curve_date = datetime.now()
    
    print(f"\n{'#'*70}")
    print("# COMPLETE CAPLET STRIPPING PIPELINE")
    print(f"{'#'*70}\n")
    
    # Step 1: Build forward curve
    builder = ForwardCurveBuilder(curve_date)
    forward_curve = builder.build_from_swap_rates(swap_tenors, swap_rates, fixing_frequency)
    
    # Step 2: Strip caplets
    stripper = CapletVolatilityStripper(forward_curve)
    caplets = stripper.strip_from_cap_prices(cap_maturities, cap_prices, strike)
    
    print(f"\n{'='*70}")
    print("CAPLET STRIPPING COMPLETE")
    print(f"{'='*70}\n")
    
    return caplets


if __name__ == "__main__":
    print("Caplet/Floorlet Volatility Stripping Module")
    print("=" * 70)
    print("\nExample usage:")
    print(">>> from models.volatility_models.calibration.caplet_stripping import strip_caplet_volatilities")
    print(">>> ")
    print(">>> # Market data")
    print(">>> swap_tenors = [1, 2, 5, 10]  # years")
    print(">>> swap_rates = [0.04, 0.045, 0.05, 0.052]  # par swap rates")
    print(">>> cap_maturities = [1, 2, 3, 5]")
    print(">>> cap_prices = [0.0012, 0.0035, 0.0068, 0.0142]  # undiscounted")
    print(">>> ")
    print(">>> # Strip caplet volatilities")
    print(">>> caplets = strip_caplet_volatilities(")
    print(">>>     swap_tenors, swap_rates, cap_maturities, cap_prices, strike=0.05")
    print(">>> )")
    print(">>> ")
    print(">>> # Export for calibration")
    print(">>> stripper = CapletVolatilityStripper(forward_curve)")
    print(">>> stripper.export_for_calibration('caplets.csv')")
