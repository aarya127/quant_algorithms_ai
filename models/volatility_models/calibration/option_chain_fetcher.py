"""
Option Chain Data Acquisition for Volatility Surface Construction

Phase 1: Data Acquisition (yfinance, QuantLib, SciPy, pandas)
Phase 2: Data Cleaning & IV Calculation
Phase 3: Surface Construction (log-moneyness, interpolation, SABR calibration)
Phase 4: Diagnostics & Visualization

Author: Stochastic Volatility Research Team
Date: February 15, 2026
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import QuantLib as ql
    QUANTLIB_AVAILABLE = True
except ImportError:
    print("⚠️  QuantLib not installed. Install with: pip install QuantLib")
    QUANTLIB_AVAILABLE = False

from scipy.interpolate import griddata, RBFInterpolator
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


class OptionChainFetcher:
    """
    Fetches and processes option chains for volatility surface construction.
    
    Follows the research-validated pipeline:
    1. Acquire raw option data
    2. Clean aggressively and compute implied volatilities
    3. Construct volatility surface (interpolation or SABR calibration)
    4. Generate diagnostics
    """
    
    def __init__(self, ticker: str, risk_free_rate: float = 0.05):
        """
        Initialize fetcher for a specific underlying.
        
        Args:
            ticker: Underlying ticker (e.g., 'SPY', '^SPX', 'NVDA')
            risk_free_rate: Risk-free rate for calculations (default 5%)
        """
        self.ticker = ticker
        self.risk_free_rate = risk_free_rate
        self.underlying = None
        self.spot_price = None
        self.raw_options = None
        self.cleaned_options = None
        self.surface_data = None
        
        print(f"🎯 Initialized OptionChainFetcher for {ticker}")
        print(f"   Risk-free rate: {risk_free_rate:.2%}")
    
    # ========================================================================
    # PHASE 1: DATA ACQUISITION
    # ========================================================================
    
    def fetch_option_chain(self) -> pd.DataFrame:
        """
        Phase 1: Fetch option chain from yfinance.
        
        Returns:
            DataFrame with raw option data
        """
        print(f"\n{'='*70}")
        print("PHASE 1: DATA ACQUISITION")
        print(f"{'='*70}")
        
        try:
            # Get underlying data
            self.underlying = yf.Ticker(self.ticker)
            
            # Get current spot price
            hist = self.underlying.history(period='1d')
            if hist.empty:
                raise ValueError(f"No price data available for {self.ticker}")
            
            self.spot_price = hist['Close'].iloc[-1]
            print(f"✓ Current spot price: ${self.spot_price:.2f}")
            
            # Get all available expiration dates
            expirations = self.underlying.options
            print(f"✓ Found {len(expirations)} expiration dates")
            
            # Fetch options for all expirations
            all_options = []
            
            for exp_date in expirations:
                try:
                    opt_chain = self.underlying.option_chain(exp_date)
                    
                    # Process calls
                    calls = opt_chain.calls.copy()
                    calls['optionType'] = 'call'
                    calls['expirationDate'] = exp_date
                    
                    # Process puts
                    puts = opt_chain.puts.copy()
                    puts['optionType'] = 'put'
                    puts['expirationDate'] = exp_date
                    
                    all_options.append(calls)
                    all_options.append(puts)
                    
                except Exception as e:
                    print(f"⚠️  Failed to fetch {exp_date}: {e}")
                    continue
            
            if not all_options:
                raise ValueError("No option data fetched")
            
            self.raw_options = pd.concat(all_options, ignore_index=True)
            
            print(f"✓ Fetched {len(self.raw_options):,} option contracts")
            print(f"  - Calls: {len(self.raw_options[self.raw_options['optionType']=='call']):,}")
            print(f"  - Puts: {len(self.raw_options[self.raw_options['optionType']=='put']):,}")
            
            return self.raw_options
            
        except Exception as e:
            print(f"❌ Error fetching option chain: {e}")
            raise
    
    # ========================================================================
    # PHASE 2: DATA CLEANING & IMPLIED VOLATILITY CALCULATION
    # ========================================================================
    
    def clean_and_compute_iv(self, 
                             min_volume: int = 10,
                             min_open_interest: int = 50,
                             max_bid_ask_spread_pct: float = 0.20,
                             min_moneyness: float = 0.7,
                             max_moneyness: float = 1.3) -> pd.DataFrame:
        """
        Phase 2: Aggressively clean data and compute implied volatilities.
        
        Cleaning steps:
        1. Remove options with low liquidity (volume, open interest)
        2. Remove wide bid-ask spreads
        3. Remove far OTM/ITM options (moneyness filter)
        4. Check for arbitrage violations (C >= max(S-K, 0))
        5. Extract forward price via put-call parity
        6. Compute implied volatility using QuantLib
        
        Args:
            min_volume: Minimum daily volume
            min_open_interest: Minimum open interest
            max_bid_ask_spread_pct: Maximum bid-ask spread as % of mid-price
            min_moneyness: Minimum strike/spot ratio
            max_moneyness: Maximum strike/spot ratio
            
        Returns:
            Cleaned DataFrame with implied volatilities
        """
        print(f"\n{'='*70}")
        print("PHASE 2: DATA CLEANING & IMPLIED VOLATILITY CALCULATION")
        print(f"{'='*70}")
        
        if self.raw_options is None:
            raise ValueError("Must fetch option chain first")
        
        df = self.raw_options.copy()
        initial_count = len(df)
        
        # Calculate mid-price
        df['mid_price'] = (df['bid'] + df['ask']) / 2
        df['bid_ask_spread'] = df['ask'] - df['bid']
        df['spread_pct'] = df['bid_ask_spread'] / df['mid_price']
        
        # Calculate time to maturity (in years)
        df['expirationDate'] = pd.to_datetime(df['expirationDate'])
        today = datetime.now()
        df['days_to_expiry'] = (df['expirationDate'] - today).dt.days
        df['time_to_maturity'] = df['days_to_expiry'] / 365.0
        
        # Calculate moneyness
        df['moneyness'] = df['strike'] / self.spot_price
        df['log_moneyness'] = np.log(df['moneyness'])
        
        print(f"\n📊 Initial dataset: {initial_count:,} contracts")
        
        # Filter 1: Remove expired options
        df = df[df['days_to_expiry'] > 0]
        print(f"✓ After removing expired: {len(df):,} ({100*len(df)/initial_count:.1f}%)")
        
        # Filter 2: Liquidity (volume)
        df = df[df['volume'].fillna(0) >= min_volume]
        print(f"✓ After volume filter (>={min_volume}): {len(df):,} ({100*len(df)/initial_count:.1f}%)")
        
        # Filter 3: Open interest
        df = df[df['openInterest'].fillna(0) >= min_open_interest]
        print(f"✓ After open interest filter (>={min_open_interest}): {len(df):,} ({100*len(df)/initial_count:.1f}%)")
        
        # Filter 4: Bid-ask spread
        df = df[df['spread_pct'] <= max_bid_ask_spread_pct]
        print(f"✓ After spread filter (<={max_bid_ask_spread_pct:.1%}): {len(df):,} ({100*len(df)/initial_count:.1f}%)")
        
        # Filter 5: Moneyness range
        df = df[(df['moneyness'] >= min_moneyness) & (df['moneyness'] <= max_moneyness)]
        print(f"✓ After moneyness filter ({min_moneyness:.1f}-{max_moneyness:.1f}): {len(df):,} ({100*len(df)/initial_count:.1f}%)")
        
        # Filter 6: Remove zero or negative prices
        df = df[df['mid_price'] > 0]
        print(f"✓ After positive price filter: {len(df):,} ({100*len(df)/initial_count:.1f}%)")
        
        # Filter 7: Arbitrage bounds
        # Call: C >= max(S - K*e^(-rT), 0)
        # Put: P >= max(K*e^(-rT) - S, 0)
        df['discount_factor'] = np.exp(-self.risk_free_rate * df['time_to_maturity'])
        df['pv_strike'] = df['strike'] * df['discount_factor']
        
        call_mask = df['optionType'] == 'call'
        put_mask = df['optionType'] == 'put'
        
        df.loc[call_mask, 'intrinsic_value'] = np.maximum(self.spot_price - df.loc[call_mask, 'pv_strike'], 0)
        df.loc[put_mask, 'intrinsic_value'] = np.maximum(df.loc[put_mask, 'pv_strike'] - self.spot_price, 0)
        
        df = df[df['mid_price'] >= df['intrinsic_value'] * 0.95]  # Allow 5% tolerance
        print(f"✓ After arbitrage check: {len(df):,} ({100*len(df)/initial_count:.1f}%)")
        
        # Extract forward price via put-call parity
        print(f"\n🔍 Extracting forward prices via put-call parity...")
        self._extract_forwards(df)
        
        # Compute implied volatilities
        if QUANTLIB_AVAILABLE:
            print(f"\n📈 Computing implied volatilities with QuantLib...")
            df['implied_volatility'] = df.apply(self._compute_iv_quantlib, axis=1)
        else:
            print(f"\n⚠️  Using yfinance implied volatility (less reliable)")
            df['implied_volatility'] = df['impliedVolatility']
        
        # Filter out failed IV calculations
        df = df[df['implied_volatility'].notna()]
        df = df[(df['implied_volatility'] > 0) & (df['implied_volatility'] < 3.0)]  # 0-300% vol
        print(f"✓ After IV computation: {len(df):,} ({100*len(df)/initial_count:.1f}%)")
        
        self.cleaned_options = df
        
        print(f"\n✅ Cleaning complete: {len(df):,} / {initial_count:,} contracts retained ({100*len(df)/initial_count:.1f}%)")
        
        return self.cleaned_options
    
    def _extract_forwards(self, df: pd.DataFrame):
        """
        Extract forward prices via put-call parity: C - P = S - K*e^(-rT)
        F = S * e^(rT) or F ≈ K + e^(rT) * (C - P)
        """
        forwards = []
        
        for exp_date in df['expirationDate'].unique():
            exp_df = df[df['expirationDate'] == exp_date]
            
            # Get ATM strike
            atm_strike = exp_df.iloc[(exp_df['strike'] - self.spot_price).abs().argsort()[:1]]['strike'].values[0]
            
            # Get call and put at ATM
            call_price = exp_df[(exp_df['strike'] == atm_strike) & (exp_df['optionType'] == 'call')]['mid_price']
            put_price = exp_df[(exp_df['strike'] == atm_strike) & (exp_df['optionType'] == 'put')]['mid_price']
            
            if len(call_price) > 0 and len(put_price) > 0:
                T = exp_df.iloc[0]['time_to_maturity']
                forward = atm_strike + np.exp(self.risk_free_rate * T) * (call_price.values[0] - put_price.values[0])
                forwards.append({
                    'expiration': exp_date,
                    'forward_price': forward,
                    'time_to_maturity': T
                })
        
        if forwards:
            forward_df = pd.DataFrame(forwards)
            print(f"✓ Extracted {len(forward_df)} forward prices:")
            for _, row in forward_df.iterrows():
                print(f"  T={row['time_to_maturity']:.2f}y → F=${row['forward_price']:.2f}")
    
    def _compute_iv_quantlib(self, row) -> Optional[float]:
        """
        Compute implied volatility using QuantLib Black-Scholes engine.
        
        More accurate than yfinance IV, especially for near-expiry options.
        """
        try:
            # Set up QuantLib date
            calculation_date = ql.Date.todaysDate()
            ql.Settings.instance().evaluationDate = calculation_date
            
            # Option parameters
            option_type = ql.Option.Call if row['optionType'] == 'call' else ql.Option.Put
            strike = row['strike']
            maturity_date = calculation_date + int(row['days_to_expiry'])
            
            # Market data
            spot = self.spot_price
            price = row['mid_price']
            
            # Black-Scholes process
            day_count = ql.Actual365Fixed()
            calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
            
            spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
            rate_handle = ql.YieldTermStructureHandle(
                ql.FlatForward(calculation_date, self.risk_free_rate, day_count)
            )
            dividend_handle = ql.YieldTermStructureHandle(
                ql.FlatForward(calculation_date, 0.0, day_count)
            )
            
            # European option
            payoff = ql.PlainVanillaPayoff(option_type, strike)
            exercise = ql.EuropeanExercise(maturity_date)
            option = ql.VanillaOption(payoff, exercise)
            
            # Try to compute IV
            try:
                iv = option.impliedVolatility(
                    price,
                    ql.BlackScholesMertonProcess(spot_handle, dividend_handle, rate_handle,
                                                  ql.BlackVolTermStructureHandle(
                                                      ql.BlackConstantVol(calculation_date, calendar, 0.20, day_count)
                                                  ))
                )
                return iv
            except:
                return None
                
        except Exception as e:
            return None
    
    # ========================================================================
    # PHASE 3: VOLATILITY SURFACE CONSTRUCTION
    # ========================================================================
    
    def construct_surface(self, method: str = 'interpolation', beta: float = 0.5) -> pd.DataFrame:
        """
        Phase 3: Construct volatility surface.
        
        Methods:
        1. 'interpolation': Convert to log-moneyness, interpolate total variance
        2. 'sabr': Calibrate SABR model per maturity slice
        
        Args:
            method: 'interpolation' or 'sabr'
            beta: Fixed beta for SABR (default 0.5 for interest rates, use 1.0 for equities)
            
        Returns:
            DataFrame with surface points (T, K, IV)
        """
        print(f"\n{'='*70}")
        print(f"PHASE 3: VOLATILITY SURFACE CONSTRUCTION (method={method})")
        print(f"{'='*70}")
        
        if self.cleaned_options is None:
            raise ValueError("Must clean data first")
        
        df = self.cleaned_options.copy()
        
        if method == 'interpolation':
            surface = self._construct_via_interpolation(df)
        elif method == 'sabr':
            surface = self._construct_via_sabr(df, beta)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.surface_data = surface
        
        print(f"\n✅ Surface constructed with {len(surface):,} points")
        print(f"   Maturity range: {surface['time_to_maturity'].min():.2f} - {surface['time_to_maturity'].max():.2f} years")
        print(f"   Strike range: ${surface['strike'].min():.2f} - ${surface['strike'].max():.2f}")
        print(f"   IV range: {surface['implied_volatility'].min():.2%} - {surface['implied_volatility'].max():.2%}")
        
        return surface
    
    def _construct_via_interpolation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Construct surface via log-moneyness and total variance interpolation.
        
        Steps:
        1. Convert to log-moneyness: k = log(K/F)
        2. Convert to total variance: w = σ² * T
        3. Interpolate w(k, T) using RBF or griddata
        4. Convert back to IV: σ = sqrt(w/T)
        """
        print("\n🔧 Using interpolation method...")
        
        # Calculate total variance
        df['total_variance'] = (df['implied_volatility'] ** 2) * df['time_to_maturity']
        
        # Prepare data for interpolation
        points = df[['log_moneyness', 'time_to_maturity']].values
        values = df['total_variance'].values
        
        # Create dense grid
        k_min, k_max = df['log_moneyness'].min(), df['log_moneyness'].max()
        T_min, T_max = df['time_to_maturity'].min(), df['time_to_maturity'].max()
        
        k_grid = np.linspace(k_min, k_max, 50)
        T_grid = np.linspace(T_min, T_max, 20)
        K_mesh, T_mesh = np.meshgrid(k_grid, T_grid)
        
        # Interpolate using RBF
        print("  Interpolating total variance surface...")
        grid_points = np.column_stack([K_mesh.ravel(), T_mesh.ravel()])
        
        try:
            # Try RBF first (smoother but slower)
            from scipy.interpolate import RBFInterpolator
            rbf = RBFInterpolator(points, values, kernel='thin_plate_spline')
            W_mesh = rbf(grid_points).reshape(K_mesh.shape)
        except:
            # Fallback to linear interpolation
            W_mesh = griddata(points, values, (K_mesh, T_mesh), method='linear')
        
        # Convert back to IV
        IV_mesh = np.sqrt(np.maximum(W_mesh / T_mesh, 0))  # Ensure non-negative
        
        # Create surface DataFrame
        surface = pd.DataFrame({
            'log_moneyness': K_mesh.ravel(),
            'time_to_maturity': T_mesh.ravel(),
            'total_variance': W_mesh.ravel(),
            'implied_volatility': IV_mesh.ravel()
        })
        
        # Convert back to strike
        surface['strike'] = self.spot_price * np.exp(surface['log_moneyness'])
        
        # Remove NaN values
        surface = surface.dropna()
        
        print(f"✓ Interpolation complete")
        
        return surface
    
    def _construct_via_sabr(self, df: pd.DataFrame, beta: float) -> pd.DataFrame:
        """
        Construct surface by calibrating SABR model per maturity slice.
        
        Uses smart initialization from diagnostics research.
        """
        print(f"\n🔧 Using SABR calibration method (β={beta})...")
        
        surface_points = []
        
        # Group by expiration
        for exp_date, group in df.groupby('expirationDate'):
            T = group.iloc[0]['time_to_maturity']
            
            if len(group) < 5:  # Need at least 5 points
                continue
            
            print(f"\n  Calibrating T={T:.2f}y ({len(group)} quotes)...")
            
            # Prepare market data
            strikes = group['strike'].values
            market_ivs = group['implied_volatility'].values
            
            # Smart initialization
            F0 = self.spot_price
            atm_iv = group.iloc[(group['moneyness'] - 1.0).abs().argsort()[:1]]['implied_volatility'].values[0]
            
            # Initial guess: α from ATM, ρ=-0.3 (typical for equities), ν=0.4
            alpha_init = atm_iv * (F0 ** (1 - beta))
            
            def sabr_objective(params):
                alpha, rho, nu = params
                error = 0
                for K, market_iv in zip(strikes, market_ivs):
                    model_iv = self._hagan_sabr_iv(F0, K, T, alpha, beta, rho, nu)
                    if model_iv is not None:
                        error += (model_iv - market_iv) ** 2
                return error
            
            # Calibrate
            bounds = [(0.001, 1.0), (-0.999, 0.999), (0.001, 2.0)]
            x0 = [alpha_init, -0.3, 0.4]
            
            result = minimize(sabr_objective, x0, bounds=bounds, method='L-BFGS-B')
            
            if result.success:
                alpha_opt, rho_opt, nu_opt = result.x
                print(f"    α={alpha_opt:.4f}, ρ={rho_opt:.3f}, ν={nu_opt:.3f}, RMSE={np.sqrt(result.fun/len(strikes)):.4f}")
                
                # Generate surface points for this maturity
                k_range = np.linspace(strikes.min(), strikes.max(), 30)
                for K in k_range:
                    iv = self._hagan_sabr_iv(F0, K, T, alpha_opt, beta, rho_opt, nu_opt)
                    if iv is not None:
                        surface_points.append({
                            'strike': K,
                            'time_to_maturity': T,
                            'implied_volatility': iv,
                            'log_moneyness': np.log(K / F0),
                            'sabr_alpha': alpha_opt,
                            'sabr_rho': rho_opt,
                            'sabr_nu': nu_opt
                        })
        
        surface = pd.DataFrame(surface_points)
        print(f"\n✓ SABR calibration complete for {len(df['expirationDate'].unique())} maturities")
        
        return surface
    
    def _hagan_sabr_iv(self, F: float, K: float, T: float, alpha: float, beta: float, rho: float, nu: float) -> Optional[float]:
        """
        Hagan et al. (2002) SABR implied volatility approximation.
        
        From diagnostics research - known to fail at "worst corner" (long T, low K).
        """
        try:
            if abs(F - K) < 1e-6:  # ATM
                return alpha / (F ** (1 - beta))
            
            # Log-moneyness
            log_FK = np.log(F / K)
            
            # z parameter
            z = (nu / alpha) * (F * K) ** ((1 - beta) / 2) * log_FK
            
            # x(z) function
            if abs(z) < 1e-6:
                x = 1.0
            else:
                sqrt_term = np.sqrt(1 - 2 * rho * z + z ** 2)
                x = np.log((sqrt_term + z - rho) / (1 - rho))
                if abs(x) < 1e-10:
                    x = 1.0
                else:
                    x = z / x
            
            # σ_B term
            sigma_B = (alpha / ((F * K) ** ((1 - beta) / 2))) * (log_FK / (1 + ((1 - beta) ** 2 / 24) * log_FK ** 2))
            
            # Correction terms
            term1 = ((1 - beta) ** 2 / 24) * (alpha ** 2 / ((F * K) ** (1 - beta)))
            term2 = (rho * beta * nu * alpha) / (4 * (F * K) ** ((1 - beta) / 2))
            term3 = ((2 - 3 * rho ** 2) / 24) * nu ** 2
            
            correction = 1 + (term1 + term2 + term3) * T
            
            iv = sigma_B * x * correction
            
            return iv if iv > 0 else None
            
        except:
            return None
    
    # ========================================================================
    # PHASE 4: DIAGNOSTICS & VISUALIZATION
    # ========================================================================
    
    def plot_2d_smile_diagnostics(self, save_path: Optional[str] = None):
        """
        Phase 4: 2D smile diagnostics across maturities.
        
        Plots implied volatility vs strike for multiple maturity slices.
        """
        print(f"\n{'='*70}")
        print("PHASE 4: DIAGNOSTICS & VISUALIZATION")
        print(f"{'='*70}")
        
        if self.cleaned_options is None:
            raise ValueError("No data to plot")
        
        df = self.cleaned_options.copy()
        
        # Select representative maturities
        unique_T = sorted(df['time_to_maturity'].unique())
        
        if len(unique_T) > 6:
            # Select 6 evenly spaced maturities
            indices = np.linspace(0, len(unique_T) - 1, 6, dtype=int)
            selected_T = [unique_T[i] for i in indices]
        else:
            selected_T = unique_T
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()
        
        for idx, T in enumerate(selected_T):
            if idx >= 6:
                break
            
            T_data = df[abs(df['time_to_maturity'] - T) < 0.01]
            
            if len(T_data) == 0:
                continue
            
            ax = axes[idx]
            
            # Plot calls and puts separately
            calls = T_data[T_data['optionType'] == 'call']
            puts = T_data[T_data['optionType'] == 'put']
            
            ax.scatter(calls['strike'], calls['implied_volatility'], c='blue', alpha=0.6, s=30, label='Calls')
            ax.scatter(puts['strike'], puts['implied_volatility'], c='red', alpha=0.6, s=30, label='Puts')
            
            # Mark ATM
            ax.axvline(self.spot_price, color='black', linestyle='--', alpha=0.5, label='ATM')
            
            ax.set_xlabel('Strike Price ($)', fontsize=10)
            ax.set_ylabel('Implied Volatility', fontsize=10)
            ax.set_title(f'T = {T:.2f} years', fontsize=12, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Format y-axis as percentage
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        plt.suptitle(f'Volatility Smile Diagnostics: {self.ticker}', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved 2D smile plot to {save_path}")
        
        plt.show()
    
    def plot_term_structure(self, strike_levels: List[float] = None, save_path: Optional[str] = None):
        """
        Plot volatility term structure for different strike levels.
        
        Shows how IV changes with maturity for fixed moneyness levels.
        """
        if self.cleaned_options is None:
            raise ValueError("No data to plot")
        
        df = self.cleaned_options.copy()
        
        # Default strike levels: 90%, 95%, 100%, 105%, 110% of spot
        if strike_levels is None:
            strike_levels = [0.90, 0.95, 1.00, 1.05, 1.10]
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(strike_levels)))
        
        for strike_pct, color in zip(strike_levels, colors):
            target_strike = self.spot_price * strike_pct
            
            # Find options near this strike
            term_data = []
            for T in sorted(df['time_to_maturity'].unique()):
                T_data = df[df['time_to_maturity'] == T]
                closest = T_data.iloc[(T_data['strike'] - target_strike).abs().argsort()[:1]]
                if len(closest) > 0:
                    term_data.append({
                        'time_to_maturity': T,
                        'implied_volatility': closest['implied_volatility'].values[0]
                    })
            
            if term_data:
                term_df = pd.DataFrame(term_data)
                ax.plot(term_df['time_to_maturity'], term_df['implied_volatility'], 
                       marker='o', linewidth=2, markersize=6, label=f'{strike_pct:.0%} Strike', color=color)
        
        ax.set_xlabel('Time to Maturity (years)', fontsize=12)
        ax.set_ylabel('Implied Volatility', fontsize=12)
        ax.set_title(f'Volatility Term Structure: {self.ticker}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved term structure plot to {save_path}")
        
        plt.show()
    
    def plot_3d_surface(self, save_path: Optional[str] = None):
        """
        Optional 3D visualization of the volatility surface.
        
        Shows IV as a function of strike and maturity.
        """
        if self.surface_data is None:
            raise ValueError("Must construct surface first")
        
        df = self.surface_data.copy()
        
        # Create meshgrid
        T_unique = sorted(df['time_to_maturity'].unique())
        K_unique = sorted(df['strike'].unique())
        
        T_grid = np.array([df[df['strike'] == K]['time_to_maturity'].values for K in K_unique])
        K_grid = np.array([df[df['strike'] == K]['strike'].values for K in K_unique])
        IV_grid = np.array([df[df['strike'] == K]['implied_volatility'].values for K in K_unique])
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(T_grid, K_grid, IV_grid, cmap='viridis', alpha=0.8, edgecolor='none')
        
        ax.set_xlabel('Time to Maturity (years)', fontsize=11)
        ax.set_ylabel('Strike Price ($)', fontsize=11)
        ax.set_zlabel('Implied Volatility', fontsize=11)
        ax.set_title(f'3D Volatility Surface: {self.ticker}', fontsize=14, fontweight='bold')
        
        # Format z-axis as percentage
        ax.zaxis.set_major_formatter(plt.FuncFormatter(lambda z, _: f'{z:.0%}'))
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved 3D surface plot to {save_path}")
        
        plt.show()
    
    def export_to_csv(self, filename: str):
        """
        Export cleaned option data to CSV for further analysis.
        """
        if self.cleaned_options is None:
            raise ValueError("No cleaned data to export")
        
        self.cleaned_options.to_csv(filename, index=False)
        print(f"✓ Exported {len(self.cleaned_options)} contracts to {filename}")


# ============================================================================
# CONVENIENCE FUNCTION: FULL PIPELINE
# ============================================================================

def run_full_pipeline(ticker: str, 
                      risk_free_rate: float = 0.05,
                      surface_method: str = 'interpolation',
                      beta: float = 1.0) -> OptionChainFetcher:
    """
    Run the complete 4-phase pipeline for a given ticker.
    
    Args:
        ticker: Underlying ticker (e.g., 'SPY', '^SPX', 'NVDA')
        risk_free_rate: Current risk-free rate
        surface_method: 'interpolation' or 'sabr'
        beta: SABR beta (0.5 for rates, 1.0 for equities)
        
    Returns:
        OptionChainFetcher instance with all data
    """
    print(f"\n{'#'*70}")
    print(f"# VOLATILITY SURFACE PIPELINE: {ticker}")
    print(f"{'#'*70}\n")
    
    # Initialize
    fetcher = OptionChainFetcher(ticker, risk_free_rate)
    
    # Phase 1: Fetch data
    fetcher.fetch_option_chain()
    
    # Phase 2: Clean and compute IV
    fetcher.clean_and_compute_iv()
    
    # Phase 3: Construct surface
    fetcher.construct_surface(method=surface_method, beta=beta)
    
    # Phase 4: Diagnostics
    fetcher.plot_2d_smile_diagnostics()
    fetcher.plot_term_structure()
    fetcher.plot_3d_surface()
    
    print(f"\n{'='*70}")
    print("✅ PIPELINE COMPLETE")
    print(f"{'='*70}\n")
    
    return fetcher


if __name__ == "__main__":
    # Example usage
    print("Option Chain Fetcher - Volatility Surface Construction")
    print("=" * 70)
    print("\nExample: Run full pipeline for SPY")
    print(">>> fetcher = run_full_pipeline('SPY', risk_free_rate=0.05, surface_method='interpolation', beta=1.0)")
    print("\nFor custom workflow:")
    print(">>> fetcher = OptionChainFetcher('SPY')")
    print(">>> fetcher.fetch_option_chain()")
    print(">>> fetcher.clean_and_compute_iv()")
    print(">>> fetcher.construct_surface(method='sabr', beta=0.5)")
    print(">>> fetcher.plot_2d_smile_diagnostics()")
