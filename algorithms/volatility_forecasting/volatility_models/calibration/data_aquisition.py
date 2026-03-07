"""
Public Option Chain Data Acquisition

Fetches option chain data from public sources (yfinance, CBOE) and processes
it for volatility surface calibration.

Phase 1: Raw data acquisition (yfinance)
Phase 2: Aggressive cleaning & IV calculation (QuantLib)
Phase 3: Forward extraction via put-call parity
Phase 4: Export for calibration pipeline
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
    print("WARNING: QuantLib not installed. Install with: pip install QuantLib")
    QUANTLIB_AVAILABLE = False


class OptionChainDataAcquisition:
    """
    Handles raw option chain data acquisition and preprocessing.
    
    Focuses purely on data fetching, cleaning, and IV calculation.
    Does NOT handle calibration or optimization (see objective_function.py).
    """
    
    def __init__(self, ticker: str, risk_free_rate: float = 0.05):
        """
        Initialize data acquisition for a specific ticker.
        
        Args:
            ticker: Underlying ticker (e.g., 'SPY', '^SPX', 'NVDA')
            risk_free_rate: Risk-free rate for IV calculations
        """
        self.ticker = ticker
        self.risk_free_rate = risk_free_rate
        self.underlying = None
        self.spot_price = None
        self.raw_options = None
        self.cleaned_options = None
        self.forward_prices = {}
        
        print(f"Initialized data acquisition for {ticker}")
        print(f"   Risk-free rate: {risk_free_rate:.2%}")
    
    # ========================================================================
    # PHASE 1: RAW DATA ACQUISITION
    # ========================================================================
    
    def fetch_option_chain(self) -> pd.DataFrame:
        """
        Fetch raw option chain from yfinance.
        
        Returns:
            DataFrame with raw option data including:
            - strike, bid, ask, last, volume, openInterest
            - impliedVolatility (yfinance estimate)
            - inTheMoney, contractSymbol
            - optionType (call/put), expirationDate
        """
        print(f"\n{'='*70}")
        print("PHASE 1: RAW DATA ACQUISITION FROM YFINANCE")
        print(f"{'='*70}")
        
        try:
            self.underlying = yf.Ticker(self.ticker)
            
            # Get current spot price
            hist = self.underlying.history(period='1d')
            if hist.empty:
                raise ValueError(f"No price data available for {self.ticker}")
            
            self.spot_price = hist['Close'].iloc[-1]
            print(f"Current spot price: ${self.spot_price:.2f}")
            
            # Get all available expiration dates
            expirations = self.underlying.options
            print(f"Found {len(expirations)} expiration dates")
            
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
                    print(f"WARNING: Failed to fetch {exp_date}: {e}")
                    continue
            
            if not all_options:
                raise ValueError("No option data fetched")
            
            self.raw_options = pd.concat(all_options, ignore_index=True)
            
            print(f"Fetched {len(self.raw_options):,} option contracts")
            print(f"  - Calls: {len(self.raw_options[self.raw_options['optionType']=='call']):,}")
            print(f"  - Puts: {len(self.raw_options[self.raw_options['optionType']=='put']):,}")
            
            return self.raw_options
            
        except Exception as e:
            print(f"ERROR: Error fetching option chain: {e}")
            raise
    
    # ========================================================================
    # PHASE 2: AGGRESSIVE DATA CLEANING
    # ========================================================================
    
    def clean_option_data(self,
                         min_volume: int = 5,
                         min_open_interest: int = 1,
                         max_bid_ask_spread_pct: float = 0.50,
                         min_moneyness: float = 0.70,
                         max_moneyness: float = 1.30) -> pd.DataFrame:
        """
        Phase 2: Clean option data with production-appropriate filters.
        
        Cleaning steps:
        1. Remove options with low liquidity
        2. Remove wide bid-ask spreads
        3. Remove far OTM/ITM options
        4. Check for arbitrage violations
        5. Remove options with invalid prices
        
        Args:
            min_volume: Minimum daily volume (default: 5 for better surface coverage)
            min_open_interest: Minimum open interest (default: 10 for better coverage)
            max_bid_ask_spread_pct: Maximum bid-ask spread as % of mid-price (default: 30%)
            min_moneyness: Minimum strike/spot ratio (default: 0.80)
            max_moneyness: Maximum strike/spot ratio (default: 1.20)
            
        Returns:
            Cleaned DataFrame
        """
        print(f"\n{'='*70}")
        print("PHASE 2: AGGRESSIVE DATA CLEANING")
        print(f"{'='*70}")
        
        if self.raw_options is None:
            raise ValueError("Must fetch option chain first")
        
        df = self.raw_options.copy()
        initial_count = len(df)
        
        # Calculate derived fields
        df['mid_price'] = (df['bid'] + df['ask']) / 2
        df['bid_ask_spread'] = df['ask'] - df['bid']
        df['spread_pct'] = df['bid_ask_spread'] / df['mid_price']
        
        # Time to maturity
        df['expirationDate'] = pd.to_datetime(df['expirationDate'])
        today = datetime.now()
        df['days_to_expiry'] = (df['expirationDate'] - today).dt.days
        df['time_to_maturity'] = df['days_to_expiry'] / 365.0
        
        # Moneyness
        df['moneyness'] = df['strike'] / self.spot_price
        df['log_moneyness'] = np.log(df['moneyness'])
        
        print(f"\nInitial dataset: {initial_count:,} contracts")
        
        # Filter 1: Remove expired
        df = df[df['days_to_expiry'] > 0]
        print(f"After removing expired: {len(df):,} ({100*len(df)/initial_count:.1f}%)")
        
        # Filter 2: Liquidity (volume)
        df = df[df['volume'].fillna(0) >= min_volume]
        print(f"After volume filter (>={min_volume}): {len(df):,} ({100*len(df)/initial_count:.1f}%)")
        
        # Filter 3: Open interest
        df = df[df['openInterest'].fillna(0) >= min_open_interest]
        print(f"After open interest filter (>={min_open_interest}): {len(df):,} ({100*len(df)/initial_count:.1f}%)")
        
        # Filter 4: Bid-ask spread
        df = df[df['spread_pct'] <= max_bid_ask_spread_pct]
        print(f"After spread filter (<={max_bid_ask_spread_pct:.1%}): {len(df):,} ({100*len(df)/initial_count:.1f}%)")
        
        # Filter 5: Moneyness range
        df = df[(df['moneyness'] >= min_moneyness) & (df['moneyness'] <= max_moneyness)]
        print(f"After moneyness filter ({min_moneyness:.1f}-{max_moneyness:.1f}): {len(df):,} ({100*len(df)/initial_count:.1f}%)")
        
        # Filter 6: Positive prices
        df = df[df['mid_price'] > 0]
        print(f"After positive price filter: {len(df):,} ({100*len(df)/initial_count:.1f}%)")
        
        # Filter 7: Arbitrage bounds
        df['discount_factor'] = np.exp(-self.risk_free_rate * df['time_to_maturity'])
        df['pv_strike'] = df['strike'] * df['discount_factor']
        
        call_mask = df['optionType'] == 'call'
        put_mask = df['optionType'] == 'put'
        
        df.loc[call_mask, 'intrinsic_value'] = np.maximum(self.spot_price - df.loc[call_mask, 'pv_strike'], 0)
        df.loc[put_mask, 'intrinsic_value'] = np.maximum(df.loc[put_mask, 'pv_strike'] - self.spot_price, 0)
        
        df = df[df['mid_price'] >= df['intrinsic_value'] * 0.95]  # 5% tolerance
        print(f"After arbitrage check: {len(df):,} ({100*len(df)/initial_count:.1f}%)")
        
        self.cleaned_options = df
        
        print(f"\nCleaning complete: {len(df):,} / {initial_count:,} contracts retained ({100*len(df)/initial_count:.1f}%)")
        
        return self.cleaned_options
    
    # ========================================================================
    # PHASE 3: FORWARD PRICE EXTRACTION
    # ========================================================================
    
    def extract_forward_prices(self) -> Dict[float, float]:
        """
        Phase 3: Extract forward prices via put-call parity.
        
        Put-call parity: C - P = S - K*e^(-rT)
        Rearranged: F = K + e^(rT) * (C - P)
        
        Returns:
            Dictionary mapping time_to_maturity -> forward_price
        """
        print(f"\n{'='*70}")
        print("PHASE 3: FORWARD PRICE EXTRACTION VIA PUT-CALL PARITY")
        print(f"{'='*70}")
        
        if self.cleaned_options is None:
            raise ValueError("Must clean data first")
        
        df = self.cleaned_options
        self.forward_prices = {}
        
        for exp_date in df['expirationDate'].unique():
            exp_df = df[df['expirationDate'] == exp_date]
            T = exp_df.iloc[0]['time_to_maturity']
            
            # Find ATM strike
            atm_strike = exp_df.iloc[(exp_df['strike'] - self.spot_price).abs().argsort()[:1]]['strike'].values[0]
            
            # Get call and put at ATM
            call_data = exp_df[(exp_df['strike'] == atm_strike) & (exp_df['optionType'] == 'call')]
            put_data = exp_df[(exp_df['strike'] == atm_strike) & (exp_df['optionType'] == 'put')]
            
            if len(call_data) > 0 and len(put_data) > 0:
                C = call_data['mid_price'].values[0]
                P = put_data['mid_price'].values[0]
                
                # F = K + e^(rT) * (C - P)
                forward = atm_strike + np.exp(self.risk_free_rate * T) * (C - P)
                self.forward_prices[T] = forward
        
        print(f"Extracted {len(self.forward_prices)} forward prices:")
        for T, F in sorted(self.forward_prices.items()):
            print(f"  T={T:.3f}y → F=${F:.2f} (vs Spot=${self.spot_price:.2f})")
        
        return self.forward_prices
    
    # ========================================================================
    # PHASE 4: IMPLIED VOLATILITY CALCULATION
    # ========================================================================
    
    def compute_implied_volatilities(self) -> pd.DataFrame:
        """
        Phase 4: Compute accurate implied volatilities using QuantLib.
        
        More reliable than yfinance's impliedVolatility field.
        Uses Black-Scholes solver with proper day count conventions.
        
        Returns:
            DataFrame with computed IV in 'implied_volatility' column
        """
        print(f"\n{'='*70}")
        print("PHASE 4: IMPLIED VOLATILITY CALCULATION WITH QUANTLIB")
        print(f"{'='*70}")
        
        if self.cleaned_options is None:
            raise ValueError("Must clean data first")
        
        df = self.cleaned_options.copy()
        
        if QUANTLIB_AVAILABLE:
            print("Using QuantLib for accurate IV calculation...")
            df['implied_volatility'] = df.apply(self._compute_iv_quantlib, axis=1)
        else:
            print("WARNING: Using yfinance IV (less reliable)")
            df['implied_volatility'] = df['impliedVolatility']
        
        # Filter out failed calculations
        initial = len(df)
        df = df[df['implied_volatility'].notna()]
        df = df[(df['implied_volatility'] > 0) & (df['implied_volatility'] < 3.0)]
        
        print(f"IV computation complete: {len(df):,} / {initial:,} options ({100*len(df)/initial:.1f}%)")
        print(f"  IV range: {df['implied_volatility'].min():.2%} - {df['implied_volatility'].max():.2%}")
        
        self.cleaned_options = df
        
        return df
    
    def _compute_iv_quantlib(self, row) -> Optional[float]:
        """
        Compute implied volatility using QuantLib Black-Scholes engine.
        """
        try:
            calculation_date = ql.Date.todaysDate()
            ql.Settings.instance().evaluationDate = calculation_date
            
            option_type = ql.Option.Call if row['optionType'] == 'call' else ql.Option.Put
            strike = row['strike']
            maturity_date = calculation_date + int(row['days_to_expiry'])
            
            spot = self.spot_price
            price = row['mid_price']
            
            day_count = ql.Actual365Fixed()
            calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
            
            spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
            rate_handle = ql.YieldTermStructureHandle(
                ql.FlatForward(calculation_date, self.risk_free_rate, day_count)
            )
            dividend_handle = ql.YieldTermStructureHandle(
                ql.FlatForward(calculation_date, 0.0, day_count)
            )
            
            payoff = ql.PlainVanillaPayoff(option_type, strike)
            exercise = ql.EuropeanExercise(maturity_date)
            option = ql.VanillaOption(payoff, exercise)
            
            try:
                iv = option.impliedVolatility(
                    price,
                    ql.BlackScholesMertonProcess(
                        spot_handle, dividend_handle, rate_handle,
                        ql.BlackVolTermStructureHandle(
                            ql.BlackConstantVol(calculation_date, calendar, 0.20, day_count)
                        )
                    )
                )
                return iv
            except:
                return None
                
        except:
            return None
    
    # ========================================================================
    # DATA EXPORT
    # ========================================================================
    
    def export_for_calibration(self, filename: str):
        """
        Export cleaned data ready for calibration pipeline.
        
        Saves to CSV with all necessary fields for objective_function.py.
        """
        if self.cleaned_options is None:
            raise ValueError("No cleaned data to export")
        
        # Select key columns for calibration
        export_cols = [
            'strike', 'mid_price', 'bid', 'ask',
            'implied_volatility', 'time_to_maturity',
            'moneyness', 'log_moneyness',
            'optionType', 'expirationDate',
            'volume', 'openInterest'
        ]
        
        export_df = self.cleaned_options[export_cols].copy()
        export_df.to_csv(filename, index=False)
        
        print(f"\nExported {len(export_df):,} contracts to {filename}")
        print(f"   Ready for calibration with objective_function.py")
    
    def get_market_data_dict(self) -> Dict[str, pd.DataFrame]:
        """
        Return market data organized by expiration for calibration.
        
        Returns:
            Dictionary mapping expiration_date -> DataFrame of options
        """
        if self.cleaned_options is None:
            raise ValueError("No cleaned data available")
        
        market_data = {}
        for exp_date in self.cleaned_options['expirationDate'].unique():
            market_data[str(exp_date)] = self.cleaned_options[
                self.cleaned_options['expirationDate'] == exp_date
            ].copy()
        
        return market_data


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def acquire_option_data(ticker: str, 
                       risk_free_rate: float = 0.05,
                       export_csv: bool = True) -> OptionChainDataAcquisition:
    """
    Run complete data acquisition pipeline for a ticker.
    
    Args:
        ticker: Underlying ticker symbol
        risk_free_rate: Current risk-free rate
        export_csv: Whether to export cleaned data to CSV
        
    Returns:
        OptionChainDataAcquisition instance with processed data
    """
    print(f"\n{'#'*70}")
    print(f"# OPTION DATA ACQUISITION PIPELINE: {ticker}")
    print(f"{'#'*70}\n")
    
    acq = OptionChainDataAcquisition(ticker, risk_free_rate)
    
    # Phase 1: Fetch
    acq.fetch_option_chain()
    
    # Phase 2: Clean
    acq.clean_option_data()
    
    # Phase 3: Forward extraction
    acq.extract_forward_prices()
    
    # Phase 4: IV calculation
    acq.compute_implied_volatilities()
    
    # Export
    if export_csv:
        acq.export_for_calibration(f'{ticker.lower()}_options.csv')
    
    print(f"\n{'='*70}")
    print("DATA ACQUISITION COMPLETE")
    print(f"{'='*70}\n")
    
    return acq


if __name__ == "__main__":
    print("Option Chain Data Acquisition Module")
    print("=" * 70)
    print("\nExample usage:")
    print(">>> from models.volatility_models.calibration.data_aquisition import acquire_option_data")
    print(">>> acq = acquire_option_data('SPY', risk_free_rate=0.05)")
    print(">>> market_data = acq.get_market_data_dict()")
