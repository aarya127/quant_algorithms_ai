"""
Volatility Model Calibration Orchestration Script

This script orchestrates the complete calibration pipeline:
1. Data acquisition from market sources (Yahoo Finance)
2. Data validation and cleaning
3. Model calibration (SABR/Heston)
4. Constraint validation
5. Results export and diagnostics

Usage:
    python run_calibration.py --ticker SPY --model sabr
    python run_calibration.py --ticker NVDA --model heston --rf-rate 0.05
"""

import argparse
import logging
import sys
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from models.volatility_models.calibration.data_aquisition import acquire_option_data
from models.volatility_models.calibration.objective_function import create_sabr_objective, create_heston_objective
from models.volatility_models.calibration.constraints_handling import ConstraintHandler, ConstraintConfig
from models.volatility_models.calibration.caplet_stripping import (
    ForwardCurveBuilder, CapletVolatilityStripper, BlackCapletPricer
)
from scipy.optimize import minimize, differential_evolution


# Configure logging
def setup_logging(log_file: str = None):
    """Set up logging configuration"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers
    )


class CalibrationOrchestrator:
    """Orchestrates the complete volatility calibration pipeline"""
    
    def __init__(self, ticker: str, model_type: str, risk_free_rate: float = 0.05, 
                 product_type: str = 'equity'):
        """
        Initialize the calibration orchestrator
        
        Args:
            ticker: Stock ticker symbol (e.g., 'SPY', 'NVDA') or rate product identifier
            model_type: Model to calibrate ('sabr' or 'heston')
            risk_free_rate: Risk-free interest rate (decimal)
            product_type: 'equity' for options, 'rates' for caplets/floorlets
        """
        self.ticker = ticker
        self.model_type = model_type.lower()
        self.risk_free_rate = risk_free_rate
        self.product_type = product_type.lower()
        self.logger = logging.getLogger(__name__)
        
        # Results storage
        self.market_data = None
        self.spot_price = None
        self.calibration_results = None
        self.diagnostics = {}
        
        self.logger.info(f"Initialized calibration for {ticker} using {model_type.upper()} model")
        self.logger.info(f"Product type: {product_type}")
        self.logger.info(f"Risk-free rate: {risk_free_rate:.4f}")
    
    def step1_acquire_data(self):
        """Step 1: Acquire market data"""
        self.logger.info("=" * 70)
        self.logger.info("STEP 1: DATA ACQUISITION")
        self.logger.info("=" * 70)
        
        try:
            if self.product_type == 'equity':
                self.logger.info(f"Fetching equity option chain data for {self.ticker}...")
                
                # Use the data acquisition module
                acq = acquire_option_data(
                    ticker=self.ticker,
                    risk_free_rate=self.risk_free_rate
                )
                
                # Get the cleaned options DataFrame
                if acq.cleaned_options is None or len(acq.cleaned_options) == 0:
                    raise ValueError("No market data retrieved after cleaning")
                
                # Store the DataFrame directly for objective function
                self.market_data = acq.cleaned_options
                self.spot_price = acq.spot_price
                
                # Log data summary
                num_options = len(self.market_data)
                self.logger.info(f"Successfully retrieved {num_options} option quotes")
                
                expiries = self.market_data['expirationDate'].unique()
                self.logger.info(f"Expiration dates: {len(expiries)} unique maturities")
                
                self.logger.info(f"Current spot price: ${self.spot_price:.2f}")
                
                # Log IV range
                if 'implied_volatility' in self.market_data.columns:
                    iv_min = self.market_data['implied_volatility'].min()
                    iv_max = self.market_data['implied_volatility'].max()
                    self.logger.info(f"IV range: {iv_min:.4f} to {iv_max:.4f}")
            
            elif self.product_type == 'rates':
                self.logger.info("Interest rate products - using caplet stripping workflow...")
                self.logger.info("NOTE: For rates, provide swap_data and cap_data as manual inputs")
                self.logger.info("Example: orchestrator.load_rate_data(swap_tenors, swap_rates, cap_maturities, cap_prices)")
                
                # For rates, we need manual data input (not from yfinance)
                # User should call load_rate_data() before running pipeline
                if not hasattr(self, 'rate_data_loaded') or not self.rate_data_loaded:
                    raise ValueError(
                        "Rate data not loaded. Call orchestrator.load_rate_data() first with:\n"
                        "  - swap_tenors: [1, 2, 5, 10] (years)\n"
                        "  - swap_rates: [0.04, 0.045, 0.05, 0.052] (decimals)\n"
                        "  - cap_maturities: [1, 2, 3, 5] (years)\n"
                        "  - cap_prices: [0.0012, 0.0035, 0.0068, 0.0142] (undiscounted)"
                    )
                
                self.logger.info(f"Using pre-loaded rate data: {len(self.market_data)} caplet vols")
            
            else:
                raise ValueError(f"Unknown product type: {self.product_type}")
            
            self.logger.info("Data acquisition completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Data acquisition failed: {str(e)}")
            return False
    
    def load_rate_data(self, swap_tenors: list, swap_rates: list, 
                       cap_maturities: list, cap_prices: list, strike: float = 0.05):
        """
        Load interest rate market data for caplet calibration.
        
        Args:
            swap_tenors: Swap tenors in years [1, 2, 5, 10]
            swap_rates: Par swap rates [0.04, 0.045, 0.05, 0.052]
            cap_maturities: Cap maturities in years [1, 2, 3, 5]
            cap_prices: Cap prices (undiscounted) [0.0012, 0.0035, ...]
            strike: Strike rate (decimal)
        """
        self.logger.info("Loading interest rate market data...")
        
        try:
            # Build forward curve from swaps
            curve_builder = ForwardCurveBuilder()
            forward_curve = curve_builder.bootstrap_from_swaps(swap_tenors, swap_rates)
            
            # Strip caplet volatilities
            pricer = BlackCapletPricer(forward_curve)
            stripper = CapletVolatilityStripper(forward_curve, pricer)
            
            caplet_vols = stripper.strip_from_cap_prices(cap_maturities, cap_prices, strike)
            
            # Convert to DataFrame format for calibration
            self.market_data = pd.DataFrame({
                'maturity': [c['maturity'] for c in caplet_vols],
                'strike': [c['strike'] for c in caplet_vols],
                'implied_volatility': [c['volatility'] for c in caplet_vols],
                'forward_rate': [c['forward_rate'] for c in caplet_vols]
            })
            
            # For rates, "spot" is the first forward rate
            self.spot_price = self.market_data['forward_rate'].iloc[0]
            
            self.rate_data_loaded = True
            
            self.logger.info(f"Loaded {len(caplet_vols)} caplet volatilities")
            self.logger.info(f"Maturity range: {min(cap_maturities):.1f} to {max(cap_maturities):.1f} years")
            self.logger.info(f"Vol range: {self.market_data['implied_volatility'].min():.4f} to {self.market_data['implied_volatility'].max():.4f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load rate data: {str(e)}")
            self.rate_data_loaded = False
            return False
    
    def step2_validate_data(self):
        """Step 2: Validate market data quality"""
        self.logger.info("=" * 70)
        self.logger.info("STEP 2: DATA VALIDATION & SURFACE DENSITY CHECK")
        self.logger.info("=" * 70)
        
        if self.market_data is None or len(self.market_data) == 0:
            self.logger.error("No market data available for validation")
            return False
        
        try:
            # Check required columns
            required_columns = ['strike', 'implied_volatility', 'mid_price', 'time_to_maturity']
            missing_columns = [c for c in required_columns if c not in self.market_data.columns]
            
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return False
            
            self.logger.info("All required columns present")
            
            # SURFACE DENSITY GATE - Critical for reliable calibration
            self.logger.info("-" * 70)
            self.logger.info("SURFACE DENSITY CHECK (Production Desk Standard)")
            self.logger.info("-" * 70)
            
            total_points = len(self.market_data)
            num_maturities = self.market_data['expirationDate'].nunique() if 'expirationDate' in self.market_data.columns else 1
            
            # Calculate strikes per maturity
            if 'expirationDate' in self.market_data.columns:
                strikes_per_maturity = self.market_data.groupby('expirationDate')['strike'].count()
                min_strikes = strikes_per_maturity.min()
                max_strikes = strikes_per_maturity.max()
                avg_strikes = strikes_per_maturity.mean()
                
                self.logger.info(f"Total IV points: {total_points}")
                self.logger.info(f"Number of maturities: {num_maturities}")
                self.logger.info(f"Strikes per maturity - Min: {min_strikes}, Max: {max_strikes}, Avg: {avg_strikes:.1f}")
            else:
                min_strikes = total_points
                avg_strikes = total_points
                self.logger.info(f"Total IV points: {total_points}")
                self.logger.info(f"Number of maturities: {num_maturities}")
            
            # Apply production-grade density gates
            gate_passed = False
            failure_reasons = []
            
            # Gate 1: Minimum strikes per maturity
            if min_strikes >= 12:
                gate_passed = True
                self.logger.info(f"Gate 1 PASS: Min {min_strikes} strikes per maturity (threshold: 12)")
            else:
                failure_reasons.append(f"Gate 1 FAIL: Only {min_strikes} strikes per maturity (need >= 12)")
            
            # Gate 2: Total IV points
            if total_points >= 20:
                if not gate_passed:
                    gate_passed = True
                self.logger.info(f"Gate 2 PASS: {total_points} total IV points (threshold: 20)")
            else:
                failure_reasons.append(f"Gate 2 FAIL: Only {total_points} total IV points (need >= 20)")
            
            # Gate 3: Sufficient maturity x strike coverage
            if num_maturities >= 3 and avg_strikes >= 7:
                if not gate_passed:
                    gate_passed = True
                self.logger.info(f"Gate 3 PASS: {num_maturities} maturities x {avg_strikes:.1f} avg strikes (threshold: 3x7)")
            else:
                failure_reasons.append(f"Gate 3 FAIL: {num_maturities} maturities x {avg_strikes:.1f} avg strikes (need >= 3x7)")
            
            if not gate_passed:
                self.logger.error("INSUFFICIENT SURFACE DENSITY FOR CALIBRATION")
                for reason in failure_reasons:
                    self.logger.error(f"  {reason}")
                self.logger.error("Decision: SKIP CALIBRATION - Mark as 'insufficient liquidity'")
                self.logger.error("Recommendation: Try more liquid ticker or relax filtering parameters")
                return False
            
            self.logger.info("Surface density check PASSED - Sufficient for reliable calibration")
            self.logger.info("-" * 70)
            
            # Validate data ranges
            ivs = self.market_data['implied_volatility'].values
            strikes = self.market_data['strike'].values
            
            # Check for NaN values
            nan_count = np.sum(np.isnan(ivs))
            if nan_count > 0:
                self.logger.warning(f"Found {nan_count} NaN values in implied volatilities")
                # Remove NaN values
                self.market_data = self.market_data.dropna(subset=['implied_volatility'])
                self.logger.info(f"Removed NaN values, {len(self.market_data)} options remaining")
            
            # Recheck after cleaning
            if len(self.market_data) == 0:
                self.logger.error("No valid data remaining after removing NaN values")
                return False
            
            # Check volatility ranges
            ivs = self.market_data['implied_volatility'].values
            self.logger.info(f"IV range: {ivs.min():.4f} to {ivs.max():.4f}")
            
            if ivs.min() < 0.01:
                self.logger.warning("Some IVs are very low (< 1%)")
            if ivs.max() > 3.0:
                self.logger.warning("Some IVs are very high (> 300%)")
            
            # Check strike distribution
            strikes = self.market_data['strike'].values
            moneyness = strikes / self.spot_price
            self.logger.info(f"Moneyness range: {moneyness.min():.3f} to {moneyness.max():.3f}")
            
            self.logger.info("Data validation completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {str(e)}")
            return False
    
    def step3_calibrate_model(self):
        """Step 3: Calibrate volatility model"""
        self.logger.info("=" * 70)
        self.logger.info(f"STEP 3: {self.model_type.upper()} MODEL CALIBRATION")
        self.logger.info("=" * 70)
        
        try:
            # Create objective function
            if self.model_type == 'sabr':
                self.logger.info("Creating SABR objective function...")
                obj_func = create_sabr_objective(
                    market_data=self.market_data,
                    spot=self.spot_price,
                    beta=1.0  # Standard for equities
                )
                initial_params = [0.2, 0.3, 0.0]  # [alpha, nu, rho]
                param_names = ['alpha', 'nu', 'rho']
                
            elif self.model_type == 'heston':
                self.logger.info("Creating Heston objective function...")
                obj_func = create_heston_objective(
                    market_data=self.market_data,
                    spot=self.spot_price
                )
                initial_params = [0.04, 2.0, 0.04, 0.3, -0.5]  # [v0, kappa, theta, xi, rho]
                param_names = ['v0', 'kappa', 'theta', 'xi', 'rho']
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            # Set up constraints
            self.logger.info("Setting up constraint handler...")
            config = ConstraintConfig(
                penalty_weight=100.0,
                use_soft_penalties=True,
                feller_penalty_weight=500.0
            )
            constraint_handler = ConstraintHandler(config)
            
            # Wrap objective with constraints
            constrained_obj = constraint_handler.wrap_objective(obj_func, model_type=self.model_type)
            
            # Get bounds
            if self.model_type == 'sabr':
                bounds = constraint_handler.get_sabr_bounds()
            else:
                bounds = constraint_handler.get_heston_bounds()
            
            self.logger.info(f"Initial parameters: {dict(zip(param_names, initial_params))}")
            
            # Calibration using differential evolution (more robust)
            self.logger.info("Starting optimization with differential evolution...")
            self.logger.info("This may take 1-2 minutes...")
            
            result = differential_evolution(
                constrained_obj,
                bounds=bounds,
                maxiter=300,
                popsize=15,
                tol=1e-6,
                seed=42,
                workers=1,
                updating='deferred',
                disp=False
            )
            
            if result.success:
                self.logger.info("Calibration converged successfully")
            else:
                self.logger.warning(f"Calibration did not fully converge: {result.message}")
            
            self.calibration_results = {
                'parameters': dict(zip(param_names, result.x)),
                'objective_value': result.fun,
                'success': result.success,
                'iterations': result.nit,
                'function_evaluations': result.nfev
            }
            
            # Log results
            self.logger.info("Calibrated parameters:")
            for name, value in self.calibration_results['parameters'].items():
                self.logger.info(f"  {name}: {value:.6f}")
            
            self.logger.info(f"Final objective value: {result.fun:.6f}")
            self.logger.info(f"Function evaluations: {result.nfev}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Calibration failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def step4_validate_constraints(self):
        """Step 4: Validate calibrated parameters against constraints"""
        self.logger.info("=" * 70)
        self.logger.info("STEP 4: CONSTRAINT VALIDATION")
        self.logger.info("=" * 70)
        
        if not self.calibration_results:
            self.logger.error("No calibration results to validate")
            return False
        
        try:
            params = self.calibration_results['parameters']
            
            if self.model_type == 'sabr':
                # SABR constraints
                alpha = params['alpha']
                nu = params['nu']
                rho = params['rho']
                
                checks = {
                    'alpha > 0': alpha > 0,
                    'nu > 0': nu > 0,
                    '-1 < rho < 1': -1 < rho < 1,
                }
                
            elif self.model_type == 'heston':
                # Heston constraints
                v0 = params['v0']
                kappa = params['kappa']
                theta = params['theta']
                xi = params['xi']
                rho = params['rho']
                
                feller = 2 * kappa * theta - xi**2
                
                checks = {
                    'v0 > 0': v0 > 0,
                    'kappa > 0': kappa > 0,
                    'theta > 0': theta > 0,
                    'xi > 0': xi > 0,
                    '-1 < rho < 1': -1 < rho < 1,
                    'Feller condition (2*kappa*theta >= xi^2)': feller >= 0
                }
                
                if feller >= 0:
                    self.logger.info(f"Feller condition: 2*{kappa:.4f}*{theta:.4f} - {xi:.4f}^2 = {feller:.6f} >= 0")
            
            # Check all constraints
            all_passed = True
            for constraint, passed in checks.items():
                status = "PASS" if passed else "FAIL"
                self.logger.info(f"  {constraint}: {status}")
                if not passed:
                    all_passed = False
            
            if all_passed:
                self.logger.info("All constraints satisfied")
            else:
                self.logger.warning("Some constraints violated - results may be unreliable")
            
            return all_passed
            
        except Exception as e:
            self.logger.error(f"Constraint validation failed: {str(e)}")
            return False
    
    def step5_export_results(self, output_dir: str = None):
        """Step 5: Export calibration results"""
        self.logger.info("=" * 70)
        self.logger.info("STEP 5: EXPORT RESULTS")
        self.logger.info("=" * 70)
        
        if not self.calibration_results:
            self.logger.error("No results to export")
            return False
        
        try:
            # Create output directory
            if output_dir is None:
                output_dir = Path(__file__).parent / 'calibration_results'
            else:
                output_dir = Path(output_dir)
            
            output_dir.mkdir(exist_ok=True)
            
            # Generate timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Export parameters
            params_file = output_dir / f'{self.ticker}_{self.model_type}_params_{timestamp}.csv'
            params_df = pd.DataFrame([self.calibration_results['parameters']])
            params_df.to_csv(params_file, index=False)
            self.logger.info(f"Saved parameters to: {params_file}")
            
            # Export market data
            if self.market_data is not None and len(self.market_data) > 0:
                data_file = output_dir / f'{self.ticker}_market_data_{timestamp}.csv'
                # market_data is already a DataFrame, just save it
                self.market_data.to_csv(data_file, index=False)
                self.logger.info(f"Saved market data to: {data_file}")
            
            # Export summary
            summary_file = output_dir / f'{self.ticker}_{self.model_type}_summary_{timestamp}.txt'
            with open(summary_file, 'w') as f:
                f.write(f"Volatility Calibration Summary\n")
                f.write(f"{'=' * 60}\n\n")
                f.write(f"Ticker: {self.ticker}\n")
                f.write(f"Model: {self.model_type.upper()}\n")
                f.write(f"Risk-free rate: {self.risk_free_rate:.4f}\n")
                f.write(f"Calibration date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write(f"Calibrated Parameters:\n")
                f.write(f"{'-' * 60}\n")
                for name, value in self.calibration_results['parameters'].items():
                    f.write(f"{name:10s}: {value:12.6f}\n")
                
                f.write(f"\nCalibration Quality:\n")
                f.write(f"{'-' * 60}\n")
                f.write(f"Objective value: {self.calibration_results['objective_value']:.6f}\n")
                f.write(f"Converged: {self.calibration_results['success']}\n")
                f.write(f"Function evaluations: {self.calibration_results['function_evaluations']}\n")
            
            self.logger.info(f"Saved summary to: {summary_file}")
            self.logger.info(f"All results exported to: {output_dir}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Export failed: {str(e)}")
            return False
    
    def run_full_pipeline(self, output_dir: str = None):
        """Execute the complete calibration pipeline"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info(f"VOLATILITY CALIBRATION PIPELINE - {self.ticker}")
        self.logger.info("=" * 70 + "\n")
        
        start_time = datetime.now()
        
        # Execute pipeline steps
        steps = [
            ("Data Acquisition", self.step1_acquire_data),
            ("Data Validation", self.step2_validate_data),
            ("Model Calibration", self.step3_calibrate_model),
            ("Constraint Validation", self.step4_validate_constraints),
            ("Export Results", lambda: self.step5_export_results(output_dir))
        ]
        
        for step_name, step_func in steps:
            success = step_func()
            if not success:
                self.logger.error(f"Pipeline failed at step: {step_name}")
                return False
        
        # Calculate elapsed time
        elapsed = datetime.now() - start_time
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        self.logger.info("=" * 70)
        self.logger.info(f"Total execution time: {elapsed.total_seconds():.2f} seconds")
        
        return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Run volatility model calibration pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Equity options
  python run_calibration.py --ticker SPY --model sabr
  python run_calibration.py --ticker NVDA --model heston --rf-rate 0.05
  python run_calibration.py --ticker AAPL --model sabr --output ./results
  
  # Interest rate products (caplets)
  python run_calibration.py --ticker USD_CAPS --model sabr --product rates
        """
    )
    
    parser.add_argument('--ticker', type=str, required=True,
                        help='Stock ticker symbol (e.g., SPY, NVDA, AAPL) or rate identifier')
    parser.add_argument('--model', type=str, choices=['sabr', 'heston'], required=True,
                        help='Volatility model to calibrate')
    parser.add_argument('--product', type=str, choices=['equity', 'rates'], default='equity',
                        help='Product type: equity options or interest rate caplets (default: equity)')
    parser.add_argument('--rf-rate', type=float, default=0.05,
                        help='Risk-free interest rate (default: 0.05)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for results (default: ./calibration_results)')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Log file path (default: stdout only)')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_file)
    
    # Create orchestrator and run pipeline
    orchestrator = CalibrationOrchestrator(
        ticker=args.ticker,
        model_type=args.model,
        risk_free_rate=args.rf_rate,
        product_type=args.product
    )
    
    success = orchestrator.run_full_pipeline(output_dir=args.output)
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
