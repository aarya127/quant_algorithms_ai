"""
Master Orchestration Script for Market Validation Suite

Executes the complete validation workflow:
1. Parameter path diagnostics
2. Greeks validation
3. Rolling window validation
4. Generate comprehensive report

Usage:
    python run_all_validations.py --ticker SPY --model sabr --history-days 90
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import validation modules
from parameter_diagnostics import ParameterPathDiagnostics
from greeks_validation import GreeksValidator
from rolling_validation import RollingWindowValidator


def create_output_dir() -> Path:
    """Create timestamped output directory."""
    output_dir = Path('validation_results')
    output_dir.mkdir(exist_ok=True)
    return output_dir


def run_parameter_diagnostics(ticker: str, model_type: str, start_date: str, output_dir: Path):
    """
    Run parameter path diagnostics.
    
    Returns:
        Success status and metrics dictionary
    """
    print("\n" + "="*80)
    print("STEP 1: PARAMETER PATH DIAGNOSTICS")
    print("="*80)
    
    try:
        diagnostics = ParameterPathDiagnostics()
        diagnostics.load_calibration_history(
            ticker=ticker,
            model_type=model_type,
            start_date=start_date
        )
        
        # Compute statistics
        stats = diagnostics.compute_statistics()
        regimes = diagnostics.detect_regimes()
        stress_df = diagnostics.identify_stress_periods()
        
        # Print report
        diagnostics.print_diagnostics_report()
        
        # Generate plots
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = output_dir / f"{ticker}_parameter_paths_{timestamp}.png"
        diagnostics.plot_parameter_paths(save_path=str(plot_path))
        
        # Export regime analysis
        regime_csv = output_dir / f"{ticker}_regime_analysis_{timestamp}.csv"
        stress_df.to_csv(regime_csv, index=False)
        print(f"\nExported regime analysis to: {regime_csv}")
        
        return True, {
            'n_observations': len(diagnostics.param_history),
            'n_regimes': regimes.n_regimes,
            'n_stress_periods': len(stress_df[stress_df['stress_score'] > 0.5]),
            'stats': stats
        }
        
    except ValueError as e:
        print(f"\nWARNING: Parameter diagnostics skipped - {e}")
        print("Need multiple calibration snapshots to build history.")
        return False, {}


def run_greeks_validation(ticker: str, model_type: str, output_dir: Path):
    """
    Run Greeks validation against Black-Scholes.
    
    Returns:
        Success status and metrics dictionary
    """
    print("\n" + "="*80)
    print("STEP 2: GREEKS VALIDATION")
    print("="*80)
    
    try:
        # Load latest calibration parameters
        from pathlib import Path
        import pandas as pd
        import glob
        
        results_dir = Path('../calibration/calibration_results')
        param_files = sorted(glob.glob(str(results_dir / f"{ticker}_{model_type}_params_*.csv")))
        
        if not param_files:
            raise ValueError(f"No calibration results found for {ticker}/{model_type}")
        
        # Load most recent parameters
        latest_params = pd.read_csv(param_files[-1]).iloc[0].to_dict()
        
        # Get spot price from most recent market data
        market_files = sorted(glob.glob(str(results_dir / f"{ticker}_market_data_*.csv")))
        if market_files:
            market_data = pd.read_csv(market_files[-1])
            spot = market_data['spot'].iloc[0] if 'spot' in market_data.columns else 682.85
        else:
            spot = 682.85  # Default
        
        print(f"\nUsing parameters: {latest_params}")
        print(f"Spot price: ${spot:.2f}")
        
        # Initialize validator
        validator = GreeksValidator(spot=spot, rate=0.05)
        
        # Define validation grid
        import numpy as np
        strikes = np.linspace(spot * 0.90, spot * 1.10, 5).tolist()
        maturities = [0.25, 0.5, 1.0]  # 3mo, 6mo, 1yr
        
        # Run validation
        results = validator.validate_option_greeks(
            strikes=strikes,
            maturities=maturities,
            model_type=model_type,
            model_params=latest_params
        )
        
        # Print report
        validator.print_validation_report()
        
        # Generate plots
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = output_dir / f"{ticker}_greeks_validation_{timestamp}.png"
        validator.plot_greeks_comparison(save_path=str(plot_path))
        
        # Compute summary metrics
        delta_errors = [r.delta_error_vs_bs for r in results]
        gamma_errors = [r.gamma_error_vs_bs for r in results]
        vega_errors = [r.vega_error_vs_bs for r in results]
        
        parity_errors = validator.check_put_call_parity_delta()
        
        return True, {
            'n_options_validated': len(results),
            'delta_mae': np.mean(delta_errors),
            'gamma_mae': np.mean(gamma_errors),
            'vega_mae': np.mean(vega_errors),
            'put_call_parity_mae': np.mean(list(parity_errors.values())),
            'parity_check': 'PASS' if np.mean(list(parity_errors.values())) < 0.01 else 'REVIEW'
        }
        
    except Exception as e:
        print(f"\nERROR: Greeks validation failed - {e}")
        return False, {}


def run_rolling_validation(ticker: str, model_type: str, start_date: str, end_date: str, output_dir: Path):
    """
    Run rolling window validation.
    
    Returns:
        Success status and metrics dictionary
    """
    print("\n" + "="*80)
    print("STEP 3: ROLLING WINDOW VALIDATION")
    print("="*80)
    
    try:
        validator = RollingWindowValidator(
            calibration_window_days=30,
            forecast_horizon_days=5,
            step_size_days=5
        )
        
        # Run validation (mock for demonstration)
        results = validator.run_validation(
            ticker=ticker,
            model_type=model_type,
            start_date=start_date,
            end_date=end_date
        )
        
        # Print summary
        validator.print_summary()
        
        # Generate plots
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = output_dir / f"{ticker}_oos_performance_{timestamp}.png"
        validator.plot_oos_performance(save_path=str(plot_path))
        
        # Export rolling metrics
        import pandas as pd
        rolling_df = pd.DataFrame([{
            'forecast_date': w.forecast_start,
            'price_rmse_pct': w.rmse_price_pct,
            'iv_rmse': w.rmse_iv,
            **w.parameters
        } for w in results.all_windows])
        
        csv_path = output_dir / f"{ticker}_rolling_metrics_{timestamp}.csv"
        rolling_df.to_csv(csv_path, index=False)
        print(f"\nExported rolling metrics to: {csv_path}")
        
        return True, {
            'n_windows': results.n_windows,
            'avg_price_rmse_pct': results.avg_price_rmse_pct,
            'avg_iv_rmse': results.avg_iv_rmse,
            'parameter_stability': results.parameter_stability
        }
        
    except Exception as e:
        print(f"\nERROR: Rolling validation failed - {e}")
        return False, {}


def generate_summary_report(ticker: str, model_type: str, 
                           param_metrics: dict, greeks_metrics: dict, 
                           rolling_metrics: dict, output_dir: Path):
    """Generate comprehensive validation summary report."""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = output_dir / f"{ticker}_validation_summary_{timestamp}.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MARKET VALIDATION SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Ticker: {ticker}\n")
        f.write(f"Model: {model_type.upper()}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Parameter diagnostics
        f.write("-"*80 + "\n")
        f.write("1. PARAMETER PATH DIAGNOSTICS\n")
        f.write("-"*80 + "\n")
        if param_metrics:
            f.write(f"Observations: {param_metrics.get('n_observations', 'N/A')}\n")
            f.write(f"Regimes detected: {param_metrics.get('n_regimes', 'N/A')}\n")
            f.write(f"Stress periods: {param_metrics.get('n_stress_periods', 'N/A')}\n")
            
            if 'stats' in param_metrics:
                f.write("\nParameter Statistics:\n")
                for param_name, stat in param_metrics['stats'].items():
                    f.write(f"  {param_name}:\n")
                    f.write(f"    Autocorr(1): {stat.autocorr_lag1:.4f}\n")
                    f.write(f"    Daily vol: {stat.daily_change_vol:.4f}\n")
                    f.write(f"    Jumps: {len(stat.jump_dates)}\n")
        else:
            f.write("SKIPPED: Insufficient calibration history\n")
        
        # Greeks validation
        f.write("\n" + "-"*80 + "\n")
        f.write("2. GREEKS VALIDATION\n")
        f.write("-"*80 + "\n")
        if greeks_metrics:
            f.write(f"Options validated: {greeks_metrics.get('n_options_validated', 'N/A')}\n")
            f.write(f"Delta MAE: {greeks_metrics.get('delta_mae', 0):.6f}\n")
            f.write(f"Gamma MAE: {greeks_metrics.get('gamma_mae', 0):.6f}\n")
            f.write(f"Vega MAE: {greeks_metrics.get('vega_mae', 0):.6f}\n")
            f.write(f"Put-call parity error: {greeks_metrics.get('put_call_parity_mae', 0):.6f}\n")
            f.write(f"Parity check: {greeks_metrics.get('parity_check', 'N/A')}\n")
        else:
            f.write("FAILED: Could not validate Greeks\n")
        
        # Rolling validation
        f.write("\n" + "-"*80 + "\n")
        f.write("3. ROLLING WINDOW VALIDATION\n")
        f.write("-"*80 + "\n")
        if rolling_metrics:
            f.write(f"Windows tested: {rolling_metrics.get('n_windows', 'N/A')}\n")
            f.write(f"Avg OOS price RMSE: {rolling_metrics.get('avg_price_rmse_pct', 0):.2f}%\n")
            f.write(f"Avg OOS IV RMSE: {rolling_metrics.get('avg_iv_rmse', 0):.4f}\n")
            
            if 'parameter_stability' in rolling_metrics:
                f.write("\nParameter Stability (Std Dev):\n")
                for param, std in rolling_metrics['parameter_stability'].items():
                    f.write(f"  {param}: {std:.4f}\n")
        else:
            f.write("FAILED: Could not complete rolling validation\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"\n{'='*80}")
    print(f"VALIDATION COMPLETE")
    print(f"{'='*80}")
    print(f"Summary report saved to: {report_path}")
    
    return report_path


def main():
    """Main orchestration function."""
    parser = argparse.ArgumentParser(
        description='Run comprehensive market validation suite for volatility models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all validations for SPY SABR with 90-day history
  python run_all_validations.py --ticker SPY --model sabr --history-days 90
  
  # Run with custom date range
  python run_all_validations.py --ticker QQQ --model sabr --start-date 2025-01-01 --end-date 2025-12-31
  
  # Skip parameter diagnostics (if no history available)
  python run_all_validations.py --ticker SPY --model sabr --skip-param-diagnostics
        """
    )
    
    parser.add_argument('--ticker', type=str, required=True,
                       help='Ticker symbol (e.g., SPY, QQQ)')
    parser.add_argument('--model', type=str, required=True, choices=['sabr', 'heston'],
                       help='Model type')
    parser.add_argument('--history-days', type=int, default=90,
                       help='Days of history for parameter diagnostics (default: 90)')
    parser.add_argument('--start-date', type=str,
                       help='Start date for rolling validation (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                       help='End date for rolling validation (YYYY-MM-DD)')
    parser.add_argument('--skip-param-diagnostics', action='store_true',
                       help='Skip parameter diagnostics (if no calibration history)')
    parser.add_argument('--skip-rolling', action='store_true',
                       help='Skip rolling window validation (faster)')
    
    args = parser.parse_args()
    
    # Calculate dates
    if not args.start_date:
        start_date = (datetime.now() - timedelta(days=args.history_days)).strftime('%Y-%m-%d')
    else:
        start_date = args.start_date
    
    if not args.end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    else:
        end_date = args.end_date
    
    print("="*80)
    print("MARKET VALIDATION SUITE - MASTER ORCHESTRATION")
    print("="*80)
    print(f"Ticker: {args.ticker}")
    print(f"Model: {args.model.upper()}")
    print(f"Period: {start_date} to {end_date}")
    print("="*80)
    
    # Create output directory
    output_dir = create_output_dir()
    print(f"\nOutput directory: {output_dir.absolute()}")
    
    # Initialize metrics
    param_metrics = {}
    greeks_metrics = {}
    rolling_metrics = {}
    
    # Run validations
    try:
        # 1. Parameter diagnostics
        if not args.skip_param_diagnostics:
            success, param_metrics = run_parameter_diagnostics(
                args.ticker, args.model, start_date, output_dir
            )
        else:
            print("\n" + "="*80)
            print("STEP 1: PARAMETER PATH DIAGNOSTICS - SKIPPED")
            print("="*80)
        
        # 2. Greeks validation
        success, greeks_metrics = run_greeks_validation(
            args.ticker, args.model, output_dir
        )
        
        # 3. Rolling validation
        if not args.skip_rolling:
            success, rolling_metrics = run_rolling_validation(
                args.ticker, args.model, start_date, end_date, output_dir
            )
        else:
            print("\n" + "="*80)
            print("STEP 3: ROLLING WINDOW VALIDATION - SKIPPED")
            print("="*80)
        
        # Generate summary report
        report_path = generate_summary_report(
            args.ticker, args.model,
            param_metrics, greeks_metrics, rolling_metrics,
            output_dir
        )
        
        print(f"\nAll validation outputs saved to: {output_dir.absolute()}")
        
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: Validation failed - {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
