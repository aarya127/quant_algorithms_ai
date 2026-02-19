"""
Rolling Window Validation for Volatility Models

Tests calibration stability and out-of-sample forecasting accuracy using:
- Rolling window calibration (e.g., 30-day calibration, 5-day forecast)
- Expanding window validation (cumulative data)
- Walk-forward analysis with performance metrics

This demonstrates that the model generalizes beyond in-sample fit.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ValidationMetrics:
    """Out-of-sample validation metrics for a single window."""
    calibration_end: datetime
    forecast_start: datetime
    forecast_end: datetime
    
    # Pricing errors
    mean_price_error_pct: float
    rmse_price_pct: float
    mae_price_pct: float
    max_price_error_pct: float
    
    # IV errors
    mean_iv_error: float
    rmse_iv: float
    mae_iv: float
    max_iv_error: float
    
    # Parameter values
    parameters: Dict[str, float]
    
    # Fit quality
    n_calibration_points: int
    n_validation_points: int
    calibration_objective: float


@dataclass
class ValidationSummary:
    """Aggregated validation results across all windows."""
    n_windows: int
    avg_price_rmse_pct: float
    avg_iv_rmse: float
    parameter_stability: Dict[str, float]  # Std dev of parameters across windows
    best_window: ValidationMetrics
    worst_window: ValidationMetrics
    all_windows: List[ValidationMetrics]


class RollingWindowValidator:
    """
    Perform rolling window validation for volatility models.
    
    Key Features:
    1. Rolling calibration with fixed window size
    2. Out-of-sample forecast validation
    3. Parameter stability tracking
    4. Regime-dependent performance analysis
    
    Usage:
        validator = RollingWindowValidator(
            calibration_window_days=30,
            forecast_horizon_days=5,
            step_size_days=5
        )
        results = validator.run_validation(
            ticker='SPY',
            model_type='sabr',
            start_date='2025-01-01',
            end_date='2025-12-31'
        )
        validator.print_summary()
    """
    
    def __init__(self,
                 calibration_window_days: int = 30,
                 forecast_horizon_days: int = 5,
                 step_size_days: int = 5):
        """
        Initialize rolling window validator.
        
        Args:
            calibration_window_days: Size of calibration window
            forecast_horizon_days: Forecast horizon for validation
            step_size_days: Days to roll forward between windows
        """
        self.calibration_window_days = calibration_window_days
        self.forecast_horizon_days = forecast_horizon_days
        self.step_size_days = step_size_days
        
        self.results: Optional[ValidationSummary] = None
        self.ticker: Optional[str] = None
        self.model_type: Optional[str] = None
    
    def run_validation(self,
                      ticker: str,
                      model_type: str,
                      start_date: str,
                      end_date: str,
                      calibration_func: Optional[Callable] = None) -> ValidationSummary:
        """
        Run rolling window validation.
        
        Args:
            ticker: Ticker symbol
            model_type: 'sabr' or 'heston'
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            calibration_func: Optional custom calibration function
            
        Returns:
            ValidationSummary with aggregated results
        """
        self.ticker = ticker
        self.model_type = model_type
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        window_results = []
        current_date = start
        
        print(f"\n{'='*70}")
        print(f"ROLLING WINDOW VALIDATION: {ticker} - {model_type.upper()}")
        print(f"{'='*70}")
        print(f"Calibration window: {self.calibration_window_days} days")
        print(f"Forecast horizon: {self.forecast_horizon_days} days")
        print(f"Step size: {self.step_size_days} days")
        print(f"Period: {start_date} to {end_date}")
        
        window_count = 0
        
        while current_date + timedelta(days=self.calibration_window_days + self.forecast_horizon_days) <= end:
            window_count += 1
            
            calibration_start = current_date
            calibration_end = current_date + timedelta(days=self.calibration_window_days)
            forecast_start = calibration_end + timedelta(days=1)
            forecast_end = forecast_start + timedelta(days=self.forecast_horizon_days)
            
            print(f"\nWindow {window_count}:")
            print(f"  Calibration: {calibration_start.date()} to {calibration_end.date()}")
            print(f"  Validation:  {forecast_start.date()} to {forecast_end.date()}")
            
            try:
                # Perform calibration on window
                if calibration_func:
                    metrics = calibration_func(ticker, model_type, calibration_start, calibration_end, 
                                              forecast_start, forecast_end)
                else:
                    metrics = self._default_validation_window(ticker, model_type, calibration_start, 
                                                             calibration_end, forecast_start, forecast_end)
                
                window_results.append(metrics)
                
                print(f"  Calibration objective: {metrics.calibration_objective:.4f}")
                print(f"  OOS price RMSE: {metrics.rmse_price_pct:.2f}%")
                print(f"  OOS IV RMSE: {metrics.rmse_iv:.4f}")
                
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
            
            # Roll forward
            current_date += timedelta(days=self.step_size_days)
        
        print(f"\n{'='*70}")
        print(f"Completed {len(window_results)} validation windows")
        
        # Aggregate results
        self.results = self._aggregate_results(window_results)
        
        return self.results
    
    def _default_validation_window(self,
                                   ticker: str,
                                   model_type: str,
                                   cal_start: datetime,
                                   cal_end: datetime,
                                   val_start: datetime,
                                   val_end: datetime) -> ValidationMetrics:
        """
        Default validation for a single window (mock implementation).
        
        In production, this would:
        1. Fetch market data for calibration period
        2. Run calibration
        3. Fetch validation data
        4. Compute out-of-sample errors
        
        For now, returns mock metrics as a demonstration.
        """
        # Mock calibration parameters
        if model_type == 'sabr':
            params = {
                'alpha': np.random.uniform(0.5, 1.5),
                'nu': np.random.uniform(0.2, 0.8),
                'rho': np.random.uniform(-0.5, 0.5)
            }
        else:
            params = {
                'v0': np.random.uniform(0.02, 0.06),
                'kappa': np.random.uniform(1.0, 3.0),
                'theta': np.random.uniform(0.02, 0.06),
                'xi': np.random.uniform(0.2, 0.5),
                'rho': np.random.uniform(-0.7, -0.3)
            }
        
        # Mock metrics
        return ValidationMetrics(
            calibration_end=cal_end,
            forecast_start=val_start,
            forecast_end=val_end,
            mean_price_error_pct=np.random.uniform(-2, 2),
            rmse_price_pct=np.random.uniform(3, 8),
            mae_price_pct=np.random.uniform(2, 6),
            max_price_error_pct=np.random.uniform(8, 15),
            mean_iv_error=np.random.uniform(-0.02, 0.02),
            rmse_iv=np.random.uniform(0.03, 0.08),
            mae_iv=np.random.uniform(0.02, 0.06),
            max_iv_error=np.random.uniform(0.08, 0.15),
            parameters=params,
            n_calibration_points=np.random.randint(20, 50),
            n_validation_points=np.random.randint(15, 40),
            calibration_objective=np.random.uniform(0.5, 1.5)
        )
    
    def _aggregate_results(self, window_results: List[ValidationMetrics]) -> ValidationSummary:
        """Aggregate validation metrics across all windows."""
        if not window_results:
            raise ValueError("No validation windows completed")
        
        # Average errors
        avg_price_rmse = np.mean([w.rmse_price_pct for w in window_results])
        avg_iv_rmse = np.mean([w.rmse_iv for w in window_results])
        
        # Parameter stability (std dev across windows)
        param_names = list(window_results[0].parameters.keys())
        param_stability = {}
        
        for param_name in param_names:
            param_series = [w.parameters[param_name] for w in window_results]
            param_stability[param_name] = np.std(param_series)
        
        # Best/worst windows
        best_idx = np.argmin([w.rmse_price_pct for w in window_results])
        worst_idx = np.argmax([w.rmse_price_pct for w in window_results])
        
        return ValidationSummary(
            n_windows=len(window_results),
            avg_price_rmse_pct=avg_price_rmse,
            avg_iv_rmse=avg_iv_rmse,
            parameter_stability=param_stability,
            best_window=window_results[best_idx],
            worst_window=window_results[worst_idx],
            all_windows=window_results
        )
    
    def print_summary(self):
        """Print validation summary report."""
        if self.results is None:
            raise ValueError("Must run validation first")
        
        print("\n" + "="*70)
        print("ROLLING WINDOW VALIDATION SUMMARY")
        print("="*70)
        print(f"\nTicker: {self.ticker}")
        print(f"Model: {self.model_type.upper()}")
        print(f"Windows validated: {self.results.n_windows}")
        
        print("\n" + "-"*70)
        print("OUT-OF-SAMPLE PERFORMANCE")
        print("-"*70)
        print(f"Average price RMSE: {self.results.avg_price_rmse_pct:.2f}%")
        print(f"Average IV RMSE: {self.results.avg_iv_rmse:.4f}")
        
        print("\n" + "-"*70)
        print("PARAMETER STABILITY (Std Dev Across Windows)")
        print("-"*70)
        for param, std_dev in self.results.parameter_stability.items():
            print(f"  {param}: {std_dev:.4f}")
        
        print("\n" + "-"*70)
        print("BEST WINDOW")
        print("-"*70)
        best = self.results.best_window
        print(f"Date: {best.forecast_start.date()}")
        print(f"Price RMSE: {best.rmse_price_pct:.2f}%")
        print(f"IV RMSE: {best.rmse_iv:.4f}")
        print(f"Parameters: {best.parameters}")
        
        print("\n" + "-"*70)
        print("WORST WINDOW")
        print("-"*70)
        worst = self.results.worst_window
        print(f"Date: {worst.forecast_start.date()}")
        print(f"Price RMSE: {worst.rmse_price_pct:.2f}%")
        print(f"IV RMSE: {worst.rmse_iv:.4f}")
        print(f"Parameters: {worst.parameters}")
    
    def plot_oos_performance(self, save_path: Optional[str] = None):
        """
        Plot out-of-sample performance over time.
        
        Args:
            save_path: Optional path to save figure
        """
        import matplotlib.pyplot as plt
        from matplotlib.dates import DateFormatter
        
        if self.results is None:
            raise ValueError("Must run validation first")
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        dates = [w.forecast_start for w in self.results.all_windows]
        price_rmse = [w.rmse_price_pct for w in self.results.all_windows]
        iv_rmse = [w.rmse_iv for w in self.results.all_windows]
        
        # Price RMSE over time
        axes[0].plot(dates, price_rmse, 'o-', linewidth=2, markersize=5)
        axes[0].axhline(self.results.avg_price_rmse_pct, color='red', linestyle='--', 
                       label=f'Average: {self.results.avg_price_rmse_pct:.2f}%')
        axes[0].set_ylabel('Price RMSE (%)', fontsize=12, fontweight='bold')
        axes[0].set_title(f'{self.ticker} - Out-of-Sample Performance', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # IV RMSE over time
        axes[1].plot(dates, iv_rmse, 'o-', color='green', linewidth=2, markersize=5)
        axes[1].axhline(self.results.avg_iv_rmse, color='red', linestyle='--',
                       label=f'Average: {self.results.avg_iv_rmse:.4f}')
        axes[1].set_ylabel('IV RMSE', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        # Parameter stability (example: first parameter)
        param_names = list(self.results.all_windows[0].parameters.keys())
        if param_names:
            param_name = param_names[0]
            param_values = [w.parameters[param_name] for w in self.results.all_windows]
            axes[2].plot(dates, param_values, 'o-', color='purple', linewidth=2, markersize=5)
            axes[2].set_ylabel(f'{param_name}', fontsize=12, fontweight='bold')
            axes[2].set_xlabel('Forecast Date', fontsize=12)
            axes[2].grid(alpha=0.3)
        
        for ax in axes:
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved OOS performance plot to {save_path}")
        
        plt.show()


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Demonstration of rolling window validation.
    
    Shows:
    1. Setup of validation windows
    2. Walk-forward analysis
    3. Out-of-sample error tracking
    4. Parameter stability assessment
    """
    
    print("ROLLING WINDOW VALIDATION - DEMONSTRATION")
    print("="*70)
    
    validator = RollingWindowValidator(
        calibration_window_days=30,
        forecast_horizon_days=5,
        step_size_days=5
    )
    
    # Run validation (with mock data for demonstration)
    results = validator.run_validation(
        ticker='SPY',
        model_type='sabr',
        start_date='2025-01-01',
        end_date='2025-03-31'
    )
    
    # Print summary
    validator.print_summary()
    
    # Visualize
    print("\n" + "="*70)
    print("GENERATING OOS PERFORMANCE VISUALIZATION")
    print("="*70)
    validator.plot_oos_performance(save_path='oos_performance.png')
