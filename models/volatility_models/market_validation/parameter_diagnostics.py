"""
Parameter Path Diagnostics for Volatility Model Calibration

Converts calibration snapshots into time-series data for temporal analysis:
- Autocorrelation of parameters (mean reversion vs trending)
- Variance of daily changes (stability diagnostics)
- Regime clustering (stress detection via parameter spikes)
- Jump detection in parameter paths

This module stands out by treating calibrated parameters as signals themselves,
not just static snapshots.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ParameterStats:
    """Statistical diagnostics for a single parameter time series."""
    name: str
    mean: float
    std: float
    min: float
    max: float
    autocorr_lag1: float
    autocorr_lag5: float
    ljung_box_pvalue: float  # Test for serial correlation
    daily_change_vol: float
    max_jump: float
    jump_dates: List[datetime]
    regime_labels: np.ndarray


@dataclass
class RegimeAnalysis:
    """Regime clustering results."""
    n_regimes: int
    regime_labels: np.ndarray
    regime_centers: np.ndarray
    regime_transitions: List[Tuple[datetime, int, int]]
    stress_regime_id: int  # Regime with highest volatility


class ParameterPathDiagnostics:
    """
    Analyze time-series properties of calibrated parameters.
    
    Key diagnostics:
    1. Autocorrelation structure (mean reversion vs persistence)
    2. Variance of daily changes (parameter stability)
    3. Regime clustering (stress identification)
    4. Jump detection (structural breaks)
    
    Usage:
        diagnostics = ParameterPathDiagnostics()
        diagnostics.load_calibration_history('SPY', 'sabr', start_date='2025-01-01')
        stats = diagnostics.compute_statistics()
        regimes = diagnostics.detect_regimes()
        diagnostics.plot_parameter_paths(save_path='param_paths.png')
    """
    
    def __init__(self):
        self.param_history: Optional[pd.DataFrame] = None
        self.ticker: Optional[str] = None
        self.model_type: Optional[str] = None
        self.param_names: List[str] = []
        
    def load_calibration_history(self,
                                 ticker: str,
                                 model_type: str,
                                 results_dir: str = '../calibration/calibration_results',
                                 start_date: Optional[str] = None,
                                 end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load historical calibration results from CSV files.
        
        Args:
            ticker: Ticker symbol
            model_type: 'sabr' or 'heston'
            results_dir: Directory containing calibration CSVs
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)
            
        Returns:
            DataFrame with columns: [date, param1, param2, ...]
        """
        import os
        import glob
        
        self.ticker = ticker
        self.model_type = model_type
        
        # Find all parameter CSV files for this ticker/model
        pattern = f"{results_dir}/{ticker}_{model_type}_params_*.csv"
        files = sorted(glob.glob(pattern))
        
        if not files:
            raise ValueError(f"No calibration history found for {ticker}/{model_type}")
        
        print(f"Found {len(files)} calibration snapshots")
        
        # Load and parse each file
        records = []
        for filepath in files:
            # Extract timestamp from filename: SPY_sabr_params_20260218_020115.csv
            basename = os.path.basename(filepath)
            timestamp_str = basename.split('_')[-2] + basename.split('_')[-1].replace('.csv', '')
            
            try:
                date = datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
            except ValueError:
                print(f"WARNING: Could not parse timestamp from {basename}, skipping")
                continue
            
            # Load parameters
            df = pd.read_csv(filepath)
            params = df.iloc[0].to_dict()
            params['date'] = date
            records.append(params)
        
        # Create time series DataFrame
        self.param_history = pd.DataFrame(records).sort_values('date').reset_index(drop=True)
        self.param_names = [col for col in self.param_history.columns if col != 'date']
        
        # Filter by date range
        if start_date:
            start = pd.to_datetime(start_date)
            self.param_history = self.param_history[self.param_history['date'] >= start]
        if end_date:
            end = pd.to_datetime(end_date)
            self.param_history = self.param_history[self.param_history['date'] <= end]
        
        print(f"Loaded {len(self.param_history)} calibrations from {self.param_history['date'].min()} to {self.param_history['date'].max()}")
        print(f"Parameters tracked: {self.param_names}")
        
        return self.param_history
    
    def compute_statistics(self, jump_threshold: float = 3.0) -> Dict[str, ParameterStats]:
        """
        Compute comprehensive statistics for each parameter.
        
        Args:
            jump_threshold: Number of std devs to classify as a jump
            
        Returns:
            Dictionary mapping parameter name to ParameterStats
        """
        if self.param_history is None:
            raise ValueError("Must call load_calibration_history() first")
        
        stats_dict = {}
        
        for param_name in self.param_names:
            series = self.param_history[param_name].values
            dates = self.param_history['date'].values
            
            # Basic statistics
            mean_val = np.mean(series)
            std_val = np.std(series)
            min_val = np.min(series)
            max_val = np.max(series)
            
            # Autocorrelation
            if len(series) > 5:
                acf_vals = acf(series, nlags=5, fft=False)
                autocorr_1 = acf_vals[1]
                autocorr_5 = acf_vals[5] if len(acf_vals) > 5 else np.nan
                
                # Ljung-Box test for serial correlation
                lb_result = acorr_ljungbox(series, lags=[5], return_df=False)
                ljung_box_pval = lb_result[1][0] if len(lb_result) > 1 else np.nan
            else:
                autocorr_1 = np.nan
                autocorr_5 = np.nan
                ljung_box_pval = np.nan
            
            # Daily changes
            daily_changes = np.diff(series)
            daily_change_vol = np.std(daily_changes)
            
            # Jump detection
            if len(daily_changes) > 0 and daily_change_vol > 0:
                z_scores = np.abs(daily_changes) / daily_change_vol
                jumps = z_scores > jump_threshold
                jump_indices = np.where(jumps)[0] + 1  # +1 because diff removes first element
                jump_dates = [dates[i] for i in jump_indices]
                max_jump = np.max(np.abs(daily_changes)) if len(daily_changes) > 0 else 0.0
            else:
                jump_dates = []
                max_jump = 0.0
            
            stats_dict[param_name] = ParameterStats(
                name=param_name,
                mean=mean_val,
                std=std_val,
                min=min_val,
                max=max_val,
                autocorr_lag1=autocorr_1,
                autocorr_lag5=autocorr_5,
                ljung_box_pvalue=ljung_box_pval,
                daily_change_vol=daily_change_vol,
                max_jump=max_jump,
                jump_dates=jump_dates,
                regime_labels=np.array([])  # Filled by detect_regimes()
            )
        
        return stats_dict
    
    def detect_regimes(self, n_regimes: int = 3) -> RegimeAnalysis:
        """
        Detect market regimes using hierarchical clustering on parameter vectors.
        
        Strategy: Cluster the multi-dimensional parameter space to identify
        distinct volatility regimes (e.g., low-vol, normal, stress).
        
        Args:
            n_regimes: Number of regimes to identify
            
        Returns:
            RegimeAnalysis object with clustering results
        """
        if self.param_history is None:
            raise ValueError("Must call load_calibration_history() first")
        
        # Standardize parameters for clustering
        param_matrix = self.param_history[self.param_names].values
        param_std = (param_matrix - param_matrix.mean(axis=0)) / param_matrix.std(axis=0)
        
        # Hierarchical clustering
        linkage_matrix = linkage(param_std, method='ward')
        labels = fcluster(linkage_matrix, n_regimes, criterion='maxclust')
        
        # Compute regime centers
        centers = np.array([param_matrix[labels == i].mean(axis=0) for i in range(1, n_regimes + 1)])
        
        # Identify stress regime (highest average parameter variance)
        regime_variances = np.array([param_matrix[labels == i].var(axis=0).mean() for i in range(1, n_regimes + 1)])
        stress_regime_id = np.argmax(regime_variances) + 1
        
        # Detect regime transitions
        dates = self.param_history['date'].values
        transitions = []
        for i in range(1, len(labels)):
            if labels[i] != labels[i-1]:
                transitions.append((dates[i], int(labels[i-1]), int(labels[i])))
        
        return RegimeAnalysis(
            n_regimes=n_regimes,
            regime_labels=labels,
            regime_centers=centers,
            regime_transitions=transitions,
            stress_regime_id=stress_regime_id
        )
    
    def identify_stress_periods(self,
                                nu_spike_threshold: float = 2.0,
                                rho_flip_threshold: float = 0.3) -> pd.DataFrame:
        """
        Identify stress periods based on parameter behavior.
        
        Common stress indicators:
        - ν (nu) spikes: High vol-of-vol during uncertainty
        - ρ (rho) flips: Correlation regime changes
        - α (alpha) surges: Elevated ATM volatility
        
        Args:
            nu_spike_threshold: Z-score threshold for ν spikes
            rho_flip_threshold: Absolute change threshold for ρ flips
            
        Returns:
            DataFrame with stress flags and severity scores
        """
        if self.param_history is None:
            raise ValueError("Must call load_calibration_history() first")
        
        df = self.param_history.copy()
        
        # ν spike detection (SABR)
        if 'nu' in self.param_names:
            nu_series = df['nu'].values
            nu_z = (nu_series - nu_series.mean()) / nu_series.std()
            df['nu_spike'] = np.abs(nu_z) > nu_spike_threshold
        
        # ρ flip detection (SABR/Heston)
        if 'rho' in self.param_names:
            rho_changes = np.abs(np.diff(df['rho'].values, prepend=df['rho'].values[0]))
            df['rho_flip'] = rho_changes > rho_flip_threshold
        
        # α surge detection (SABR)
        if 'alpha' in self.param_names:
            alpha_z = (df['alpha'] - df['alpha'].mean()) / df['alpha'].std()
            df['alpha_surge'] = alpha_z > 2.0
        
        # Overall stress score (0-1)
        stress_cols = [col for col in df.columns if col.endswith('_spike') or col.endswith('_flip') or col.endswith('_surge')]
        if stress_cols:
            df['stress_score'] = df[stress_cols].sum(axis=1) / len(stress_cols)
        else:
            df['stress_score'] = 0.0
        
        return df
    
    def print_diagnostics_report(self):
        """Print comprehensive diagnostics report."""
        stats = self.compute_statistics()
        regimes = self.detect_regimes()
        stress_df = self.identify_stress_periods()
        
        print("\n" + "="*70)
        print("PARAMETER PATH DIAGNOSTICS REPORT")
        print("="*70)
        print(f"\nTicker: {self.ticker}")
        print(f"Model: {self.model_type.upper()}")
        print(f"Period: {self.param_history['date'].min()} to {self.param_history['date'].max()}")
        print(f"Observations: {len(self.param_history)}")
        
        print("\n" + "-"*70)
        print("PARAMETER STATISTICS")
        print("-"*70)
        
        for param_name, stat in stats.items():
            print(f"\n{param_name.upper()}:")
            print(f"  Mean:              {stat.mean:.4f}")
            print(f"  Std Dev:           {stat.std:.4f}")
            print(f"  Range:             [{stat.min:.4f}, {stat.max:.4f}]")
            print(f"  Autocorr (lag 1):  {stat.autocorr_lag1:.4f}")
            print(f"  Autocorr (lag 5):  {stat.autocorr_lag5:.4f}")
            print(f"  Ljung-Box p-val:   {stat.ljung_box_pvalue:.4f}")
            print(f"  Daily change vol:  {stat.daily_change_vol:.4f}")
            print(f"  Max jump:          {stat.max_jump:.4f}")
            print(f"  Jump events:       {len(stat.jump_dates)}")
            
            if len(stat.jump_dates) > 0:
                print(f"  Jump dates:        {[str(d)[:10] for d in stat.jump_dates[:3]]}")
        
        print("\n" + "-"*70)
        print("REGIME ANALYSIS")
        print("-"*70)
        print(f"\nDetected {regimes.n_regimes} regimes")
        print(f"Stress regime ID: {regimes.stress_regime_id}")
        print(f"Regime transitions: {len(regimes.regime_transitions)}")
        
        if len(regimes.regime_transitions) > 0:
            print("\nRecent transitions:")
            for date, from_regime, to_regime in regimes.regime_transitions[-5:]:
                print(f"  {str(date)[:19]}: Regime {from_regime} -> {to_regime}")
        
        print("\n" + "-"*70)
        print("STRESS PERIOD IDENTIFICATION")
        print("-"*70)
        
        stress_periods = stress_df[stress_df['stress_score'] > 0.5]
        print(f"\nHigh stress periods: {len(stress_periods)} days")
        
        if len(stress_periods) > 0:
            print(f"Max stress score: {stress_df['stress_score'].max():.2f}")
            print(f"Avg stress score: {stress_df['stress_score'].mean():.2f}")
            print("\nTop 5 stress dates:")
            top_stress = stress_df.nlargest(5, 'stress_score')[['date', 'stress_score']]
            for _, row in top_stress.iterrows():
                print(f"  {str(row['date'])[:19]}: {row['stress_score']:.2f}")
    
    def plot_parameter_paths(self, save_path: Optional[str] = None):
        """
        Plot parameter time series with regime coloring and jump markers.
        
        Args:
            save_path: Optional path to save figure
        """
        import matplotlib.pyplot as plt
        from matplotlib.dates import DateFormatter
        
        if self.param_history is None:
            raise ValueError("Must call load_calibration_history() first")
        
        stats = self.compute_statistics()
        regimes = self.detect_regimes()
        
        n_params = len(self.param_names)
        fig, axes = plt.subplots(n_params, 1, figsize=(14, 4*n_params), sharex=True)
        
        if n_params == 1:
            axes = [axes]
        
        dates = self.param_history['date']
        
        for idx, param_name in enumerate(self.param_names):
            ax = axes[idx]
            values = self.param_history[param_name]
            
            # Plot parameter path with regime coloring
            for regime_id in range(1, regimes.n_regimes + 1):
                mask = regimes.regime_labels == regime_id
                regime_dates = dates[mask]
                regime_values = values[mask]
                
                color = 'red' if regime_id == regimes.stress_regime_id else f'C{regime_id-1}'
                ax.scatter(regime_dates, regime_values, c=color, alpha=0.6, s=30, 
                          label=f'Regime {regime_id}' + (' (Stress)' if regime_id == regimes.stress_regime_id else ''))
            
            ax.plot(dates, values, 'k-', alpha=0.3, linewidth=1)
            
            # Mark jumps
            jump_dates = stats[param_name].jump_dates
            if len(jump_dates) > 0:
                jump_values = [values[dates == jd].values[0] for jd in jump_dates if (dates == jd).any()]
                ax.scatter(jump_dates, jump_values, marker='x', c='black', s=100, linewidths=2, label='Jumps')
            
            # Add mean line
            ax.axhline(stats[param_name].mean, color='gray', linestyle='--', alpha=0.5, label='Mean')
            
            ax.set_ylabel(param_name, fontsize=12, fontweight='bold')
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(alpha=0.3)
            
            # Add autocorrelation annotation
            ac1 = stats[param_name].autocorr_lag1
            ax.text(0.02, 0.95, f'AC(1)={ac1:.3f}', transform=ax.transAxes, 
                   fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        axes[-1].set_xlabel('Date', fontsize=12)
        axes[-1].xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        fig.suptitle(f'{self.ticker} - {self.model_type.upper()} Parameter Paths with Regime Detection', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved parameter path plot to {save_path}")
        
        plt.show()


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Example usage: Analyze SPY SABR calibration history.
    
    Demonstrates:
    1. Loading historical calibration snapshots
    2. Computing autocorrelation and variance diagnostics
    3. Detecting market regimes via clustering
    4. Identifying stress periods
    5. Visualizing parameter paths
    """
    
    print("PARAMETER PATH DIAGNOSTICS - DEMONSTRATION")
    print("="*70)
    
    diagnostics = ParameterPathDiagnostics()
    
    try:
        # Load calibration history
        print("\nLoading calibration history...")
        diagnostics.load_calibration_history(
            ticker='SPY',
            model_type='sabr',
            results_dir='../calibration/calibration_results'
        )
        
        # Print comprehensive report
        diagnostics.print_diagnostics_report()
        
        # Generate visualization
        print("\n" + "="*70)
        print("GENERATING PARAMETER PATH VISUALIZATION")
        print("="*70)
        diagnostics.plot_parameter_paths(save_path='spy_parameter_paths.png')
        
    except ValueError as e:
        print(f"\nWARNING: {e}")
        print("\nTo use this tool, run multiple calibrations over time:")
        print("  cd ../calibration")
        print("  python run_calibration.py --ticker SPY --model sabr")
        print("  (repeat daily to build history)")
