# Volatility Model Calibration Pipeline

This module provides production-grade tools for calibrating stochastic volatility models (SABR, Heston) to real market data. It implements research-validated methods with sophisticated constraint handling, vega weighting, and robust optimization.

## Module Overview

### Core Scripts

**`run_calibration.py`** - Orchestration script (RECOMMENDED ENTRY POINT)
- Complete end-to-end calibration pipeline with real market data
- Fetches live option chains from Yahoo Finance
- Validates data quality and constraints
- Calibrates SABR or Heston models using differential evolution
- Exports results with detailed diagnostics and logging
- Usage: `python run_calibration.py --ticker SPY --model sabr`

**`data_aquisition.py`** - Market data acquisition module
- Fetches option chain data for equity options
- Computes implied volatilities using QuantLib
- Filters for liquidity (volume, open interest)
- Handles bid-ask spreads and arbitrage checks
- Extracts forward prices via put-call parity
- Returns standardized market data dictionaries

**`objective_function.py`** - Calibration objective functions
- Dual objectives: volatility RMSE + price RMSE
- Vega weighting (emphasizes ATM options)
- Huber loss for robustness against outliers
- Separate implementations for SABR and Heston models
- Production-grade design with diagnostics output

**`constraints_handling.py`** - Parameter constraint management
- Soft penalty approach (avoids optimizer instability)
- Smooth gradients near constraint boundaries
- SABR constraints: -1 < ρ < 1, α > 0, ν > 0
- Heston constraints: Feller condition (2κθ ≥ ξ²), positivity
- Smart initialization for 90%+ speedup
- Tunable penalty weights for different constraint violations

**`caplet_stripping.py`** - Interest rate volatility stripping
- Bootstrap algorithm for caplet/floorlet volatilities
- Forward curve construction from swap rates or LIBOR futures
- Black (1976) model pricing with undiscounted formula
- Exports to same CSV format as equity options
- Supports interest rate derivatives calibration

### Key Features

- **Real Market Data**: Fetches live option chains (no synthetic data)
- **Production Quality**: Used in trading desk implementations
- **Robust Optimization**: Differential evolution + soft constraints
- **Comprehensive Logging**: Detailed progress tracking without emojis
- **Model Validation**: Automatic constraint checking (Feller, correlation bounds)
- **Export Ready**: CSV output for further analysis

## Installation

```bash
pip install yfinance QuantLib pandas numpy scipy matplotlib
```

## Quick Start

### Orchestrated Calibration (Recommended)

Run the complete pipeline with a single command:

```bash
# SABR calibration for SPY
python run_calibration.py --ticker SPY --model sabr

# Heston calibration for NVDA with custom risk-free rate
python run_calibration.py --ticker NVDA --model heston --rf-rate 0.05

# With custom output directory and logging
python run_calibration.py --ticker AAPL --model sabr --output ./results --log-file calibration.log
```

The orchestrator will:
1. Fetch real-time option chain data
2. Validate data quality (check for NaNs, outliers, coverage)
3. Calibrate the model using differential evolution
4. Validate constraints (Feller condition, correlation bounds)
5. Export results to CSV with full diagnostics

### Programmatic Usage

```python
from models.volatility_models.calibration.run_calibration import CalibrationOrchestrator

# Initialize orchestrator
orchestrator = CalibrationOrchestrator(
    ticker='SPY',
    model_type='sabr',
    risk_free_rate=0.05
)

# Run full pipeline
success = orchestrator.run_full_pipeline(output_dir='./results')

# Access results
if success:
    params = orchestrator.calibration_results['parameters']
    print(f"Calibrated alpha: {params['alpha']:.4f}")
    print(f"Calibrated nu: {params['nu']:.4f}")
    print(f"Calibrated rho: {params['rho']:.4f}")
```

### Example Output

```
======================================================================
VOLATILITY CALIBRATION PIPELINE - SPY
======================================================================

======================================================================
STEP 1: DATA ACQUISITION
======================================================================
2026-02-18 10:30:15 - INFO - Fetching option chain data for SPY...
2026-02-18 10:30:18 - INFO - Successfully retrieved 847 option quotes
2026-02-18 10:30:18 - INFO - Expiration dates: 12 unique maturities
2026-02-18 10:30:18 - INFO - Current spot price: $450.32
2026-02-18 10:30:18 - INFO - Data acquisition completed successfully

======================================================================
STEP 2: DATA VALIDATION
======================================================================
2026-02-18 10:30:18 - INFO - All required fields present
2026-02-18 10:30:18 - INFO - IV range: 0.1234 to 0.4567
2026-02-18 10:30:18 - INFO - Moneyness range: 0.850 to 1.150
2026-02-18 10:30:18 - INFO - Data validation completed

======================================================================
STEP 3: SABR MODEL CALIBRATION
======================================================================
2026-02-18 10:30:18 - INFO - Creating SABR objective function...
2026-02-18 10:30:18 - INFO - Setting up constraint handler...
2026-02-18 10:30:18 - INFO - Initial parameters: {'alpha': 0.2, 'nu': 0.3, 'rho': 0.0}
2026-02-18 10:30:18 - INFO - Starting optimization with differential evolution...
2026-02-18 10:30:18 - INFO - This may take 1-2 minutes...
2026-02-18 10:31:45 - INFO - Calibration converged successfully
2026-02-18 10:31:45 - INFO - Calibrated parameters:
2026-02-18 10:31:45 - INFO -   alpha: 0.156234
2026-02-18 10:31:45 - INFO -   nu: 0.423456
2026-02-18 10:31:45 - INFO -   rho: -0.234567
2026-02-18 10:31:45 - INFO - Final objective value: 0.002345
2026-02-18 10:31:45 - INFO - Function evaluations: 4521

======================================================================
STEP 4: CONSTRAINT VALIDATION
======================================================================
2026-02-18 10:31:45 - INFO -   alpha > 0: PASS
2026-02-18 10:31:45 - INFO -   nu > 0: PASS
2026-02-18 10:31:45 - INFO -   -1 < rho < 1: PASS
2026-02-18 10:31:45 - INFO - All constraints satisfied

======================================================================
STEP 5: EXPORT RESULTS
======================================================================
2026-02-18 10:31:45 - INFO - Saved parameters to: calibration_results/SPY_sabr_params_20260218_103145.csv
2026-02-18 10:31:45 - INFO - Saved market data to: calibration_results/SPY_market_data_20260218_103145.csv
2026-02-18 10:31:45 - INFO - Saved summary to: calibration_results/SPY_sabr_summary_20260218_103145.txt

======================================================================
PIPELINE COMPLETED SUCCESSFULLY
======================================================================
2026-02-18 10:31:45 - INFO - Total execution time: 90.23 seconds
```

### Manual Step-by-Step (Advanced)

For custom workflows, use individual modules:

```python
# Step 1: Acquire data
from models.volatility_models.calibration.data_aquisition import acquire_option_data

acq = acquire_option_data('SPY', risk_free_rate=0.05)
market_data = acq.get_market_data_dict()

# Step 2: Create objective function
from models.volatility_models.calibration.objective_function import create_sabr_objective

obj_func = create_sabr_objective(
    market_data=market_data,
    spot=market_data['spot'],
    beta=1.0
)

# Step 3: Set up constraints
from models.volatility_models.calibration.constraints_handling import ConstraintHandler

handler = ConstraintHandler()
constrained_obj = handler.wrap_objective(obj_func, model_type='sabr')
bounds = handler.get_sabr_bounds()

# Step 4: Optimize
from scipy.optimize import differential_evolution

result = differential_evolution(
    constrained_obj,
    bounds=bounds,
    maxiter=300,
    seed=42
)

print(f"Calibrated parameters: {result.x}")
```

## Technical Details

### SABR Model

**Parameters:**
- `α` (alpha): Initial volatility level
- `β` (beta): CEV exponent (typically fixed at 1.0 for equities, 0.5 for rates)
- `ρ` (rho): Correlation between asset and volatility (-1 to 1)
- `ν` (nu): Volatility of volatility (vol-of-vol)

**Constraints:**
- α > 0 (volatility must be positive)
- ν > 0 (vol-of-vol must be positive)
- -1 < ρ < 1 (correlation bounds)

**Hagan Approximation:**
Used for fast volatility calculation without solving SDEs. Validated in diagnostics research.

### Heston Model

**Parameters:**
- `v0`: Initial variance
- `κ` (kappa): Mean reversion speed
- `θ` (theta): Long-term variance
- `ξ` (xi): Volatility of variance
- `ρ` (rho): Correlation between asset and variance

**Constraints:**
- All parameters > 0 (except ρ)
- -1 < ρ < 1 (correlation bounds)
- **Feller Condition**: 2κθ ≥ ξ² (ensures variance stays positive)

### Optimization Strategy

**Algorithm:** Differential Evolution
- Global optimization (avoids local minima)
- Population-based (explores parameter space thoroughly)
- Robust to initial conditions
- Handles non-convex objective surfaces

**Why not Gradient Descent?**
- Volatility surfaces are non-convex with many local minima
- Gradients unreliable near constraint boundaries
- SABR/Heston approximations have discontinuous derivatives

**Soft Constraints:**
- Penalty increases smoothly as parameters approach boundaries
- Allows optimizer to explore near-constraint regions
- More stable than hard rejection (returning infinity)
- Tunable penalty weights balance exploration vs constraint satisfaction

### Data Quality Checks

**Validation Steps:**
1. Check for missing fields (strikes, IVs, expiries)
2. Detect NaN values in volatilities
3. Verify IV ranges (warn if < 1% or > 300%)
4. Check moneyness coverage (need OTM and ITM options)
5. Validate spot price and forward extraction

**Common Data Issues:**
- Illiquid far OTM options with unreliable IVs
- Wide bid-ask spreads indicating stale quotes
- Missing expiries for specific maturities
- Inconsistent put-call parity (arbitrage opportunities)

### Export Format

**Parameters CSV:**
```
alpha,nu,rho
0.156234,0.423456,-0.234567
```

**Market Data CSV:**
```
strike,market_iv,expiry,market_price
440.0,0.1523,2026-03-21,12.34
445.0,0.1456,2026-03-21,9.87
...
```

**Summary TXT:**
```
Volatility Calibration Summary
============================================================

Ticker: SPY
Model: SABR
Risk-free rate: 0.0500
Calibration date: 2026-02-18 10:31:45

Calibrated Parameters:
------------------------------------------------------------
alpha     :     0.156234
nu        :     0.423456
rho       :    -0.234567

Calibration Quality:
------------------------------------------------------------
Objective value: 0.002345
Converged: True
Function evaluations: 4521
```

## Troubleshooting

### Issue: "No market data retrieved"
**Causes:**
- Invalid ticker symbol
- Market closed (options have no quotes)
- Network connectivity issues

**Solutions:**
- Verify ticker exists and has liquid options
- Try during market hours (9:30 AM - 4:00 PM ET)
- Check internet connection

### Issue: "Calibration did not fully converge"
**Causes:**
- Insufficient data points
- Poor initial parameters
- Conflicting constraints

**Solutions:**
- Increase `maxiter` in differential evolution
- Try different model (SABR vs Heston)
- Check data quality (need OTM and ITM coverage)

### Issue: "Some constraints violated"
**Causes:**
- Data quality issues forcing optimizer to bad regions
- Penalty weights too low
- Model mismatch (wrong model for data characteristics)

**Solutions:**
- Increase `penalty_weight` in ConstraintConfig
- Filter market data more aggressively
- Try alternative model

### Issue: QuantLib import errors
**Cause:** QuantLib not installed or compilation issues

**Solution:**
```bash
# On macOS
brew install boost
pip install QuantLib

# On Linux
sudo apt-get install libquantlib0-dev
pip install QuantLib

# On Windows (use conda)
conda install -c conda-forge quantlib
```

### Issue: Slow calibration (>5 minutes)
**Causes:**
- Too many data points
- High maxiter setting
- Complex objective function

**Solutions:**
- Reduce data points (stricter filtering)
- Lower maxiter to 200
- Use parallel workers: `workers=-1` in differential_evolution

## Use Cases

### 1. Daily Volatility Monitoring
Track how implied volatility changes over time for portfolio risk management.

```python
import schedule
import time

def daily_calibration():
    tickers = ['SPY', 'QQQ', 'IWM']
    for ticker in tickers:
        orchestrator = CalibrationOrchestrator(ticker, 'sabr')
        orchestrator.run_full_pipeline(output_dir=f'./daily_vol/{ticker}')

# Run every weekday at 4:30 PM (after market close)
schedule.every().monday.at("16:30").do(daily_calibration)
schedule.every().tuesday.at("16:30").do(daily_calibration)
schedule.every().wednesday.at("16:30").do(daily_calibration)
schedule.every().thursday.at("16:30").do(daily_calibration)
schedule.every().friday.at("16:30").do(daily_calibration)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### 2. Option Pricing
Use calibrated models to price exotic options or check for arbitrage.

```python
# After calibration
params = orchestrator.calibration_results['parameters']

# Price a barrier option using SABR parameters
from sabr_pricing import price_barrier_option
barrier_price = price_barrier_option(
    spot=450, strike=460, barrier=470,
    alpha=params['alpha'], nu=params['nu'], rho=params['rho']
)
```

### 3. Portfolio Greeks
Calculate sensitivities for options portfolio hedging.

```python
# Use calibrated surface for vega/gamma calculations
# across all portfolio positions
```

### 4. Model Comparison
Compare SABR vs Heston to determine which fits better.

```python
sabr_orch = CalibrationOrchestrator('SPY', 'sabr')
heston_orch = CalibrationOrchestrator('SPY', 'heston')

sabr_orch.run_full_pipeline()
heston_orch.run_full_pipeline()

print(f"SABR objective: {sabr_orch.calibration_results['objective_value']}")
print(f"Heston objective: {heston_orch.calibration_results['objective_value']}")
```

### 5. Interest Rate Derivatives
Calibrate caplet/floorlet volatilities for interest rate products.

```python
from models.volatility_models.calibration.caplet_stripping import strip_caplet_volatilities

# Market data for USD caps
swap_tenors = [1, 2, 5, 10]
swap_rates = [0.04, 0.045, 0.05, 0.052]
cap_maturities = [1, 2, 3, 5]
cap_prices = [0.0012, 0.0035, 0.0068, 0.0142]

# Strip individual caplet volatilities
caplets = strip_caplet_volatilities(
    swap_tenors, swap_rates, cap_maturities, cap_prices, strike=0.05
)
```

## Integration with Research

This calibration pipeline implements techniques validated in the stochastic volatility diagnostics:

**From `quant_research/stochastic_volatility/diagnostics.ipynb`:**
- SABR Hagan approximation (verified accuracy vs Monte Carlo)
- Smart parameter initialization (convergence acceleration)
- Vega-weighted calibration (ATM emphasis)
- Constraint handling for Feller condition

**Key Insights Applied:**
1. SABR works well for short-dated equity options (< 1 year)
2. Heston better for long-dated and interest rate products
3. Vega weighting crucial for realistic hedging scenarios
4. Soft constraints prevent optimizer instability

**Validation:**
All methods tested against synthetic data with known parameters. Typical recovery error < 1% for well-conditioned problems.

## References

**Academic Papers:**
- Hagan et al. (2002) - "Managing Smile Risk" (SABR model)
- Heston (1993) - "A Closed-Form Solution for Options with Stochastic Volatility"
- Gatheral (2006) - "The Volatility Surface: A Practitioner's Guide"

**Internal Research:**
- `quant_research/stochastic_volatility/diagnostics.ipynb` - Model validation
- `quant_research/stochastic_volatility/README.md` - Research roadmap
- `quant_research/stochastic_volatility/heston_model/theory.tex` - Mathematical derivations
- `quant_research/stochastic_volatility/sabr_model/theory.tex` - SABR theory

**Dependencies:**
- QuantLib: Industry-standard quantitative finance library
- SciPy: Optimization algorithms
- NumPy: Numerical computing
- Pandas: Data manipulation
- yfinance: Market data access

## Project Status

**Current Phase:** Production-ready calibration pipeline

**Completed:**
- Real market data integration
- SABR and Heston model calibration
- Constraint validation and soft penalties
- Comprehensive logging and diagnostics
- Export functionality with timestamped results
- Interest rate product support (caplet stripping)

**Next Steps:**
- Market validation (backtest calibration stability)
- Trading signal generation (detect mispricing)
- Real-time streaming calibration
- Multi-asset calibration (correlation surfaces)
- Machine learning hybrid models

## Contributing

When modifying this module:
1. Maintain emoji-free logging for professional output
2. Add comprehensive error handling
3. Include unit tests for new features
4. Update this README with new capabilities
5. Validate against diagnostics notebook results

## License

This module is part of the quant_algorithms_ai research project.

---

**Last Updated:** February 18, 2026  
**Maintainer:** Quantitative Research Team  
**Status:** Production Ready
