# Trading Signals Module

**⚠️ CRITICAL DISCLAIMER: This is SIGNAL RESEARCH, not a deployable trading strategy.**

## What This Module Does

Generates trading signals based on calibrated volatility models (SABR, Heston) with:
- Reality checks (liquidity, spreads, strike availability)
- Transaction cost estimates
- Stress testing under adverse conditions
- Multi-leg strategy support

## What This Module Does NOT Do

**Real-world constraints NOT fully modeled:**
- ❌ Transaction costs (only rough estimates)
- ❌ Bid-ask spreads (partial modeling)
- ❌ Discrete strike availability (basic checks only)
- ❌ Liquidity constraints (simplified thresholds)
- ❌ Market impact
- ❌ Execution slippage
- ❌ Regulatory requirements
- ❌ Capital requirements
- ❌ Risk management systems

**Production deployment would require:**
- Live market data feeds
- Exchange-specific fee structures
- Real-time liquidity monitoring
- Execution management systems
- Compliance infrastructure
- Professional risk management

---

## Module Structure

```
signals/
├── signal_generator.py         # Core signal generation with reality checks
├── strategy_signals.py         # Multi-leg strategies (spreads, calendars)
├── run_signals.py             # Orchestration script
└── README.md                  # This file
```

---

## Quick Start

### 1. Generate Simple Signals

```bash
cd models/volatility_models/signals
python signal_generator.py
```

**Output:**
- Signal type (long/short volatility)
- IV differential (model vs market)
- Transaction cost estimates
- Net edge after costs
- Reality check results

### 2. Generate Strategy Signals

```bash
python strategy_signals.py
```

**Output:**
- Multi-leg strategy signals
- P&L analysis (max profit/loss, breakevens)
- Portfolio Greeks
- Cost analysis per leg

### 3. Full Signal Pipeline

```bash
python run_signals.py --ticker SPY --model sabr
```

**Pipeline:**
1. Load calibrated model parameters
2. Fetch current market data
3. Generate signals for all available strikes
4. Apply reality checks
5. Filter tradeable signals
6. Stress test viable signals
7. Export results

---

## Signal Types

### Single-Leg Signals

**Long Volatility:**
- `LONG_CALL`: OTM call (model IV > market IV, upside bet)
- `LONG_PUT`: OTM put (model IV > market IV, downside bet)
- `LONG_STRADDLE`: ATM straddle (expect large move)

**Short Volatility:**
- `SHORT_CALL`: OTM call (model IV < market IV)
- `SHORT_PUT`: OTM put (model IV < market IV)
- `SHORT_STRADDLE`: ATM straddle (expect range-bound)

### Multi-Leg Strategies

**Vertical Spreads:**
- `BULL_CALL_SPREAD`: Long lower strike, short upper strike
- `BEAR_PUT_SPREAD`: Long higher strike, short lower strike
- Limited risk, limited profit

**Calendar Spreads:**
- `CALENDAR_SPREAD`: Short near-term, long far-term
- Profit from time decay differential

**Volatility Spreads:**
- `LONG_STRANGLE`: OTM put + call (expect big move)
- `SHORT_STRANGLE`: OTM put + call (expect low vol)

**Skew Trades:**
- `RISK_REVERSAL`: Long call + short put (exploit skew)

---

## Signal Logic

### IV Differential Threshold

```python
if model_iv > market_iv + 0.02:  # 2% threshold
    signal = LONG_VOLATILITY
elif model_iv < market_iv - 0.02:
    signal = SHORT_VOLATILITY
else:
    signal = NO_TRADE  # Insufficient edge
```

### Reality Checks

**Liquidity:**
- Minimum open interest: 100 contracts
- Minimum daily volume: 20 contracts

**Spreads:**
- Maximum bid-ask spread: 10% of mid price

**Strike Availability:**
- Confirms strike exists in market data

### Transaction Cost Estimates

**Bid-Ask Spread:**
```python
spread_cost_bps = (ask - bid) / mid * 10000 / 2
```

**Commission:**
```python
commission_bps = ~130 bps per contract
# Assumes $0.65/contract, ~$50 notional
```

**Net Edge:**
```python
net_edge_bps = raw_edge_bps - spread_cost_bps - commission_bps
```

**Minimum Net Edge:**
- 50 bps (0.5%) required to generate signal

---

## Stress Testing

### Cost Sensitivity

Tests signal viability under:
- **2x spread cost**: Doubled bid-ask spread
- **3x spread cost**: Tripled bid-ask spread
- **2x commission**: Doubled transaction fees
- **2x all costs**: Both spread and commission doubled

### Liquidity Sensitivity

Tests signal under degraded liquidity:
- **50% OI drop**: Open interest halved
- **50% volume drop**: Daily volume halved
- **Both 50% drop**: Combined liquidity shock

### Example Output

```
SIGNAL STRESS TEST REPORT
======================================================================

COST SENSITIVITY (Net Edge in bps)
----------------------------------------------------------------------
✓ base_case            :      120 bps
✓ spread_2x            :       80 bps
✗ spread_3x            :       40 bps
✓ commission_2x        :       90 bps
✓ all_costs_2x         :       60 bps

LIQUIDITY SENSITIVITY
----------------------------------------------------------------------
✓ PASS   base_case
✓ PASS   oi_50pct_drop
✗ FAIL   volume_50pct_drop
✗ FAIL   both_50pct_drop

======================================================================
```

---

## Integration with Calibration

### Workflow

1. **Calibrate Model** (see `calibration/`)
   ```bash
   python run_calibration.py --ticker SPY --model sabr
   ```

2. **Generate Signals**
   ```bash
   python run_signals.py --ticker SPY --model sabr
   ```

3. **Validate Signals** (see `market_validation/`)
   ```bash
   python run_all_validations.py --ticker SPY --model sabr
   ```

### Data Flow

```
Calibration → Model Parameters (α, ν, ρ)
              ↓
Market Data → Current IV Surface
              ↓
Signal Generator → Model IV vs Market IV
              ↓
Reality Checks → Liquidity, Spreads, Costs
              ↓
Tradeable Signals → Filtered by net edge
              ↓
Stress Tests → Sensitivity analysis
```

---

## Example Output

```
======================================================================
TRADING SIGNAL REPORT (RESEARCH ONLY)
======================================================================

Timestamp: 2026-02-28 14:30:00
Signal: LONG_CALL
Confidence: 75.00%

======================================================================
VOLATILITY ANALYSIS
======================================================================
Model IV: 0.2500 (25.00%)
Market IV: 0.2200 (22.00%)
IV Differential: 3.00%
Raw Edge: 300 bps

======================================================================
TRANSACTION COST ESTIMATES
======================================================================
Bid-Ask Spread: 120 bps
Commission: 130 bps
Total Costs: 250 bps

✓ NET EDGE: 50 bps

======================================================================
REALITY CHECKS
======================================================================
Liquidity: ✓ PASS
Spread: ✓ PASS
Strike Available: ✓ PASS

======================================================================
RATIONALE
======================================================================
LONG_CALL: Model IV=0.250 vs Market IV=0.220. Net edge: 50 bps after 
costs. Market underpricing volatility by 3%, presenting potential 
opportunity.

======================================================================
⚠️ DISCLAIMER
======================================================================
This is SIGNAL RESEARCH, not a trading recommendation.
Real trading requires:
  - Live market data and execution systems
  - Full transaction cost modeling
  - Risk management and position limits
  - Regulatory compliance
======================================================================
```

---

## What Makes This Stand Out

### vs Typical Student Projects

**Most students do:**
- ❌ Ignore transaction costs entirely
- ❌ Assume perfect liquidity
- ❌ No bid-ask spread consideration
- ❌ "Buy if model > market" with no edge threshold
- ❌ No stress testing

**This project does:**
- ✅ Explicit transaction cost estimates
- ✅ Liquidity checks (OI, volume)
- ✅ Bid-ask spread modeling (partial)
- ✅ Minimum net edge threshold (50 bps)
- ✅ Comprehensive stress testing
- ✅ **Honest disclaimers** about limitations

### What Interviewers Notice

**Red Flags (Avoided):**
- ❌ Claiming "profitable strategy" without costs
- ❌ Backtests with unrealistic fills
- ❌ Ignoring market microstructure

**Green Flags (Demonstrated):**
- ✅ Upfront about limitations
- ✅ Conservative cost assumptions
- ✅ Stress testing under adverse conditions
- ✅ Clear distinction: RESEARCH vs TRADING
- ✅ Understanding of real-world constraints

**Key Talking Points:**
1. "I model spreads and costs, but acknowledge they're simplified"
2. "Stress tests show signals break down at 3x spreads"
3. "Liquidity checks filter out illiquid options"
4. "This is research-grade, not production-ready"
5. "Real trading needs live data feeds and execution infrastructure"

---

## Advanced Features

### Portfolio Greeks

For multi-leg strategies, computes net Greeks:
- **Delta**: Directional exposure
- **Gamma**: Convexity
- **Vega**: Volatility exposure
- **Theta**: Time decay

### Strategy-Specific Analysis

**Bull Call Spread:**
- Max profit = (upper_strike - lower_strike - net_debit)
- Max loss = net_debit
- Breakeven = lower_strike + net_debit

**Calendar Spread:**
- Max profit at near expiration if spot = strike
- Theta decay analysis

**Strangle:**
- Two breakevens (put_strike - debit, call_strike + debit)
- Unlimited profit potential (long)
- Limited profit, unlimited loss (short)

---

## Configuration

### Reality Check Config

```python
config = RealityCheckConfig(
    min_open_interest=100,       # Minimum OI
    max_bid_ask_spread_pct=0.10, # Max 10% spread
    min_volume=20,               # Min daily volume
    avg_commission_per_contract=0.65,
    avg_spread_cost_pct=0.05,    # Conservative 5%
    min_net_edge_bps=50          # 0.5% minimum net edge
)
```

### Customization

Adjust thresholds based on:
- Asset class (equity, FX, rates)
- Market liquidity regime
- Risk tolerance
- Desired signal frequency

---

## Limitations & Future Work

### Current Limitations

1. **Cost Modeling**: Rough estimates, not live data
2. **Liquidity**: Binary checks (pass/fail), not continuous
3. **Market Impact**: Not modeled
4. **Execution**: Assumes instant fills at mid
5. **Risk Management**: No position sizing or stops

### Production Requirements

**Infrastructure:**
- Live market data feeds (Level 2)
- Order management system (OMS)
- Execution management system (EMS)
- Real-time risk monitoring

**Costs:**
- Exchange-specific fee schedules
- Prime broker costs
- Clearing fees
- Regulatory fees (SEC, FINRA)

**Risk:**
- Position limits by Greeks
- VaR/CVaR monitoring
- Correlation analysis
- Scenario stress testing

**Compliance:**
- Pattern day trader rules
- Margin requirements
- Reporting obligations

---

## References

### Transaction Costs
- Harris, L. (2003). *Trading and Exchanges*
- Hasbrouck, J. (2007). *Empirical Market Microstructure*

### Option Trading
- Natenberg, S. (1994). *Option Volatility and Pricing*
- Taleb, N. (1997). *Dynamic Hedging*

### Volatility Trading
- Gatheral, J. (2006). *The Volatility Surface*
- Sinclair, E. (2013). *Volatility Trading*

---

## Contact & Disclaimer

**Author**: Aarya Sharma  
**Purpose**: Educational research only  
**Status**: Signal research, NOT production trading system

⚠️ **DO NOT use for live trading without:**
- Professional risk management
- Live market data infrastructure
- Regulatory compliance
- Adequate capital and risk controls

This module demonstrates understanding of volatility trading concepts while being explicit about real-world limitations. It is designed for portfolio demonstration and learning purposes only.
