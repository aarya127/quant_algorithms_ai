"""
Trading Signal Generation - Research Module

⚠️ IMPORTANT: This is SIGNAL RESEARCH, not a deployable trading strategy.

Real-world constraints NOT fully modeled:
- Transaction costs (commissions, fees)
- Bid-ask spreads
- Discrete strike availability
- Liquidity constraints
- Market impact
- Execution slippage

This module generates signals based on calibrated volatility models and validates
them under partial stress-testing. Production deployment would require full
implementation of execution costs and market microstructure.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class SignalType(Enum):
    """Trading signal types."""
    LONG_CALL = "long_call"
    SHORT_CALL = "short_call"
    LONG_PUT = "long_put"
    SHORT_PUT = "short_put"
    LONG_STRADDLE = "long_straddle"
    SHORT_STRADDLE = "short_straddle"
    BULL_SPREAD = "bull_spread"
    BEAR_SPREAD = "bear_spread"
    CALENDAR_SPREAD = "calendar_spread"
    NO_TRADE = "no_trade"


@dataclass
class TradingSignal:
    """
    Trading signal with metadata and reality checks.
    
    Attributes:
        timestamp: Signal generation time
        signal_type: Type of trade (long/short, strategy)
        confidence: Signal strength [0, 1]
        strikes: Strike prices involved
        maturities: Expiration dates
        model_iv: Model-implied volatility
        market_iv: Market-implied volatility
        edge_bps: Expected edge in basis points (BEFORE costs)
        
        # Reality check flags
        liquidity_adequate: True if open interest sufficient
        spread_reasonable: True if bid-ask spread < threshold
        strike_available: True if strikes exist in market
        
        # Cost estimates
        estimated_spread_cost_bps: Bid-ask spread cost
        estimated_commission_bps: Commission cost
        estimated_total_cost_bps: Total transaction cost
        
        # Net edge after costs
        net_edge_bps: edge_bps - estimated_total_cost_bps
    """
    timestamp: datetime
    signal_type: SignalType
    confidence: float
    strikes: List[float]
    maturities: List[float]
    model_iv: float
    market_iv: float
    edge_bps: float
    
    # Reality checks
    liquidity_adequate: bool
    spread_reasonable: bool
    strike_available: bool
    
    # Cost estimates
    estimated_spread_cost_bps: float
    estimated_commission_bps: float
    estimated_total_cost_bps: float
    
    # Net edge
    net_edge_bps: float
    
    # Rationale
    signal_rationale: str


@dataclass
class RealityCheckConfig:
    """
    Configuration for reality checks.
    
    These are CONSERVATIVE estimates. Real trading would require:
    - Live market data feeds
    - Exchange-specific fee structures
    - Dynamic liquidity monitoring
    - Real-time spread tracking
    """
    min_open_interest: int = 100  # Minimum OI for liquidity
    max_bid_ask_spread_pct: float = 0.10  # Max 10% spread
    min_volume: int = 20  # Minimum daily volume
    
    # Cost assumptions (ROUGH ESTIMATES)
    avg_commission_per_contract: float = 0.65  # Per contract per side
    avg_spread_cost_pct: float = 0.05  # 5% of mid price (conservative)
    
    # Minimum edge required after costs
    min_net_edge_bps: float = 50  # 0.5% minimum net edge


class VolatilitySignalGenerator:
    """
    Generate trading signals from volatility model calibrations.
    
    **SIGNAL LOGIC:**
    - Model IV > Market IV + threshold → Market underpricing volatility → Long volatility
    - Model IV < Market IV - threshold → Market overpricing volatility → Short volatility
    
    **REALITY CHECKS:**
    - Liquidity: Check open interest, volume
    - Spreads: Validate bid-ask spread within threshold
    - Availability: Confirm strikes exist in market
    - Costs: Estimate transaction costs and net edge
    
    **⚠️ DISCLAIMER:**
    This is for research purposes only. Real trading requires:
    - Live market data and execution infrastructure
    - Risk management systems
    - Compliance and regulatory oversight
    - Professional trading capital
    """
    
    def __init__(self, config: Optional[RealityCheckConfig] = None):
        """
        Initialize signal generator.
        
        Args:
            config: Reality check configuration
        """
        self.config = config or RealityCheckConfig()
        self.signals: List[TradingSignal] = []
    
    def generate_signal(self,
                       spot: float,
                       strike: float,
                       maturity: float,
                       model_iv: float,
                       market_iv: float,
                       market_data: Optional[Dict] = None) -> TradingSignal:
        """
        Generate trading signal with reality checks.
        
        Args:
            spot: Current spot price
            strike: Strike price
            maturity: Time to maturity (years)
            model_iv: Model-calibrated implied volatility
            market_iv: Market-observed implied volatility
            market_data: Optional market data (volume, OI, bid/ask)
            
        Returns:
            TradingSignal with reality check flags
        """
        timestamp = datetime.now()
        
        # Calculate raw edge (in basis points)
        iv_diff = model_iv - market_iv
        edge_bps = iv_diff * 10000  # Convert to basis points
        
        # Determine signal type based on IV differential
        signal_type, confidence = self._determine_signal_type(
            iv_diff, spot, strike, maturity
        )
        
        # Reality checks
        liquidity_ok, spread_ok, strike_ok = self._reality_checks(
            strike, maturity, market_data
        )
        
        # Estimate transaction costs
        spread_cost = self._estimate_spread_cost(spot, strike, market_data)
        commission_cost = self._estimate_commission_cost()
        total_cost = spread_cost + commission_cost
        
        # Net edge after costs
        net_edge = edge_bps - total_cost
        
        # Signal rationale
        rationale = self._generate_rationale(
            signal_type, model_iv, market_iv, net_edge,
            liquidity_ok, spread_ok, strike_ok
        )
        
        signal = TradingSignal(
            timestamp=timestamp,
            signal_type=signal_type,
            confidence=confidence,
            strikes=[strike],
            maturities=[maturity],
            model_iv=model_iv,
            market_iv=market_iv,
            edge_bps=edge_bps,
            liquidity_adequate=liquidity_ok,
            spread_reasonable=spread_ok,
            strike_available=strike_ok,
            estimated_spread_cost_bps=spread_cost,
            estimated_commission_bps=commission_cost,
            estimated_total_cost_bps=total_cost,
            net_edge_bps=net_edge,
            signal_rationale=rationale
        )
        
        self.signals.append(signal)
        return signal
    
    def _determine_signal_type(self,
                               iv_diff: float,
                               spot: float,
                               strike: float,
                               maturity: float) -> Tuple[SignalType, float]:
        """
        Determine signal type and confidence based on IV differential.
        
        Logic:
        - IV_diff > 0.02 (2%): Market underpricing → Long volatility
        - IV_diff < -0.02: Market overpricing → Short volatility
        - |IV_diff| < 0.02: No trade (insufficient edge)
        """
        threshold = 0.02  # 2% IV differential threshold
        
        if abs(iv_diff) < threshold:
            return SignalType.NO_TRADE, 0.0
        
        # Confidence scales with IV differential magnitude
        confidence = min(abs(iv_diff) / 0.10, 1.0)  # Max confidence at 10% diff
        
        moneyness = strike / spot
        
        if iv_diff > threshold:
            # Model predicts higher volatility → Long volatility
            if 0.95 <= moneyness <= 1.05:
                # ATM → Straddle
                return SignalType.LONG_STRADDLE, confidence
            elif moneyness > 1.05:
                # OTM call
                return SignalType.LONG_CALL, confidence
            else:
                # OTM put
                return SignalType.LONG_PUT, confidence
        else:
            # Model predicts lower volatility → Short volatility
            if 0.95 <= moneyness <= 1.05:
                return SignalType.SHORT_STRADDLE, confidence
            elif moneyness > 1.05:
                return SignalType.SHORT_CALL, confidence
            else:
                return SignalType.SHORT_PUT, confidence
    
    def _reality_checks(self,
                       strike: float,
                       maturity: float,
                       market_data: Optional[Dict]) -> Tuple[bool, bool, bool]:
        """
        Perform reality checks on market conditions.
        
        Returns:
            (liquidity_adequate, spread_reasonable, strike_available)
        """
        if market_data is None:
            # No market data → Assume worst case
            return False, False, False
        
        # Liquidity check
        open_interest = market_data.get('open_interest', 0)
        volume = market_data.get('volume', 0)
        liquidity_ok = (
            open_interest >= self.config.min_open_interest and
            volume >= self.config.min_volume
        )
        
        # Spread check
        bid = market_data.get('bid', 0)
        ask = market_data.get('ask', 0)
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
        
        if mid > 0:
            spread_pct = (ask - bid) / mid
            spread_ok = spread_pct <= self.config.max_bid_ask_spread_pct
        else:
            spread_ok = False
        
        # Strike availability check
        strike_ok = market_data.get('strike_exists', False)
        
        return liquidity_ok, spread_ok, strike_ok
    
    def _estimate_spread_cost(self,
                             spot: float,
                             strike: float,
                             market_data: Optional[Dict]) -> float:
        """
        Estimate bid-ask spread cost in basis points.
        
        **ROUGH ESTIMATE** - Real trading would need:
        - Real-time bid/ask data
        - Historical spread analysis
        - Liquidity-adjusted costs
        """
        if market_data is None or 'bid' not in market_data:
            # No data → Use conservative default
            return self.config.avg_spread_cost_pct * 10000  # Convert to bps
        
        bid = market_data['bid']
        ask = market_data['ask']
        
        if bid <= 0 or ask <= 0:
            return self.config.avg_spread_cost_pct * 10000
        
        mid = (bid + ask) / 2
        spread_pct = (ask - bid) / mid
        
        # Half the spread (assuming mid execution)
        return (spread_pct / 2) * 10000
    
    def _estimate_commission_cost(self) -> float:
        """
        Estimate commission cost in basis points.
        
        **ROUGH ESTIMATE** - Real costs vary by:
        - Broker (Interactive Brokers: $0.65/contract)
        - Contract size
        - Volume discounts
        - Regulatory fees
        """
        # Assume $0.65 per contract, ~$50 notional → ~130 bps
        # This is a PLACEHOLDER - real calculation needs contract value
        return 130
    
    def _generate_rationale(self,
                           signal_type: SignalType,
                           model_iv: float,
                           market_iv: float,
                           net_edge: float,
                           liquidity_ok: bool,
                           spread_ok: bool,
                           strike_ok: bool) -> str:
        """Generate human-readable signal rationale."""
        rationale = f"{signal_type.value.upper()}: "
        
        if signal_type == SignalType.NO_TRADE:
            rationale += "Insufficient edge (model IV ≈ market IV)"
            return rationale
        
        rationale += f"Model IV={model_iv:.3f} vs Market IV={market_iv:.3f}. "
        rationale += f"Net edge: {net_edge:.0f} bps after costs. "
        
        # Reality check warnings
        warnings_list = []
        if not liquidity_ok:
            warnings_list.append("LOW LIQUIDITY")
        if not spread_ok:
            warnings_list.append("WIDE SPREAD")
        if not strike_ok:
            warnings_list.append("STRIKE UNAVAILABLE")
        
        if warnings_list:
            rationale += f"⚠️ {', '.join(warnings_list)}"
        
        return rationale
    
    def filter_tradeable_signals(self) -> List[TradingSignal]:
        """
        Filter signals that pass all reality checks.
        
        Returns:
            List of signals with:
            - Positive net edge
            - Adequate liquidity
            - Reasonable spreads
            - Available strikes
        """
        tradeable = []
        
        for signal in self.signals:
            if (signal.net_edge_bps >= self.config.min_net_edge_bps and
                signal.liquidity_adequate and
                signal.spread_reasonable and
                signal.strike_available and
                signal.signal_type != SignalType.NO_TRADE):
                tradeable.append(signal)
        
        return tradeable
    
    def print_signal_report(self, signal: TradingSignal):
        """Print detailed signal report."""
        print("\n" + "="*70)
        print("TRADING SIGNAL REPORT (RESEARCH ONLY)")
        print("="*70)
        print(f"\nTimestamp: {signal.timestamp}")
        print(f"Signal: {signal.signal_type.value.upper()}")
        print(f"Confidence: {signal.confidence:.2%}")
        
        print(f"\n{'='*70}")
        print("VOLATILITY ANALYSIS")
        print("="*70)
        print(f"Model IV: {signal.model_iv:.4f} ({signal.model_iv*100:.2f}%)")
        print(f"Market IV: {signal.market_iv:.4f} ({signal.market_iv*100:.2f}%)")
        print(f"IV Differential: {(signal.model_iv - signal.market_iv)*100:.2f}%")
        print(f"Raw Edge: {signal.edge_bps:.0f} bps")
        
        print(f"\n{'='*70}")
        print("TRANSACTION COST ESTIMATES")
        print("="*70)
        print(f"Bid-Ask Spread: {signal.estimated_spread_cost_bps:.0f} bps")
        print(f"Commission: {signal.estimated_commission_bps:.0f} bps")
        print(f"Total Costs: {signal.estimated_total_cost_bps:.0f} bps")
        print(f"\n✓ NET EDGE: {signal.net_edge_bps:.0f} bps")
        
        print(f"\n{'='*70}")
        print("REALITY CHECKS")
        print("="*70)
        status = lambda x: "✓ PASS" if x else "✗ FAIL"
        print(f"Liquidity: {status(signal.liquidity_adequate)}")
        print(f"Spread: {status(signal.spread_reasonable)}")
        print(f"Strike Available: {status(signal.strike_available)}")
        
        print(f"\n{'='*70}")
        print("RATIONALE")
        print("="*70)
        print(signal.signal_rationale)
        
        print(f"\n{'='*70}")
        print("⚠️ DISCLAIMER")
        print("="*70)
        print("This is SIGNAL RESEARCH, not a trading recommendation.")
        print("Real trading requires:")
        print("  - Live market data and execution systems")
        print("  - Full transaction cost modeling")
        print("  - Risk management and position limits")
        print("  - Regulatory compliance")
        print("="*70 + "\n")


# ============================================================================
# STRESS TESTING
# ============================================================================

class SignalStressTester:
    """
    Stress test signals under adverse market conditions.
    
    Tests sensitivity to:
    - Wider bid-ask spreads
    - Higher transaction costs
    - Lower liquidity
    - Increased volatility
    """
    
    def __init__(self, generator: VolatilitySignalGenerator):
        self.generator = generator
    
    def stress_test_costs(self, signal: TradingSignal) -> Dict[str, float]:
        """
        Test signal under increased cost scenarios.
        
        Returns:
            Dict with net edge under various cost scenarios
        """
        base_edge = signal.edge_bps
        
        scenarios = {
            'base_case': signal.net_edge_bps,
            'spread_2x': base_edge - (2 * signal.estimated_spread_cost_bps) - signal.estimated_commission_bps,
            'spread_3x': base_edge - (3 * signal.estimated_spread_cost_bps) - signal.estimated_commission_bps,
            'commission_2x': base_edge - signal.estimated_spread_cost_bps - (2 * signal.estimated_commission_bps),
            'all_costs_2x': base_edge - (2 * signal.estimated_total_cost_bps),
        }
        
        return scenarios
    
    def stress_test_liquidity(self, 
                             signal: TradingSignal,
                             market_data: Dict) -> Dict[str, bool]:
        """
        Test signal under degraded liquidity conditions.
        
        Returns:
            Dict with tradeable status under various liquidity scenarios
        """
        oi = market_data.get('open_interest', 0)
        vol = market_data.get('volume', 0)
        
        scenarios = {
            'base_case': signal.liquidity_adequate,
            'oi_50pct_drop': oi * 0.5 >= self.generator.config.min_open_interest,
            'volume_50pct_drop': vol * 0.5 >= self.generator.config.min_volume,
            'both_50pct_drop': (oi * 0.5 >= self.generator.config.min_open_interest and
                               vol * 0.5 >= self.generator.config.min_volume),
        }
        
        return scenarios
    
    def print_stress_test_report(self, 
                                 signal: TradingSignal,
                                 market_data: Dict):
        """Print comprehensive stress test report."""
        print("\n" + "="*70)
        print("SIGNAL STRESS TEST REPORT")
        print("="*70)
        
        # Cost sensitivity
        cost_scenarios = self.stress_test_costs(signal)
        print("\nCOST SENSITIVITY (Net Edge in bps)")
        print("-"*70)
        for scenario, net_edge in cost_scenarios.items():
            status = "✓" if net_edge > 50 else "✗"
            print(f"{status} {scenario:20s}: {net_edge:>8.0f} bps")
        
        # Liquidity sensitivity
        liq_scenarios = self.stress_test_liquidity(signal, market_data)
        print("\nLIQUIDITY SENSITIVITY")
        print("-"*70)
        for scenario, tradeable in liq_scenarios.items():
            status = "✓ PASS" if tradeable else "✗ FAIL"
            print(f"{status:8s} {scenario}")
        
        print("\n" + "="*70 + "\n")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Demonstrate signal generation with reality checks.
    """
    
    print("="*70)
    print("TRADING SIGNAL GENERATION - RESEARCH DEMONSTRATION")
    print("="*70)
    print("\n⚠️ DISCLAIMER: This is SIGNAL RESEARCH, not trading advice.")
    print("Real trading requires live data, execution systems, and compliance.\n")
    
    # Initialize generator with conservative config
    config = RealityCheckConfig(
        min_open_interest=100,
        max_bid_ask_spread_pct=0.10,
        min_volume=20,
        min_net_edge_bps=50
    )
    generator = VolatilitySignalGenerator(config)
    
    # Example: Signal from calibrated SABR model
    spot = 682.85  # SPY current price
    strike = 690.0  # Slightly OTM call
    maturity = 0.25  # 3 months
    
    # Model predicts higher volatility than market
    model_iv = 0.25  # Model: 25% IV
    market_iv = 0.22  # Market: 22% IV
    
    # Mock market data (in reality, fetch from exchange)
    market_data = {
        'open_interest': 500,
        'volume': 100,
        'bid': 8.50,
        'ask': 8.70,
        'strike_exists': True
    }
    
    # Generate signal
    print("Generating signal...")
    signal = generator.generate_signal(
        spot=spot,
        strike=strike,
        maturity=maturity,
        model_iv=model_iv,
        market_iv=market_iv,
        market_data=market_data
    )
    
    # Print signal report
    generator.print_signal_report(signal)
    
    # Stress test
    stress_tester = SignalStressTester(generator)
    stress_tester.print_stress_test_report(signal, market_data)
    
    # Filter tradeable signals
    tradeable = generator.filter_tradeable_signals()
    print(f"Tradeable signals: {len(tradeable)} / {len(generator.signals)}")
