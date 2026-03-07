"""
Advanced Strategy Signals - Multi-leg Trades

Generates signals for complex option strategies:
- Vertical spreads (bull/bear)
- Calendar spreads (time decay)
- Volatility spreads (straddle/strangle)
- Skew trades (risk reversal)

⚠️ RESEARCH ONLY - Transaction costs partially modeled.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from signal_generator import (
    SignalType, TradingSignal, RealityCheckConfig, VolatilitySignalGenerator
)


@dataclass
class StrategyLeg:
    """Single leg of multi-leg strategy."""
    position: str  # "long" or "short"
    option_type: str  # "call" or "put"
    strike: float
    maturity: float
    quantity: int
    model_iv: float
    market_iv: float
    
    # Market data
    bid: float = 0.0
    ask: float = 0.0
    open_interest: int = 0
    volume: int = 0


@dataclass
class StrategySignal:
    """
    Multi-leg strategy signal with comprehensive analysis.
    """
    timestamp: datetime
    strategy_name: str
    strategy_type: str  # "spread", "calendar", "volatility"
    legs: List[StrategyLeg]
    
    # P&L analysis
    max_profit: float
    max_loss: float
    breakeven_points: List[float]
    
    # Edge analysis
    total_edge_bps: float
    total_cost_bps: float
    net_edge_bps: float
    
    # Reality checks
    all_strikes_available: bool
    all_liquidity_adequate: bool
    all_spreads_reasonable: bool
    
    # Greeks (portfolio Greeks)
    portfolio_delta: float
    portfolio_gamma: float
    portfolio_vega: float
    portfolio_theta: float
    
    confidence: float
    rationale: str


class StrategySignalGenerator:
    """
    Generate advanced multi-leg strategy signals.
    
    **STRATEGIES:**
    1. Vertical Spreads: Directional with limited risk
    2. Calendar Spreads: Time decay with volatility view
    3. Skew Trades: Exploit smile mispricing
    4. Volatility Spreads: Pure volatility play
    """
    
    def __init__(self, config: Optional[RealityCheckConfig] = None):
        self.config = config or RealityCheckConfig()
        self.signals: List[StrategySignal] = []
    
    # ========================================================================
    # VERTICAL SPREADS
    # ========================================================================
    
    def generate_bull_call_spread(self,
                                  spot: float,
                                  lower_strike: float,
                                  upper_strike: float,
                                  maturity: float,
                                  model_ivs: Dict[float, float],
                                  market_data: Dict[float, Dict]) -> StrategySignal:
        """
        Bull call spread: Long lower strike, short upper strike.
        
        **LOGIC:**
        - Bullish directional play
        - Limited profit: (upper_strike - lower_strike - net_debit)
        - Limited loss: net_debit
        - Breakeven: lower_strike + net_debit
        
        **SIGNAL:**
        - Generate if model predicts lower IV than market (sell vol)
        - Or if spot expected to rise modestly
        """
        legs = [
            self._create_leg("long", "call", lower_strike, maturity, 
                           model_ivs.get(lower_strike, 0), market_data.get(lower_strike, {})),
            self._create_leg("short", "call", upper_strike, maturity,
                           model_ivs.get(upper_strike, 0), market_data.get(upper_strike, {}))
        ]
        
        # P&L analysis
        lower_cost = market_data.get(lower_strike, {}).get('ask', 0)
        upper_credit = market_data.get(upper_strike, {}).get('bid', 0)
        net_debit = lower_cost - upper_credit
        
        max_profit = (upper_strike - lower_strike) - net_debit
        max_loss = net_debit
        breakeven = lower_strike + net_debit
        
        # Edge calculation
        edge_bps = self._calculate_strategy_edge(legs)
        cost_bps = self._calculate_strategy_cost(legs)
        net_edge = edge_bps - cost_bps
        
        # Reality checks
        strikes_ok, liquidity_ok, spreads_ok = self._check_strategy_reality(legs, market_data)
        
        # Portfolio Greeks (net position)
        delta = self._estimate_portfolio_delta(legs, spot)
        gamma = self._estimate_portfolio_gamma(legs, spot)
        vega = self._estimate_portfolio_vega(legs)
        theta = self._estimate_portfolio_theta(legs, maturity)
        
        confidence = self._calculate_strategy_confidence(
            net_edge, strikes_ok, liquidity_ok, spreads_ok
        )
        
        rationale = (
            f"Bull call spread: Expect moderate rise in {spot:.2f}. "
            f"Max profit ${max_profit:.2f}, max loss ${max_loss:.2f}, "
            f"breakeven ${breakeven:.2f}. Net edge: {net_edge:.0f} bps."
        )
        
        return StrategySignal(
            timestamp=datetime.now(),
            strategy_name="Bull Call Spread",
            strategy_type="spread",
            legs=legs,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_points=[breakeven],
            total_edge_bps=edge_bps,
            total_cost_bps=cost_bps,
            net_edge_bps=net_edge,
            all_strikes_available=strikes_ok,
            all_liquidity_adequate=liquidity_ok,
            all_spreads_reasonable=spreads_ok,
            portfolio_delta=delta,
            portfolio_gamma=gamma,
            portfolio_vega=vega,
            portfolio_theta=theta,
            confidence=confidence,
            rationale=rationale
        )
    
    # ========================================================================
    # CALENDAR SPREADS
    # ========================================================================
    
    def generate_calendar_spread(self,
                                 spot: float,
                                 strike: float,
                                 near_maturity: float,
                                 far_maturity: float,
                                 model_ivs: Dict[Tuple[float, float], float],
                                 market_data: Dict[Tuple[float, float], Dict]) -> StrategySignal:
        """
        Calendar spread: Short near-term, long far-term (same strike).
        
        **LOGIC:**
        - Profit from time decay differential
        - Near-term theta > far-term theta
        - Maximum profit at expiration if spot = strike
        
        **SIGNAL:**
        - Generate if term structure shows model edge
        - Profit if near-term decays faster than far-term
        """
        legs = [
            self._create_leg("short", "call", strike, near_maturity,
                           model_ivs.get((strike, near_maturity), 0),
                           market_data.get((strike, near_maturity), {})),
            self._create_leg("long", "call", strike, far_maturity,
                           model_ivs.get((strike, far_maturity), 0),
                           market_data.get((strike, far_maturity), {}))
        ]
        
        # P&L analysis
        near_credit = market_data.get((strike, near_maturity), {}).get('bid', 0)
        far_cost = market_data.get((strike, far_maturity), {}).get('ask', 0)
        net_debit = far_cost - near_credit
        
        max_profit = self._estimate_calendar_max_profit(strike, near_maturity, far_maturity)
        max_loss = net_debit
        breakeven = [strike]  # Simplified - actual has two breakevens
        
        edge_bps = self._calculate_strategy_edge(legs)
        cost_bps = self._calculate_strategy_cost(legs)
        net_edge = edge_bps - cost_bps
        
        strikes_ok, liquidity_ok, spreads_ok = self._check_strategy_reality(legs, market_data)
        
        delta = self._estimate_portfolio_delta(legs, spot)
        gamma = self._estimate_portfolio_gamma(legs, spot)
        vega = self._estimate_portfolio_vega(legs)
        theta = self._estimate_portfolio_theta(legs, (near_maturity + far_maturity) / 2)
        
        confidence = self._calculate_strategy_confidence(
            net_edge, strikes_ok, liquidity_ok, spreads_ok
        )
        
        rationale = (
            f"Calendar spread: Profit from time decay. "
            f"Near-term expires {near_maturity*365:.0f}d, far-term {far_maturity*365:.0f}d. "
            f"Max profit ~${max_profit:.2f}, max loss ${max_loss:.2f}. "
            f"Net edge: {net_edge:.0f} bps."
        )
        
        return StrategySignal(
            timestamp=datetime.now(),
            strategy_name="Calendar Spread",
            strategy_type="calendar",
            legs=legs,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_points=breakeven,
            total_edge_bps=edge_bps,
            total_cost_bps=cost_bps,
            net_edge_bps=net_edge,
            all_strikes_available=strikes_ok,
            all_liquidity_adequate=liquidity_ok,
            all_spreads_reasonable=spreads_ok,
            portfolio_delta=delta,
            portfolio_gamma=gamma,
            portfolio_vega=vega,
            portfolio_theta=theta,
            confidence=confidence,
            rationale=rationale
        )
    
    # ========================================================================
    # VOLATILITY TRADES
    # ========================================================================
    
    def generate_strangle_signal(self,
                                 spot: float,
                                 put_strike: float,
                                 call_strike: float,
                                 maturity: float,
                                 model_ivs: Dict[float, float],
                                 market_data: Dict[float, Dict],
                                 long_vol: bool = True) -> StrategySignal:
        """
        Strangle: OTM put + OTM call (long or short volatility).
        
        **LONG STRANGLE:**
        - Expect large move in either direction
        - Limited loss: net_debit
        - Unlimited profit potential
        - Breakevens: put_strike - net_debit, call_strike + net_debit
        
        **SHORT STRANGLE:**
        - Expect low volatility (range-bound)
        - Limited profit: net_credit
        - Unlimited loss potential (manage with stops)
        - Breakevens: same as long
        """
        position = "long" if long_vol else "short"
        
        legs = [
            self._create_leg(position, "put", put_strike, maturity,
                           model_ivs.get(put_strike, 0),
                           market_data.get(put_strike, {})),
            self._create_leg(position, "call", call_strike, maturity,
                           model_ivs.get(call_strike, 0),
                           market_data.get(call_strike, {}))
        ]
        
        # P&L analysis
        put_price = market_data.get(put_strike, {}).get('ask' if long_vol else 'bid', 0)
        call_price = market_data.get(call_strike, {}).get('ask' if long_vol else 'bid', 0)
        
        if long_vol:
            net_debit = put_price + call_price
            max_loss = net_debit
            max_profit = float('inf')  # Unlimited
            breakevens = [put_strike - net_debit, call_strike + net_debit]
        else:
            net_credit = put_price + call_price
            max_profit = net_credit
            max_loss = float('inf')  # Unlimited (need risk management)
            breakevens = [put_strike - net_credit, call_strike + net_credit]
        
        edge_bps = self._calculate_strategy_edge(legs)
        cost_bps = self._calculate_strategy_cost(legs)
        net_edge = edge_bps - cost_bps
        
        strikes_ok, liquidity_ok, spreads_ok = self._check_strategy_reality(legs, market_data)
        
        delta = self._estimate_portfolio_delta(legs, spot)
        gamma = self._estimate_portfolio_gamma(legs, spot)
        vega = self._estimate_portfolio_vega(legs)
        theta = self._estimate_portfolio_theta(legs, maturity)
        
        confidence = self._calculate_strategy_confidence(
            net_edge, strikes_ok, liquidity_ok, spreads_ok
        )
        
        vol_direction = "high" if long_vol else "low"
        rationale = (
            f"{'Long' if long_vol else 'Short'} strangle: Expect {vol_direction} volatility. "
            f"Breakevens: ${breakevens[0]:.2f} - ${breakevens[1]:.2f}. "
            f"Net edge: {net_edge:.0f} bps."
        )
        
        return StrategySignal(
            timestamp=datetime.now(),
            strategy_name=f"{'Long' if long_vol else 'Short'} Strangle",
            strategy_type="volatility",
            legs=legs,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_points=breakevens,
            total_edge_bps=edge_bps,
            total_cost_bps=cost_bps,
            net_edge_bps=net_edge,
            all_strikes_available=strikes_ok,
            all_liquidity_adequate=liquidity_ok,
            all_spreads_reasonable=spreads_ok,
            portfolio_delta=delta,
            portfolio_gamma=gamma,
            portfolio_vega=vega,
            portfolio_theta=theta,
            confidence=confidence,
            rationale=rationale
        )
    
    # ========================================================================
    # SKEW TRADES
    # ========================================================================
    
    def generate_risk_reversal(self,
                              spot: float,
                              put_strike: float,
                              call_strike: float,
                              maturity: float,
                              model_ivs: Dict[float, float],
                              market_data: Dict[float, Dict]) -> StrategySignal:
        """
        Risk reversal: Long call + short put (or vice versa).
        
        **LOGIC:**
        - Exploit skew mispricing
        - If put IV > call IV (typical downside skew), sell put / buy call
        - Synthetic long stock position with volatility edge
        
        **SIGNAL:**
        - Generate when model predicts skew mean reversion
        - Or directional view with volatility hedge
        """
        # Determine direction based on skew
        put_iv = model_ivs.get(put_strike, market_data.get(put_strike, {}).get('iv', 0))
        call_iv = model_ivs.get(call_strike, market_data.get(call_strike, {}).get('iv', 0))
        
        # Standard: long call, short put (bullish + sell downside skew)
        legs = [
            self._create_leg("long", "call", call_strike, maturity,
                           model_ivs.get(call_strike, 0),
                           market_data.get(call_strike, {})),
            self._create_leg("short", "put", put_strike, maturity,
                           model_ivs.get(put_strike, 0),
                           market_data.get(put_strike, {}))
        ]
        
        # P&L analysis
        call_cost = market_data.get(call_strike, {}).get('ask', 0)
        put_credit = market_data.get(put_strike, {}).get('bid', 0)
        net_debit = call_cost - put_credit
        
        # Simplified P&L (similar to synthetic long)
        max_profit = float('inf')  # Unlimited upside
        max_loss = put_strike + net_debit  # If spot → 0
        breakeven = call_strike + net_debit
        
        edge_bps = self._calculate_strategy_edge(legs)
        cost_bps = self._calculate_strategy_cost(legs)
        net_edge = edge_bps - cost_bps
        
        strikes_ok, liquidity_ok, spreads_ok = self._check_strategy_reality(legs, market_data)
        
        delta = self._estimate_portfolio_delta(legs, spot)
        gamma = self._estimate_portfolio_gamma(legs, spot)
        vega = self._estimate_portfolio_vega(legs)
        theta = self._estimate_portfolio_theta(legs, maturity)
        
        confidence = self._calculate_strategy_confidence(
            net_edge, strikes_ok, liquidity_ok, spreads_ok
        )
        
        rationale = (
            f"Risk reversal: Exploit skew mispricing. "
            f"Put IV={put_iv:.3f}, Call IV={call_iv:.3f}. "
            f"Breakeven: ${breakeven:.2f}. Net edge: {net_edge:.0f} bps."
        )
        
        return StrategySignal(
            timestamp=datetime.now(),
            strategy_name="Risk Reversal",
            strategy_type="spread",
            legs=legs,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_points=[breakeven],
            total_edge_bps=edge_bps,
            total_cost_bps=cost_bps,
            net_edge_bps=net_edge,
            all_strikes_available=strikes_ok,
            all_liquidity_adequate=liquidity_ok,
            all_spreads_reasonable=spreads_ok,
            portfolio_delta=delta,
            portfolio_gamma=gamma,
            portfolio_vega=vega,
            portfolio_theta=theta,
            confidence=confidence,
            rationale=rationale
        )
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _create_leg(self, position: str, option_type: str, strike: float,
                   maturity: float, model_iv: float, market_data: Dict) -> StrategyLeg:
        """Create strategy leg with market data."""
        return StrategyLeg(
            position=position,
            option_type=option_type,
            strike=strike,
            maturity=maturity,
            quantity=1,
            model_iv=model_iv,
            market_iv=market_data.get('iv', 0),
            bid=market_data.get('bid', 0),
            ask=market_data.get('ask', 0),
            open_interest=market_data.get('open_interest', 0),
            volume=market_data.get('volume', 0)
        )
    
    def _calculate_strategy_edge(self, legs: List[StrategyLeg]) -> float:
        """Calculate total edge across all legs in basis points."""
        total_edge = 0.0
        for leg in legs:
            iv_diff = leg.model_iv - leg.market_iv
            edge = iv_diff * 10000  # Convert to bps
            if leg.position == "short":
                edge = -edge  # Reverse for short positions
            total_edge += edge
        return total_edge
    
    def _calculate_strategy_cost(self, legs: List[StrategyLeg]) -> float:
        """Estimate total transaction costs in basis points."""
        total_cost = 0.0
        for leg in legs:
            # Spread cost
            if leg.bid > 0 and leg.ask > 0:
                mid = (leg.bid + leg.ask) / 2
                spread_pct = (leg.ask - leg.bid) / mid
                spread_cost = (spread_pct / 2) * 10000  # Half spread in bps
            else:
                spread_cost = 500  # Conservative default
            
            # Commission cost (per leg)
            commission_cost = 130  # ~$0.65 per contract
            
            total_cost += (spread_cost + commission_cost)
        
        return total_cost
    
    def _check_strategy_reality(self, legs: List[StrategyLeg],
                               market_data: Dict) -> Tuple[bool, bool, bool]:
        """Check reality for all legs."""
        strikes_ok = all(
            market_data.get(leg.strike, {}).get('strike_exists', False)
            for leg in legs
        )
        
        liquidity_ok = all(
            leg.open_interest >= self.config.min_open_interest and
            leg.volume >= self.config.min_volume
            for leg in legs
        )
        
        spreads_ok = all(
            ((leg.ask - leg.bid) / ((leg.bid + leg.ask) / 2) <= self.config.max_bid_ask_spread_pct
             if leg.bid > 0 and leg.ask > 0 else False)
            for leg in legs
        )
        
        return strikes_ok, liquidity_ok, spreads_ok
    
    def _estimate_portfolio_delta(self, legs: List[StrategyLeg], spot: float) -> float:
        """Rough estimate of portfolio delta."""
        total_delta = 0.0
        for leg in legs:
            # Simplified delta estimation
            moneyness = leg.strike / spot
            if leg.option_type == "call":
                delta = 0.5 if 0.95 <= moneyness <= 1.05 else (0.2 if moneyness > 1.05 else 0.8)
            else:
                delta = -0.5 if 0.95 <= moneyness <= 1.05 else (-0.2 if moneyness < 0.95 else -0.8)
            
            if leg.position == "short":
                delta = -delta
            
            total_delta += delta
        return total_delta
    
    def _estimate_portfolio_gamma(self, legs: List[StrategyLeg], spot: float) -> float:
        """Rough estimate of portfolio gamma."""
        total_gamma = 0.0
        for leg in legs:
            # Gamma highest ATM
            moneyness = leg.strike / spot
            gamma = 0.05 if 0.95 <= moneyness <= 1.05 else 0.01
            
            if leg.position == "short":
                gamma = -gamma
            
            total_gamma += gamma
        return total_gamma
    
    def _estimate_portfolio_vega(self, legs: List[StrategyLeg]) -> float:
        """Rough estimate of portfolio vega."""
        total_vega = 0.0
        for leg in legs:
            # Simplified vega (increases with time, ATM highest)
            vega = leg.maturity * 20  # Rough estimate
            
            if leg.position == "short":
                vega = -vega
            
            total_vega += vega
        return total_vega
    
    def _estimate_portfolio_theta(self, legs: List[StrategyLeg], maturity: float) -> float:
        """Rough estimate of portfolio theta (time decay)."""
        total_theta = 0.0
        for leg in legs:
            # Theta increases as expiration approaches
            theta = -5 / np.sqrt(maturity)  # Rough estimate
            
            if leg.position == "short":
                theta = -theta  # Short positions benefit from decay
            
            total_theta += theta
        return total_theta
    
    def _estimate_calendar_max_profit(self, strike: float,
                                     near_mat: float, far_mat: float) -> float:
        """Estimate maximum profit for calendar spread."""
        # Max profit occurs at near expiration if spot = strike
        # Rough estimate: difference in time value
        return strike * 0.05  # ~5% of strike (very rough)
    
    def _calculate_strategy_confidence(self, net_edge: float,
                                      strikes_ok: bool,
                                      liquidity_ok: bool,
                                      spreads_ok: bool) -> float:
        """Calculate strategy confidence [0, 1]."""
        # Base confidence from net edge
        if net_edge < 0:
            return 0.0
        
        edge_confidence = min(net_edge / 200, 1.0)  # Max at 200 bps
        
        # Penalize for failed reality checks
        reality_penalty = 0.0
        if not strikes_ok:
            reality_penalty += 0.4
        if not liquidity_ok:
            reality_penalty += 0.3
        if not spreads_ok:
            reality_penalty += 0.3
        
        return max(edge_confidence - reality_penalty, 0.0)
    
    def print_strategy_report(self, signal: StrategySignal):
        """Print comprehensive strategy report."""
        print("\n" + "="*70)
        print(f"STRATEGY SIGNAL: {signal.strategy_name.upper()}")
        print("="*70)
        print(f"Timestamp: {signal.timestamp}")
        print(f"Type: {signal.strategy_type}")
        print(f"Confidence: {signal.confidence:.2%}")
        
        print(f"\n{'='*70}")
        print("LEGS")
        print("="*70)
        for i, leg in enumerate(signal.legs, 1):
            print(f"\nLeg {i}: {leg.position.upper()} {leg.option_type.upper()}")
            print(f"  Strike: ${leg.strike:.2f}")
            print(f"  Maturity: {leg.maturity*365:.0f} days")
            print(f"  Model IV: {leg.model_iv:.4f}")
            print(f"  Market IV: {leg.market_iv:.4f}")
            print(f"  Bid/Ask: ${leg.bid:.2f} / ${leg.ask:.2f}")
            print(f"  OI: {leg.open_interest}, Volume: {leg.volume}")
        
        print(f"\n{'='*70}")
        print("P&L ANALYSIS")
        print("="*70)
        max_profit_str = f"${signal.max_profit:.2f}" if signal.max_profit != float('inf') else "Unlimited"
        max_loss_str = f"${signal.max_loss:.2f}" if signal.max_loss != float('inf') else "Unlimited"
        print(f"Max Profit: {max_profit_str}")
        print(f"Max Loss: {max_loss_str}")
        print(f"Breakeven(s): {', '.join(f'${b:.2f}' for b in signal.breakeven_points)}")
        
        print(f"\n{'='*70}")
        print("EDGE ANALYSIS")
        print("="*70)
        print(f"Total Edge: {signal.total_edge_bps:.0f} bps")
        print(f"Transaction Costs: {signal.total_cost_bps:.0f} bps")
        print(f"✓ NET EDGE: {signal.net_edge_bps:.0f} bps")
        
        print(f"\n{'='*70}")
        print("PORTFOLIO GREEKS")
        print("="*70)
        print(f"Delta: {signal.portfolio_delta:+.3f}")
        print(f"Gamma: {signal.portfolio_gamma:+.3f}")
        print(f"Vega: {signal.portfolio_vega:+.2f}")
        print(f"Theta: {signal.portfolio_theta:+.2f}")
        
        print(f"\n{'='*70}")
        print("REALITY CHECKS")
        print("="*70)
        status = lambda x: "✓ PASS" if x else "✗ FAIL"
        print(f"All Strikes Available: {status(signal.all_strikes_available)}")
        print(f"All Liquidity Adequate: {status(signal.all_liquidity_adequate)}")
        print(f"All Spreads Reasonable: {status(signal.all_spreads_reasonable)}")
        
        print(f"\n{'='*70}")
        print("RATIONALE")
        print("="*70)
        print(signal.rationale)
        
        print(f"\n{'='*70}")
        print("⚠️ DISCLAIMER")
        print("="*70)
        print("This is SIGNAL RESEARCH, not a trading recommendation.")
        print("Multi-leg strategies have complex risk profiles.")
        print("Requires live monitoring and risk management.")
        print("="*70 + "\n")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """Demonstrate advanced strategy signals."""
    
    print("="*70)
    print("ADVANCED STRATEGY SIGNALS - DEMONSTRATION")
    print("="*70)
    print("\n⚠️ RESEARCH ONLY - Not trading advice\n")
    
    generator = StrategySignalGenerator()
    
    # Example: Bull call spread on SPY
    spot = 682.85
    
    # Mock market data
    market_data = {
        680.0: {'bid': 10.5, 'ask': 10.7, 'open_interest': 500, 'volume': 100, 'iv': 0.22, 'strike_exists': True},
        690.0: {'bid': 5.2, 'ask': 5.4, 'open_interest': 400, 'volume': 80, 'iv': 0.24, 'strike_exists': True}
    }
    
    model_ivs = {680.0: 0.21, 690.0: 0.23}  # Model predicts slightly lower IV
    
    signal = generator.generate_bull_call_spread(
        spot=spot,
        lower_strike=680.0,
        upper_strike=690.0,
        maturity=0.25,
        model_ivs=model_ivs,
        market_data=market_data
    )
    
    generator.print_strategy_report(signal)
